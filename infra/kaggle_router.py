"""Kaggle resource router for a shared pool of accounts.

Given a directory of ``kaggle.json`` credential files (one per account), probe every
account's weekly accelerator quota and its currently active kernels, then answer the
practical question: *whose account can run my notebook right now?*

Everything is read-only against the Kaggle public API and uses only the stdlib.

Endpoints used (``https://www.kaggle.com/api/v1``, HTTP basic auth ``username:key``):

* ``GET /kernels/quota``                       -> weekly GPU/TPU seconds used, reserved, allowed
* ``GET /kernels/list?user=&sortBy=dateRun``   -> the account's most recently run kernels
* ``GET /kernels/status?userName=&kernelSlug=``-> per-kernel run status

``timeReserved`` in the quota payload is quota held by a session that is running *right
now*, so it is subtracted from the remaining budget as well as ``timeUsed``.

CLI
---
    python infra/kaggle_router.py --tokens-dir ~/kaggle_tokens            # report
    python infra/kaggle_router.py --tokens-dir DIR --min-gpu-hours 6      # filter
    python infra/kaggle_router.py --tokens-dir DIR --running              # active kernels only
    python infra/kaggle_router.py --tokens-dir DIR --json                 # machine readable

API keys are never logged, printed, or included in ``repr``/JSON output.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

API_BASE = "https://www.kaggle.com/api/v1"
USER_AGENT = "ga-research-kaggle-router/1.0"

#: Kernel states that mean the account is actively occupying a session.
ACTIVE_STATES = frozenset({"running", "queued", "cancelRequested"})

SECONDS_PER_HOUR = 3600.0


# --------------------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KaggleAccount:
    """One ``kaggle.json`` credential pair. ``key`` is kept out of repr/str."""

    username: str
    key: str = field(repr=False)
    label: str = ""
    path: Optional[str] = None

    @property
    def name(self) -> str:
        return self.label or self.username

    @classmethod
    def from_file(cls, path: str) -> "KaggleAccount":
        with open(path, "r") as fh:
            blob = json.load(fh)
        try:
            username, key = blob["username"], blob["key"]
        except KeyError as exc:  # pragma: no cover - malformed credential file
            raise ValueError("%s is missing %s" % (path, exc)) from exc
        label = os.path.splitext(os.path.basename(path))[0].strip()
        return cls(username=username, key=key, label=label, path=path)

    @property
    def auth_header(self) -> str:
        token = base64.b64encode(("%s:%s" % (self.username, self.key)).encode()).decode()
        return "Basic " + token

    def env(self) -> Dict[str, str]:
        """Env vars for handing this account to the ``kaggle`` CLI / library."""
        return {"KAGGLE_USERNAME": self.username, "KAGGLE_KEY": self.key}

    def write_kaggle_json(self, dest: str = "~/.kaggle/kaggle.json") -> str:
        """Install this account as the active ``kaggle.json`` (mode 600)."""
        dest = os.path.expanduser(dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as fh:
            json.dump({"username": self.username, "key": self.key}, fh)
        os.chmod(dest, 0o600)
        return dest


# --------------------------------------------------------------------------------------
# quota / status models
# --------------------------------------------------------------------------------------
@dataclass
class Quota:
    """Weekly accelerator budget for one resource kind (``gpu`` or ``tpu``)."""

    kind: str
    used_s: float = 0.0
    reserved_s: float = 0.0
    allowed_s: float = 0.0
    pay_to_scale: bool = False

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.allowed_s - self.used_s - self.reserved_s)

    @property
    def remaining_h(self) -> float:
        return self.remaining_s / SECONDS_PER_HOUR

    @property
    def used_h(self) -> float:
        return self.used_s / SECONDS_PER_HOUR

    @property
    def reserved_h(self) -> float:
        return self.reserved_s / SECONDS_PER_HOUR

    @property
    def allowed_h(self) -> float:
        return self.allowed_s / SECONDS_PER_HOUR

    @property
    def utilization(self) -> float:
        """Fraction of the weekly budget consumed or held, in ``[0, 1]``."""
        if self.allowed_s <= 0:
            return 1.0
        return min(1.0, (self.used_s + self.reserved_s) / self.allowed_s)

    @classmethod
    def from_payload(cls, kind: str, payload: Optional[Dict[str, Any]]) -> "Quota":
        payload = payload or {}

        def seconds(node: str) -> float:
            raw = payload.get(node) or {}
            return float(raw.get("seconds", 0) or 0) + float(raw.get("nanos", 0) or 0) / 1e9

        return cls(
            kind=kind,
            used_s=seconds("timeUsed"),
            reserved_s=seconds("timeReserved"),
            allowed_s=seconds("totalTimeAllowed"),
            pay_to_scale=bool(payload.get("isPayToScaleEnabled", False)),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "used_h": round(self.used_h, 2),
            "reserved_h": round(self.reserved_h, 2),
            "allowed_h": round(self.allowed_h, 2),
            "remaining_h": round(self.remaining_h, 2),
            "utilization": round(self.utilization, 3),
        }


@dataclass
class KernelRun:
    """A kernel of the account together with its last known run status."""

    ref: str
    status: str
    last_run: Optional[str] = None
    failure: str = ""

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATES

    @property
    def url(self) -> str:
        return "https://www.kaggle.com/code/" + self.ref

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ref": self.ref,
            "status": self.status,
            "last_run": self.last_run,
            "active": self.is_active,
            "url": self.url,
        }


@dataclass
class AccountStatus:
    """Probe result for a single account."""

    account: KaggleAccount
    ok: bool = False
    error: Optional[str] = None
    gpu: Quota = field(default_factory=lambda: Quota("gpu"))
    tpu: Quota = field(default_factory=lambda: Quota("tpu"))
    quota_refresh: Optional[str] = None
    kernels: List[KernelRun] = field(default_factory=list)
    kernels_checked: bool = False
    max_concurrent: int = 2

    @property
    def name(self) -> str:
        return self.account.name

    @property
    def username(self) -> str:
        return self.account.username

    @property
    def active_kernels(self) -> List[KernelRun]:
        return [k for k in self.kernels if k.is_active]

    @property
    def free_slots(self) -> int:
        """Session slots left, per the configured concurrency limit."""
        return max(0, self.max_concurrent - len(self.active_kernels))

    @property
    def busy(self) -> bool:
        """True if a session is running now (an active kernel, or held GPU/TPU quota)."""
        return bool(self.active_kernels) or self.gpu.reserved_s > 0 or self.tpu.reserved_s > 0

    def quota(self, resource: str) -> Quota:
        resource = resource.lower()
        if resource == "gpu":
            return self.gpu
        if resource == "tpu":
            return self.tpu
        raise ValueError("resource must be 'gpu' or 'tpu', got %r" % resource)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.name,
            "username": self.username,
            "ok": self.ok,
            "error": self.error,
            "gpu": self.gpu.as_dict(),
            "tpu": self.tpu.as_dict(),
            "quota_refresh": self.quota_refresh,
            "busy": self.busy,
            "free_slots": self.free_slots if self.kernels_checked else None,
            "active_kernels": [k.as_dict() for k in self.active_kernels],
            "kernels_checked": self.kernels_checked,
        }


# --------------------------------------------------------------------------------------
# router
# --------------------------------------------------------------------------------------
class KaggleRouter:
    """Probe a pool of Kaggle accounts and route work to the ones with spare capacity.

    >>> router = KaggleRouter.from_dir("~/kaggle_tokens")
    >>> statuses = router.probe_all()                     # doctest: +SKIP
    >>> free = router.available(min_gpu_hours=6)          # doctest: +SKIP
    >>> print(router.report())                            # doctest: +SKIP
    """

    def __init__(
        self,
        accounts: Sequence[KaggleAccount],
        timeout: float = 30.0,
        workers: int = 8,
        retries: int = 2,
        max_concurrent: int = 2,
    ) -> None:
        if not accounts:
            raise ValueError("no Kaggle accounts supplied")
        self.accounts = list(accounts)
        self.timeout = timeout
        self.workers = max(1, workers)
        self.retries = max(0, retries)
        self.max_concurrent = max_concurrent
        self._statuses: List[AccountStatus] = []

    # -- construction ------------------------------------------------------------------
    @classmethod
    def from_files(cls, paths: Iterable[str], **kwargs: Any) -> "KaggleRouter":
        return cls([KaggleAccount.from_file(os.path.expanduser(p)) for p in paths], **kwargs)

    @classmethod
    def from_dir(cls, directory: str, pattern: str = "*.json", **kwargs: Any) -> "KaggleRouter":
        directory = os.path.expanduser(directory)
        paths = sorted(glob.glob(os.path.join(directory, pattern)))
        if not paths:
            raise FileNotFoundError("no %s credential files in %s" % (pattern, directory))
        accounts, broken = [], []
        for path in paths:
            try:
                accounts.append(KaggleAccount.from_file(path))
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                broken.append("%s (%s)" % (os.path.basename(path), exc))
        if broken:
            print("skipped unreadable credential files: %s" % ", ".join(broken), file=sys.stderr)
        return cls(accounts, **kwargs)

    @classmethod
    def from_env(cls, var: str = "KAGGLE_TOKENS_DIR", **kwargs: Any) -> "KaggleRouter":
        directory = os.environ.get(var)
        if not directory:
            raise RuntimeError("%s is not set" % var)
        return cls.from_dir(directory, **kwargs)

    # -- HTTP --------------------------------------------------------------------------
    def request(
        self,
        account: KaggleAccount,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        **params: Any
    ) -> Any:
        """Authenticated call against the Kaggle v1 API. GET unless ``body`` is given.

        Raises ``PermissionError`` on 401/403, ``FileNotFoundError`` on 404, and
        ``RuntimeError`` with the server's message on other failures.
        """
        url = API_BASE + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {"Authorization": account.auth_header, "User-Agent": USER_AGENT}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers)

        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode()[:400]
                except Exception:  # pragma: no cover - body already consumed
                    pass
                if exc.code in (401, 403):
                    raise PermissionError("HTTP %s %s" % (exc.code, detail))
                if exc.code == 404:
                    raise FileNotFoundError("HTTP 404 %s %s" % (path, detail))
                if exc.code < 500:
                    raise RuntimeError("HTTP %s %s %s" % (exc.code, path, detail))
                last = exc
            except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(str(last))

    def _get(self, account: KaggleAccount, path: str, **params: Any) -> Any:
        return self.request(account, path, **params)

    # -- probing -----------------------------------------------------------------------
    def probe(
        self, account: KaggleAccount, check_kernels: bool = True, recent: int = 10
    ) -> AccountStatus:
        """Probe one account: quota first, then the status of its recent kernels."""
        status = AccountStatus(account=account, max_concurrent=self.max_concurrent)
        try:
            payload = self._get(account, "/kernels/quota")
        except Exception as exc:
            status.error = "%s: %s" % (type(exc).__name__, exc)
            return status

        status.ok = True
        status.gpu = Quota.from_payload("gpu", payload.get("gpuQuota"))
        status.tpu = Quota.from_payload("tpu", payload.get("tpuQuota"))
        status.quota_refresh = payload.get("quotaRefreshTime")

        if not check_kernels or recent <= 0:
            return status

        try:
            listing = self._get(
                account,
                "/kernels/list",
                user=account.username,
                page_size=recent,
                sortBy="dateRun",
            )
        except Exception as exc:
            status.error = "quota ok, kernel listing failed (%s)" % exc
            return status

        for item in listing or []:
            ref = item.get("ref") or ""
            if "/" not in ref:
                continue
            owner, slug = ref.split("/", 1)
            try:
                run = self._get(account, "/kernels/status", userName=owner, kernelSlug=slug)
            except FileNotFoundError:
                continue  # kernel has never been run
            except Exception:
                continue
            status.kernels.append(
                KernelRun(
                    ref=ref,
                    status=run.get("status", "unknown"),
                    last_run=item.get("lastRunTime"),
                    failure=run.get("failureMessage", "") or "",
                )
            )
        status.kernels_checked = True
        return status

    def probe_all(self, check_kernels: bool = True, recent: int = 10) -> List[AccountStatus]:
        """Probe every account in parallel (one thread per account)."""
        with ThreadPoolExecutor(max_workers=min(self.workers, len(self.accounts))) as pool:
            self._statuses = list(
                pool.map(lambda acc: self.probe(acc, check_kernels, recent), self.accounts)
            )
        return self._statuses

    @property
    def statuses(self) -> List[AccountStatus]:
        return self._statuses or self.probe_all()

    # -- routing -----------------------------------------------------------------------
    def available(
        self,
        min_gpu_hours: float = 1.0,
        resource: str = "gpu",
        require_idle: bool = False,
        require_free_slot: bool = False,
    ) -> List[AccountStatus]:
        """Accounts with spare capacity, richest in remaining quota first.

        Args:
            min_gpu_hours: minimum remaining hours of ``resource`` to qualify.
            resource: ``"gpu"`` or ``"tpu"``.
            require_idle: only accounts with no running session at all.
            require_free_slot: only accounts below the concurrency limit.
        """
        out = []
        for status in self.statuses:
            if not status.ok:
                continue
            if status.quota(resource).remaining_h < min_gpu_hours:
                continue
            if require_idle and status.busy:
                continue
            if require_free_slot and status.kernels_checked and status.free_slots <= 0:
                continue
            out.append(status)
        return sorted(out, key=lambda s: s.quota(resource).remaining_h, reverse=True)

    def best(self, **kwargs: Any) -> Optional[AccountStatus]:
        """The single most idle / quota-rich account, or ``None`` if the pool is exhausted."""
        candidates = self.available(**kwargs)
        return candidates[0] if candidates else None

    def running(self) -> List[KernelRun]:
        """Every kernel that is running or queued across the whole pool."""
        return [k for s in self.statuses for k in s.active_kernels]

    def running_by_account(self) -> Dict[str, List[KernelRun]]:
        return {s.name: s.active_kernels for s in self.statuses if s.active_kernels}

    def total_remaining_hours(self, resource: str = "gpu") -> float:
        return sum(s.quota(resource).remaining_h for s in self.statuses if s.ok)

    # -- reporting ---------------------------------------------------------------------
    def report(self, resource: str = "gpu") -> str:
        statuses = self.statuses
        rows = [("ACCOUNT", "KAGGLE USER", "GPU LEFT", "TPU LEFT", "RUNNING", "STATE")]
        for status in sorted(
            statuses, key=lambda s: (not s.ok, -s.quota(resource).remaining_h)
        ):
            if not status.ok:
                rows.append((status.name, status.username, "-", "-", "-", "AUTH FAIL"))
                continue
            active = status.active_kernels
            if active:
                state = "BUSY"
            elif status.quota(resource).remaining_h <= 0:
                state = "NO QUOTA"
            else:
                state = "FREE"
            rows.append((
                status.name,
                status.username,
                "%5.1f/%.0fh" % (status.gpu.remaining_h, status.gpu.allowed_h),
                "%5.1f/%.0fh" % (status.tpu.remaining_h, status.tpu.allowed_h),
                str(len(active)) if status.kernels_checked else "?",
                state,
            ))
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        lines = []
        for idx, row in enumerate(rows):
            lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
            if idx == 0:
                lines.append("-" * len(lines[0]))

        refresh = next((s.quota_refresh for s in statuses if s.quota_refresh), None)
        lines.append("")
        lines.append(
            "pool: %d/%d accounts reachable, %.1f GPU-hours and %.1f TPU-hours spare%s"
            % (
                sum(1 for s in statuses if s.ok),
                len(statuses),
                self.total_remaining_hours("gpu"),
                self.total_remaining_hours("tpu"),
                (", quota resets %s" % refresh) if refresh else "",
            )
        )
        active = self.running_by_account()
        if active:
            lines.append("")
            lines.append("active kernels:")
            for name, kernels in active.items():
                for kernel in kernels:
                    lines.append("  %-28s %-10s %s" % (name, kernel.status, kernel.url))
        elif any(s.kernels_checked for s in statuses):
            lines.append("no kernels running or queued anywhere in the pool")
        else:
            lines.append("kernel states not checked (--no-kernels); quota figures only")
        for status in statuses:
            if status.error:
                lines.append("  ! %s: %s" % (status.name, status.error))
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accounts": [s.as_dict() for s in self.statuses],
            "gpu_hours_spare": round(self.total_remaining_hours("gpu"), 2),
            "tpu_hours_spare": round(self.total_remaining_hours("tpu"), 2),
            "running": [k.as_dict() for k in self.running()],
        }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route work across a pool of Kaggle accounts.")
    parser.add_argument(
        "--tokens-dir",
        default=os.environ.get("KAGGLE_TOKENS_DIR"),
        help="directory of kaggle.json files (default: $KAGGLE_TOKENS_DIR)",
    )
    parser.add_argument("--pattern", default="*.json", help="credential filename glob")
    parser.add_argument("--resource", choices=("gpu", "tpu"), default="gpu")
    parser.add_argument(
        "--min-gpu-hours",
        type=float,
        default=0.0,
        help="only list accounts with at least this many hours of the chosen resource left",
    )
    parser.add_argument("--recent", type=int, default=10, help="recent kernels to status-check")
    parser.add_argument("--no-kernels", action="store_true", help="quota only, skip kernel checks")
    parser.add_argument("--idle-only", action="store_true", help="exclude accounts with a session")
    parser.add_argument("--running", action="store_true", help="print only active kernels")
    parser.add_argument("--best", action="store_true", help="print the single best account")
    parser.add_argument("--max-concurrent", type=int, default=2, help="session slots per account")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.tokens_dir:
        parser.error("--tokens-dir is required (or set KAGGLE_TOKENS_DIR)")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    router = KaggleRouter.from_dir(
        args.tokens_dir,
        pattern=args.pattern,
        workers=args.workers,
        max_concurrent=args.max_concurrent,
    )
    router.probe_all(check_kernels=not args.no_kernels, recent=args.recent)

    if args.json:
        print(json.dumps(router.as_dict(), indent=2))
        return 0

    if args.running:
        active = router.running_by_account()
        if not active:
            print("no kernels running or queued across %d accounts" % len(router.accounts))
            return 0
        for name, kernels in active.items():
            for kernel in kernels:
                print("%-28s %-10s %s" % (name, kernel.status, kernel.url))
        return 0

    if args.best:
        pick = router.best(
            min_gpu_hours=args.min_gpu_hours,
            resource=args.resource,
            require_idle=args.idle_only,
        )
        if pick is None:
            print("no account satisfies the request")
            return 1
        print(
            "%s (%s): %.1f %s-hours left, %d active kernel(s)"
            % (
                pick.name,
                pick.username,
                pick.quota(args.resource).remaining_h,
                args.resource,
                len(pick.active_kernels),
            )
        )
        return 0

    print(router.report(resource=args.resource))
    if args.min_gpu_hours or args.idle_only:
        picks = router.available(
            min_gpu_hours=args.min_gpu_hours,
            resource=args.resource,
            require_idle=args.idle_only,
        )
        print("")
        print(
            "candidates (>=%.1f %s-hours%s): %s"
            % (
                args.min_gpu_hours,
                args.resource,
                ", idle only" if args.idle_only else "",
                ", ".join("%s (%.1fh)" % (p.name, p.quota(args.resource).remaining_h) for p in picks)
                or "none",
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
