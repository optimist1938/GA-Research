"""Fair sharing of a pool of shared accelerator accounts, driven by the official Kaggle CLI.

Given a directory of per-account credentials, report each account's weekly accelerator
quota and its currently active kernels, then answer: *whose account can run my notebook
right now?*

Everything goes through the ``kaggle`` CLI (Kaggle/kaggle-cli, 2.x) rather than raw HTTP:

* ``kaggle quota --format json``                  -> weekly GPU/TPU hours used / remaining
* ``kaggle kernels list --mine --format json``    -> the account's kernels, newest run first
* ``kaggle kernels status <owner/slug>``          -> that kernel's latest run state
* ``kaggle config view``                          -> the username behind a token

Credentials
-----------
Two layouts are supported, and may be mixed in one directory:

* **new style** (preferred): one directory per account holding a ``token`` file with an
  access token, passed to the CLI as ``KAGGLE_API_TOKEN``; ``access_token`` (the name
  ``kaggle auth login`` uses under ``~/.kaggle``) is accepted too, so a symlink to
  ``~/.kaggle`` can be one of the accounts::

      tokens/account5_GrigoryZ/token
      tokens/account1_me -> ~/.kaggle        (holds access_token)

* **legacy**: a ``kaggle.json`` with ``{"username": ..., "key": ...}``, passed as
  ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``::

      tokens/account5_GrigoryZ.json

Each CLI call runs with its own ``KAGGLE_CONFIG_DIR``, so accounts never read each
other's state or the developer's own ``~/.kaggle``.

CLI
---
    python gpu_pool/router.py --tokens-dir ~/kaggle_tokens            # report
    python gpu_pool/router.py --tokens-dir DIR --min-gpu-hours 6      # filter
    python gpu_pool/router.py --tokens-dir DIR --running              # active kernels
    python gpu_pool/router.py --tokens-dir DIR --json                 # machine readable

Tokens are never logged, printed, or included in ``repr``/JSON output.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: File names an account directory may hold its access token under, in order of preference:
#: ``token`` is this tool's layout, ``access_token`` is what ``kaggle auth login`` writes.
TOKEN_FILES = ("token", "access_token")

#: Kernel states that mean the account is actively occupying a session.
ACTIVE_STATES = frozenset({"running", "queued", "cancelRequested"})

#: ``kaggle kernels status`` prints ``KernelWorkerStatus.QUEUED``; map to our vocabulary.
_STATUS_ALIASES = {
    "queued": "queued",
    "running": "running",
    "complete": "complete",
    "error": "error",
    "cancelrequested": "cancelRequested",
    "cancelacknowledged": "cancelAcknowledged",
    "cancelled": "cancelAcknowledged",
    "canceled": "cancelAcknowledged",
}

SECONDS_PER_HOUR = 3600.0

#: Kaggle ``machine_shape`` values -> short label used in reports. Quota is pooled across all
#: GPU types (``kaggle quota`` has one GPU row), so these describe *which hardware a run got*,
#: never a separate budget. There is no value for the editor's "GPU T4 x2"
#: (Kaggle/kaggle-cli#1196), and TPU shapes are accepted but provision a non-TPU image
#: (Kaggle/kaggle-cli#1197).
SHAPE_LABELS = {
    "NvidiaTeslaT4": "t4",
    "NvidiaTeslaP100": "p100",
    "NvidiaL4": "l4",
    "NvidiaRtxPro6000": "rtx6000",
    "Tpu1VmV38": "tpu-v3",
    "TpuV5E8": "tpu-v5e",
}


#: ``kaggle kernels pull -m`` writes the literal string "None" into ``machine_shape`` for a
#: CPU kernel (a str(None) leak), so these all mean "no accelerator".
_NO_SHAPE = frozenset({"", "none", "null"})


#: Kaggle only offers these accelerators outside a competition. Everything else (L4, RTX Pro
#: 6000, TPU) is reachable only from a kernel attached to a competition the account has entered.
OPEN_ACCELERATORS = frozenset({"cpu", "t4", "p100"})


def normalize_accelerator(value: str) -> str:
    """Accept a short label (``"l4"``) or a Kaggle shape (``"NvidiaL4"``) -> short label."""
    text = (value or "").strip()
    return shape_label(text) if text in SHAPE_LABELS else (text.lower() or "cpu")


def requires_competition(accelerator: str) -> bool:
    """True if ``accelerator`` is only available inside a competition."""
    return normalize_accelerator(accelerator) not in OPEN_ACCELERATORS


def shape_label(shape: Optional[str]) -> str:
    """``"NvidiaL4"`` -> ``"l4"``; unknown shapes pass through, no accelerator -> ``"cpu"``."""
    if not shape or str(shape).strip().lower() in _NO_SHAPE:
        return "cpu"
    return SHAPE_LABELS.get(shape, shape)


def default_cli() -> str:
    """The ``kaggle`` executable: ``$KAGGLE_CLI``, else the first one on PATH."""
    return os.environ.get("KAGGLE_CLI") or shutil.which("kaggle") or "kaggle"


def _parse_hours(value: Any) -> float:
    """``"0.14h"`` / ``"29.86h"`` / ``12`` -> float hours."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else 0.0


def _normalize_status(raw: str) -> str:
    """``'x/y has status "KernelWorkerStatus.QUEUED"'`` -> ``'queued'``."""
    text = (raw or "").strip()
    quoted = re.findall(r'"([^"]+)"', text)
    token = quoted[-1] if quoted else text
    token = token.rsplit(".", 1)[-1]                      # KernelWorkerStatus.QUEUED -> QUEUED
    key = token.replace("_", "").replace(" ", "").lower()
    return _STATUS_ALIASES.get(key, token or "unknown")


_PROGRESS_LINE = re.compile(r"it/s|%\||^\s*\d+%|^[\s|\u2500-\u259f\u2588]+$")


def filter_progress(text: str, keep_last: int = 0) -> List[str]:
    """Drop progress-bar redraws and duplicate lines, keeping the informative ones.

    A pose training log is almost entirely tqdm output (41k lines -> ~130 real ones), which
    buries the timings, losses and metrics. ``keep_last`` trims to the most recent N kept lines.
    """
    seen, kept = set(), []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or _PROGRESS_LINE.search(line) or line in seen:
            continue
        seen.add(line)
        kept.append(line)
    return kept[-keep_last:] if keep_last else kept


class CliError(RuntimeError):
    """A ``kaggle`` invocation failed. Carries the command's own stderr."""


# --------------------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PoolAccount:
    """One pool account. The secret is kept out of repr/str and JSON."""

    label: str
    token: Optional[str] = field(default=None, repr=False)
    legacy_username: Optional[str] = None
    legacy_key: Optional[str] = field(default=None, repr=False)
    path: Optional[str] = None
    #: Resolved from the credential via ``kaggle config view``; see ``PoolRouter.username``.
    resolved_username: Optional[str] = None

    @property
    def name(self) -> str:
        return self.label

    @property
    def username(self) -> str:
        """Best known Kaggle username: resolved, else the legacy one, else the label."""
        return self.resolved_username or self.legacy_username or self.label

    @property
    def uses_token(self) -> bool:
        return bool(self.token)

    def env(self) -> Dict[str, str]:
        """Credential environment for a ``kaggle`` subprocess."""
        if self.token:
            return {"KAGGLE_API_TOKEN": self.token}
        if self.legacy_username and self.legacy_key:
            return {"KAGGLE_USERNAME": self.legacy_username, "KAGGLE_KEY": self.legacy_key}
        raise ValueError("account %s has no usable credential" % self.label)

    # -- loading ----------------------------------------------------------------------
    @classmethod
    def from_token_dir(cls, directory: str) -> "PoolAccount":
        """An account directory holding a ``token`` or ``access_token`` file (new-style access
        token; the latter is what ``kaggle auth login`` writes under ``~/.kaggle``)."""
        for name in TOKEN_FILES:
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                break
        else:
            raise ValueError("%s holds none of %s" % (directory, ", ".join(TOKEN_FILES)))
        with open(path, "r") as fh:
            token = fh.read().strip()
        if not token:
            raise ValueError("%s is empty" % path)
        return cls(label=os.path.basename(directory.rstrip("/")).strip(), token=token, path=path)

    @classmethod
    def from_json(cls, path: str) -> "PoolAccount":
        """A legacy ``kaggle.json`` with ``username`` and ``key``."""
        with open(path, "r") as fh:
            blob = json.load(fh)
        try:
            username, key = blob["username"], blob["key"]
        except KeyError as exc:
            raise ValueError("%s is missing %s" % (path, exc)) from exc
        return cls(
            label=os.path.splitext(os.path.basename(path))[0].strip(),
            legacy_username=username,
            legacy_key=key,
            path=path,
        )


# --------------------------------------------------------------------------------------
# quota / status models
# --------------------------------------------------------------------------------------
@dataclass
class Quota:
    """Weekly accelerator budget for one resource kind (``gpu`` or ``tpu``)."""

    kind: str
    used_h: float = 0.0
    remaining_h: float = 0.0
    allowed_h: float = 0.0
    refresh_at: Optional[str] = None

    @property
    def utilization(self) -> float:
        """Fraction of the weekly budget consumed, in ``[0, 1]``."""
        if self.allowed_h <= 0:
            return 1.0
        return min(1.0, max(0.0, self.used_h / self.allowed_h))

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Quota":
        return cls(
            kind=str(row.get("resource", "")).lower() or "gpu",
            used_h=_parse_hours(row.get("used")),
            remaining_h=_parse_hours(row.get("remaining")),
            allowed_h=_parse_hours(row.get("total")),
            refresh_at=row.get("refreshAt"),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "used_h": round(self.used_h, 2),
            "remaining_h": round(self.remaining_h, 2),
            "allowed_h": round(self.allowed_h, 2),
            "utilization": round(self.utilization, 3),
        }


@dataclass
class KernelRun:
    """A kernel of the account together with its latest run status."""

    ref: str
    status: str
    last_run: Optional[str] = None
    #: ``machine_shape`` the kernel requests, when accelerator info was collected.
    machine_shape: Optional[str] = None

    @property
    def accelerator(self) -> Optional[str]:
        """Short label for the requested hardware, or ``None`` if not collected."""
        return shape_label(self.machine_shape) if self.machine_shape is not None else None

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
            "accelerator": self.accelerator,
            "machine_shape": self.machine_shape,
            "url": self.url,
        }


@dataclass
class AccountStatus:
    """Probe result for a single account."""

    account: PoolAccount
    ok: bool = False
    error: Optional[str] = None
    gpu: Quota = field(default_factory=lambda: Quota("gpu"))
    tpu: Quota = field(default_factory=lambda: Quota("tpu"))
    kernels: List[KernelRun] = field(default_factory=list)
    kernels_checked: bool = False
    #: Competitions the account has entered (``kaggle competitions list --group entered``).
    competitions: List[str] = field(default_factory=list)
    competitions_checked: bool = False
    max_concurrent: int = 2

    @property
    def name(self) -> str:
        return self.account.name

    @property
    def username(self) -> str:
        return self.account.username

    @property
    def quota_refresh(self) -> Optional[str]:
        return self.gpu.refresh_at or self.tpu.refresh_at

    @property
    def active_kernels(self) -> List[KernelRun]:
        return [k for k in self.kernels if k.is_active]

    @property
    def free_slots(self) -> int:
        """Session slots left. Kaggle caps concurrent batch GPU sessions at 2 and says so
        only by refusing a push ("Maximum batch GPU session count of 2 reached")."""
        return max(0, self.max_concurrent - len(self.active_kernels))

    @property
    def busy(self) -> bool:
        return bool(self.active_kernels)

    @property
    def accelerators_seen(self) -> Dict[str, int]:
        """How many of the inspected kernels asked for each accelerator.

        Evidence of what this account actually gets from Kaggle. It is not a quota: every GPU
        type draws on the same weekly GPU budget.
        """
        counts: Dict[str, int] = {}
        for kernel in self.kernels:
            if kernel.machine_shape is None:
                continue
            counts[kernel.accelerator] = counts.get(kernel.accelerator, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def active_accelerators(self) -> List[str]:
        """Accelerators the account's running/queued kernels are holding."""
        return [k.accelerator for k in self.active_kernels if k.accelerator]

    def has_run_on(self, accelerator: str) -> bool:
        """True if an inspected kernel of this account requested ``accelerator``."""
        return normalize_accelerator(accelerator) in self.accelerators_seen

    def can_use(self, accelerator: str) -> bool:
        """Whether this account can reach ``accelerator`` at all.

        T4, P100 and CPU are always available. L4 / RTX Pro 6000 / TPU only exist inside a
        competition, so they need at least one entered competition -- and the kernel must cite
        it in ``competition_sources``. Unknown when competitions were not collected, in which
        case observed history is the fallback answer.
        """
        if not requires_competition(accelerator):
            return True
        if self.competitions_checked:
            return bool(self.competitions)
        return self.has_run_on(accelerator)

    @property
    def premium_accelerators(self) -> bool:
        """True if competition-gated accelerators (L4, RTX Pro 6000, TPU) are reachable."""
        return self.can_use("l4")

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
            "accelerators_seen": self.accelerators_seen,
            "active_accelerators": self.active_accelerators,
            "competitions": list(self.competitions),
            "competitions_checked": self.competitions_checked,
            "premium_accelerators": self.premium_accelerators,
        }


# --------------------------------------------------------------------------------------
# router
# --------------------------------------------------------------------------------------
class PoolRouter:
    """Probe a pool of Kaggle accounts through the CLI and route work to the free ones.

    >>> router = PoolRouter.from_dir("~/kaggle_tokens")
    >>> router.probe_all()                                 # doctest: +SKIP
    >>> router.best(min_gpu_hours=6, require_idle=True)     # doctest: +SKIP
    """

    def __init__(
        self,
        accounts: Sequence[PoolAccount],
        cli: Optional[str] = None,
        timeout: float = 120.0,
        workers: int = 8,
        max_concurrent: int = 2,
    ) -> None:
        if not accounts:
            raise ValueError("no Kaggle accounts supplied")
        self.accounts = list(accounts)
        self.cli = cli or default_cli()
        self.timeout = timeout
        self.workers = max(1, workers)
        self.max_concurrent = max_concurrent
        self._statuses: List[AccountStatus] = []

    # -- construction ------------------------------------------------------------------
    @classmethod
    def from_dir(cls, directory: str, **kwargs: Any) -> "PoolRouter":
        """Load every account in ``directory``: ``<account>/token`` (or ``access_token``) dirs
        and ``*.json`` files."""
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            raise FileNotFoundError("%s is not a directory" % directory)

        accounts: List[PoolAccount] = []
        broken: List[str] = []
        for entry in sorted(os.listdir(directory)):
            path = os.path.join(directory, entry)
            try:
                if os.path.isdir(path) and any(
                    os.path.isfile(os.path.join(path, name)) for name in TOKEN_FILES
                ):
                    accounts.append(PoolAccount.from_token_dir(path))
                elif entry.lower().endswith(".json") and not entry.startswith("."):
                    accounts.append(PoolAccount.from_json(path))
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                broken.append("%s (%s)" % (entry, exc))
        if broken:
            print("skipped unusable credentials: %s" % ", ".join(broken), file=sys.stderr)
        if not accounts:
            raise FileNotFoundError(
                "no credentials in %s (expected <account>/token or <account>/access_token dirs, "
                "or *.json)" % directory
            )
        return cls(accounts, **kwargs)

    @classmethod
    def from_env(cls, var: str = "KAGGLE_TOKENS_DIR", **kwargs: Any) -> "PoolRouter":
        directory = os.environ.get(var)
        if not directory:
            raise RuntimeError("%s is not set" % var)
        return cls.from_dir(directory, **kwargs)

    # -- CLI ---------------------------------------------------------------------------
    def run_cli(
        self,
        account: PoolAccount,
        *args: str,
        check: bool = True,
        timeout: Optional[float] = None,
        partial_on_timeout: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run ``kaggle <args>`` as ``account``, isolated from other accounts' config.

        ``partial_on_timeout`` returns whatever was printed before the deadline instead of
        raising, for streaming commands such as ``kernels logs --follow`` that never exit.
        """
        with tempfile.TemporaryDirectory(prefix="kaggle-cfg-") as config_dir:
            env = dict(os.environ)
            # Drop any ambient credential so only this account's can apply.
            for stale in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
                env.pop(stale, None)
            env.update(account.env())
            env["KAGGLE_CONFIG_DIR"] = config_dir
            try:
                proc = subprocess.run(
                    [self.cli, *args],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout or self.timeout,
                )
            except FileNotFoundError as exc:
                raise CliError(
                    "kaggle CLI not found at %r; pip install kaggle, or set $KAGGLE_CLI" % self.cli
                ) from exc
            except subprocess.TimeoutExpired as exc:
                if partial_on_timeout:
                    def _text(blob: Any) -> str:
                        if isinstance(blob, bytes):
                            return blob.decode(errors="replace")
                        return blob or ""
                    return subprocess.CompletedProcess(
                        exc.cmd, 0, _text(exc.output), _text(exc.stderr)
                    )
                raise CliError("kaggle %s timed out after %ss" % (args[0], self.timeout)) from exc
        if check and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise CliError(
                "kaggle %s failed (exit %d): %s"
                % (" ".join(args), proc.returncode, detail[-1] if detail else "no output")
            )
        return proc

    def cli_json(self, account: PoolAccount, *args: str) -> Any:
        """Run a CLI command with ``--format json`` and parse its output."""
        proc = self.run_cli(account, *args, "--format", "json")
        text = (proc.stdout or "").strip()
        start = min((i for i in (text.find("["), text.find("{")) if i != -1), default=-1)
        if start == -1:
            # The CLI reports an empty result in prose ("No competitions found"), which is an
            # answer, not a failure.
            if re.match(r"(?i)^no\s+\w+.*\bfound\b", text) or not text:
                return []
            raise CliError("kaggle %s printed no JSON: %s" % (" ".join(args), text[:200]))
        try:
            return json.loads(text[start:])
        except ValueError as exc:
            raise CliError("kaggle %s printed invalid JSON: %s" % (" ".join(args), exc))

    # -- individual calls --------------------------------------------------------------
    def username(self, account: PoolAccount) -> str:
        """Resolve the account's Kaggle username from its credential."""
        if account.resolved_username:
            return account.resolved_username
        proc = self.run_cli(account, "config", "view", check=False)
        match = re.search(r"username:\s*(\S+)", proc.stdout or "")
        return match.group(1) if match and match.group(1) != "None" else account.username

    def quotas(self, account: PoolAccount) -> Dict[str, Quota]:
        rows = self.cli_json(account, "quota")
        out = {q.kind: q for q in (Quota.from_row(r) for r in rows or [])}
        out.setdefault("gpu", Quota("gpu"))
        out.setdefault("tpu", Quota("tpu"))
        return out

    def kernel_status(self, account: PoolAccount, ref: str) -> str:
        """Latest run state of ``owner/slug``: queued / running / complete / error / ..."""
        proc = self.run_cli(account, "kernels", "status", ref, check=False)
        text = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 and "status" not in text:
            return "unknown"
        return _normalize_status(text)

    def kernel_logs(
        self,
        account: PoolAccount,
        ref: str,
        follow: bool = False,
        follow_seconds: float = 30.0,
    ) -> str:
        """Execution logs of the latest run.

        Plain ``kaggle kernels logs`` returns **nothing while the kernel is running**, then the
        whole log once it finishes. ``follow=True`` uses ``--follow``, which streams the live
        session; that never exits on its own, so it is read for ``follow_seconds`` and stopped,
        returning what arrived. Pass the result through :func:`filter_progress` to drop
        progress-bar noise, which dominates a training log.
        """
        if follow:
            proc = self.run_cli(
                account, "kernels", "logs", "-f", ref,
                check=False, timeout=follow_seconds, partial_on_timeout=True,
            )
        else:
            proc = self.run_cli(account, "kernels", "logs", ref, check=False)
        return ((proc.stdout or "") + (proc.stderr or "")).strip()

    def kernel_metadata(self, account: PoolAccount, ref: str) -> Dict[str, Any]:
        """``kernel-metadata.json`` of ``owner/slug``, via ``kaggle kernels pull -m``.

        This is the only place the CLI exposes ``machine_shape``, i.e. which accelerator a
        kernel asks for. One subprocess per kernel, so callers keep the set small.
        """
        with tempfile.TemporaryDirectory(prefix="kaggle-meta-") as out:
            proc = self.run_cli(account, "kernels", "pull", ref, "-p", out, "-m", check=False)
            path = os.path.join(out, "kernel-metadata.json")
            if proc.returncode != 0 or not os.path.isfile(path):
                return {}
            try:
                with open(path) as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                return {}

    def machine_shape(self, account: PoolAccount, ref: str) -> Optional[str]:
        """The accelerator ``ref`` requests (``"NvidiaL4"``), or ``None`` for CPU/unknown."""
        return self.kernel_metadata(account, ref).get("machine_shape") or None

    def entered_competitions(self, account: PoolAccount) -> List[str]:
        """Competition slugs the account has entered; the gate for non-T4/P100 accelerators."""
        rows = self.cli_json(account, "competitions", "list", "--group", "entered") or []
        out = []
        for row in rows:
            ref = row.get("ref") or row.get("title") or ""
            out.append(str(ref).rstrip("/").rsplit("/", 1)[-1])
        return [c for c in out if c]

    def my_kernels(self, account: PoolAccount, page_size: int = 20) -> List[Dict[str, Any]]:
        return self.cli_json(
            account, "kernels", "list", "--mine",
            "--page-size", str(page_size), "--sort-by", "dateRun",
        ) or []

    # -- probing -----------------------------------------------------------------------
    def probe(
        self,
        account: PoolAccount,
        check_kernels: bool = True,
        recent: int = 10,
        recent_within_hours: float = 24.0,
        with_accelerators: bool = False,
        with_competitions: bool = False,
    ) -> AccountStatus:
        """Probe one account: quota, then the status of its recently run kernels.

        Only kernels whose ``lastRunTime`` is within ``recent_within_hours`` are
        status-checked; each check is a CLI call, and an older kernel cannot be running.

        ``with_accelerators`` additionally pulls each inspected kernel's metadata to learn
        which GPU type it asked for. That is one extra CLI call per kernel, so it is off by
        default and always collected for kernels that are currently active.
        """
        status = AccountStatus(account=account, max_concurrent=self.max_concurrent)
        try:
            quotas = self.quotas(account)
        except Exception as exc:
            status.error = "%s: %s" % (type(exc).__name__, exc)
            return status

        status.ok = True
        status.gpu, status.tpu = quotas["gpu"], quotas["tpu"]
        resolved = self.username(account)
        if resolved and resolved != account.username:
            object.__setattr__(account, "resolved_username", resolved)
        else:
            object.__setattr__(account, "resolved_username", resolved or account.username)

        if with_competitions:
            try:
                status.competitions = self.entered_competitions(account)
                status.competitions_checked = True
            except Exception as exc:
                status.error = "quota ok, competitions failed (%s)" % exc

        if not check_kernels or recent <= 0:
            return status

        try:
            listing = self.my_kernels(account, page_size=max(recent, 1))
        except Exception as exc:
            status.error = "quota ok, kernel listing failed (%s)" % exc
            return status

        cutoff = datetime.now(timezone.utc) - timedelta(hours=recent_within_hours)
        for item in listing[:recent]:
            ref = item.get("ref") or ""
            if "/" not in ref:
                continue
            last_run = item.get("lastRunTime")
            if last_run and recent_within_hours > 0:
                try:
                    when = datetime.fromisoformat(
                        str(last_run).replace("Z", "").split("+")[0]
                    ).replace(tzinfo=timezone.utc)
                    if when < cutoff:
                        continue        # too old to still be running
                except ValueError:
                    pass
            state = self.kernel_status(account, ref)
            run = KernelRun(ref=ref, status=state, last_run=last_run)
            # An active kernel's hardware is the interesting part, so fetch that regardless.
            if with_accelerators or run.is_active:
                run.machine_shape = self.machine_shape(account, ref) or ""
            status.kernels.append(run)
        status.kernels_checked = True
        return status

    def probe_all(
        self,
        check_kernels: bool = True,
        recent: int = 10,
        recent_within_hours: float = 24.0,
        with_accelerators: bool = False,
        with_competitions: bool = False,
    ) -> List[AccountStatus]:
        """Probe every account in parallel (one thread per account)."""
        with ThreadPoolExecutor(max_workers=min(self.workers, len(self.accounts))) as pool:
            self._statuses = list(
                pool.map(
                    lambda acc: self.probe(
                        acc, check_kernels, recent, recent_within_hours,
                        with_accelerators, with_competitions,
                    ),
                    self.accounts,
                )
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
        accelerator: Optional[str] = None,
    ) -> List[AccountStatus]:
        """Accounts with spare capacity, richest in remaining quota first.

        ``accelerator`` keeps only accounts that can reach that GPU type: T4/P100 anywhere,
        L4 / RTX Pro 6000 / TPU only where a competition has been entered. Quota is pooled
        across types, so this is about eligibility, never budget.
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
            if accelerator and not status.can_use(accelerator):
                continue
            out.append(status)
        return sorted(out, key=lambda s: s.quota(resource).remaining_h, reverse=True)

    def best(self, **kwargs: Any) -> Optional[AccountStatus]:
        candidates = self.available(**kwargs)
        return candidates[0] if candidates else None

    def account_named(self, name: str) -> Optional[AccountStatus]:
        for status in self.statuses:
            if name in (status.name, status.username):
                return status
        return None

    def running(self) -> List[KernelRun]:
        return [k for s in self.statuses for k in s.active_kernels]

    def running_by_account(self) -> Dict[str, List[KernelRun]]:
        return {s.name: s.active_kernels for s in self.statuses if s.active_kernels}

    def total_remaining_hours(self, resource: str = "gpu") -> float:
        return sum(s.quota(resource).remaining_h for s in self.statuses if s.ok)

    # -- reporting ---------------------------------------------------------------------
    def report(self, resource: str = "gpu") -> str:
        statuses = self.statuses
        rows = [("ACCOUNT", "KAGGLE USER", "GPU LEFT", "TPU LEFT", "RUNNING", "STATE")]
        for status in sorted(statuses, key=lambda s: (not s.ok, -s.quota(resource).remaining_h)):
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
        seen_any = {k: v for s in statuses for k, v in s.accelerators_seen.items()}
        if seen_any:
            lines.append("")
            lines.append("accelerators seen in inspected runs (one pooled GPU budget, not per type):")
            for status in statuses:
                if status.accelerators_seen:
                    lines.append("  %-28s %s" % (
                        status.name,
                        ", ".join("%s x%d" % (a, n) for a, n in status.accelerators_seen.items()),
                    ))
            pool_counts: Dict[str, int] = {}
            for status in statuses:
                for accel, n in status.accelerators_seen.items():
                    pool_counts[accel] = pool_counts.get(accel, 0) + n
            lines.append("  %-28s %s" % (
                "POOL", ", ".join("%s x%d" % (a, n) for a, n in
                                  sorted(pool_counts.items(), key=lambda kv: (-kv[1], kv[0])))))

        if any(st.competitions_checked for st in statuses):
            lines.append("")
            lines.append("competitions entered (the gate for L4 / RTX Pro 6000 / TPU):")
            for status in statuses:
                if not status.competitions_checked:
                    continue
                comps = status.competitions
                lines.append("  %-28s %-4s %s" % (
                    status.name,
                    "YES" if comps else "no",
                    (", ".join(comps[:3]) + (" +%d more" % (len(comps) - 3) if len(comps) > 3 else ""))
                    if comps else "T4 / P100 only",
                ))
            eligible = [st.name for st in statuses if st.competitions_checked and st.competitions]
            lines.append("  %-28s %d of %d accounts can reach the competition-only GPUs"
                         % ("POOL", len(eligible), sum(1 for st in statuses if st.competitions_checked)))

        active = self.running_by_account()
        if active:
            lines.append("")
            lines.append("active kernels:")
            for name, kernels in active.items():
                for kernel in kernels:
                    lines.append("  %-28s %-10s %-8s %s" % (
                        name, kernel.status, kernel.accelerator or "?", kernel.url))
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
        help="credentials directory (default: $KAGGLE_TOKENS_DIR)",
    )
    parser.add_argument("--kaggle-cli", default=None, help="path to the kaggle executable")
    parser.add_argument("--resource", choices=("gpu", "tpu"), default="gpu")
    parser.add_argument("--min-gpu-hours", type=float, default=0.0)
    parser.add_argument("--recent", type=int, default=10, help="recent kernels to status-check")
    parser.add_argument(
        "--recent-hours",
        type=float,
        default=24.0,
        help="only status-check kernels run within this many hours (0 = all)",
    )
    parser.add_argument("--no-kernels", action="store_true", help="quota only, skip kernel checks")
    parser.add_argument(
        "--accelerators",
        action="store_true",
        help="also report which GPU type each inspected kernel used (one extra CLI call per kernel)",
    )
    parser.add_argument(
        "--accelerator",
        help="only count accounts that can reach this GPU type (t4, p100, l4, rtx6000, tpu-v5e)",
    )
    parser.add_argument(
        "--competitions",
        action="store_true",
        help="also report entered competitions, which gate every GPU beyond T4/P100",
    )
    parser.add_argument("--idle-only", action="store_true")
    parser.add_argument("--running", action="store_true", help="print only active kernels")
    parser.add_argument("--best", action="store_true", help="print the single best account")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.tokens_dir:
        parser.error("--tokens-dir is required (or set KAGGLE_TOKENS_DIR)")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    router = PoolRouter.from_dir(
        args.tokens_dir,
        cli=args.kaggle_cli,
        workers=args.workers,
        max_concurrent=args.max_concurrent,
    )
    router.probe_all(
        check_kernels=not args.no_kernels,
        recent=args.recent,
        recent_within_hours=args.recent_hours,
        with_accelerators=args.accelerators,
        with_competitions=args.competitions or bool(args.accelerator),
    )

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
            accelerator=args.accelerator,
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
    if args.min_gpu_hours or args.idle_only or args.accelerator:
        picks = router.available(
            min_gpu_hours=args.min_gpu_hours,
            resource=args.resource,
            require_idle=args.idle_only,
            accelerator=args.accelerator,
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
