"""Launch notebooks onto the shared Kaggle account pool.

Describe a run declaratively, let :class:`~infra.kaggle_router.KaggleRouter` pick the
account with the most spare accelerator quota, and push it:

    from infra.kaggle_router import KaggleRouter
    from infra.kaggle_launcher import KaggleLauncher, NotebookSpec, git_run_spec

    launcher = KaggleLauncher(KaggleRouter.from_dir("~/kaggle_tokens"))
    spec = git_run_spec(
        title="clifford-flow-default",
        repo="https://github.com/optimist1938/Clifford-Flow-Matching.git",
        command="python -m src.main",
        accelerator="t4",
    )
    handle = launcher.launch(spec, min_gpu_hours=6, require_idle=True)
    print(handle.url, handle.wait(timeout=600))

Accelerator selection travels in the push metadata as ``machineShape``; see
``ACCELERATORS`` for the values Kaggle accepts.

CLI
---
    python infra/kaggle_launcher.py --tokens-dir DIR --repo URL --command CMD \
        --title my-run --accelerator t4 [--dry-run] [--wait 600]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:  # package import
    from .kaggle_router import KaggleAccount, KaggleRouter, ACTIVE_STATES
except ImportError:  # run as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from infra.kaggle_router import KaggleAccount, KaggleRouter, ACTIVE_STATES

#: Friendly alias -> Kaggle ``machineShape`` value. ``None`` means CPU only.
#: ``t4``/``p100``/``l4``/``rtx6000`` are confirmed working; TPU values are accepted by
#: the push API but currently provision a non-TPU image (Kaggle/kaggle-cli#1197), and
#: there is no value for the editor's "GPU T4 x2" (Kaggle/kaggle-cli#1196).
ACCELERATORS: Dict[str, Optional[str]] = {
    "none": None,
    "cpu": None,
    "t4": "NvidiaTeslaT4",
    "p100": "NvidiaTeslaP100",
    "l4": "NvidiaL4",
    "rtx6000": "NvidiaRtxPro6000",
    "tpu-v3": "Tpu1VmV38",
    "tpu-v5e": "TpuV5E8",
}

TERMINAL_STATES = frozenset({"complete", "error", "cancelAcknowledged"})


def _slugify(title: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in title.strip()]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:48] or "notebook"


# --------------------------------------------------------------------------------------
# what to run
# --------------------------------------------------------------------------------------
@dataclass
class NotebookSpec:
    """A notebook to run, plus the machine it should run on."""

    title: str
    cells: Optional[List[str]] = None
    notebook_path: Optional[str] = None
    slug: Optional[str] = None
    #: Numeric Kaggle kernel id, when updating an existing kernel rather than creating one.
    kernel_id: Optional[int] = None
    accelerator: str = "none"
    enable_internet: bool = True
    is_private: bool = True
    datasets: List[str] = field(default_factory=list)
    kernel_sources: List[str] = field(default_factory=list)
    model_sources: List[str] = field(default_factory=list)
    competitions: List[str] = field(default_factory=list)
    #: Kaggle secrets the code expects. The API cannot create secrets, so this is
    #: checked by a preflight cell and used to filter target accounts.
    required_secrets: List[str] = field(default_factory=list)
    docker_pinning: str = "latest"

    def __post_init__(self) -> None:
        if not self.cells and not self.notebook_path:
            raise ValueError("NotebookSpec needs either cells or notebook_path")
        if self.accelerator not in ACCELERATORS:
            raise ValueError(
                "unknown accelerator %r; choose from %s"
                % (self.accelerator, ", ".join(sorted(ACCELERATORS)))
            )
        self.slug = self.slug or _slugify(self.title)

    # -- machine ----------------------------------------------------------------------
    @property
    def machine_shape(self) -> Optional[str]:
        return ACCELERATORS[self.accelerator]

    @property
    def is_tpu(self) -> bool:
        shape = self.machine_shape
        return bool(shape and shape.lower().startswith("tpu"))

    @property
    def is_gpu(self) -> bool:
        return bool(self.machine_shape) and not self.is_tpu

    @property
    def resource(self) -> str:
        """Which quota this run consumes: ``gpu``, ``tpu`` or ``cpu``."""
        if self.is_tpu:
            return "tpu"
        return "gpu" if self.is_gpu else "cpu"

    # -- source -----------------------------------------------------------------------
    def _preflight_cell(self) -> str:
        """A first cell that fails loudly instead of 9 hours later."""
        lines = [
            "import subprocess, sys",
            'print("python", sys.version.split()[0])',
            'print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",'
            ' "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()'
            ' or "no GPU visible")',
        ]
        if self.required_secrets:
            lines += [
                "from kaggle_secrets import UserSecretsClient",
                "_secrets = UserSecretsClient()",
                "for _name in %r:" % (list(self.required_secrets),),
                "    try:",
                "        _secrets.get_secret(_name)",
                '        print("secret ok:", _name)',
                "    except Exception as exc:",
                '        raise RuntimeError("missing Kaggle secret %s on this account'
                ' - add it in Add-ons > Secrets" % _name) from exc',
            ]
        return "\n".join(lines)

    def to_ipynb(self) -> Dict[str, Any]:
        if self.notebook_path:
            with open(os.path.expanduser(self.notebook_path), "r") as fh:
                return json.load(fh)
        sources = [self._preflight_cell()] + list(self.cells or [])
        return {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": src,
                }
                for src in sources
            ],
        }

    def push_body(self, username: str) -> Dict[str, Any]:
        """The ``POST /kernels/push`` payload for this spec on ``username``."""
        # ``slug`` carries "username/kernel-slug"; ``id`` is the *numeric* kernel id and is
        # only sent when updating a kernel we already know the id of.
        body = {
            "slug": "%s/%s" % (username, self.slug),
            "newTitle": self.title,
            "text": json.dumps(self.to_ipynb()),
            "language": "python",
            "kernelType": "notebook",
            "isPrivate": self.is_private,
            "enableGpu": self.is_gpu,
            "enableTpu": self.is_tpu,
            "enableInternet": self.enable_internet,
            "datasetDataSources": list(self.datasets),
            "competitionDataSources": list(self.competitions),
            "kernelDataSources": list(self.kernel_sources),
            "modelDataSources": list(self.model_sources),
            "categoryIds": [],
            "dockerImagePinningType": self.docker_pinning,
        }
        if self.machine_shape:
            body["machineShape"] = self.machine_shape
        if self.kernel_id is not None:
            body["id"] = int(self.kernel_id)
        return body


def git_run_spec(
    title: str,
    repo: str,
    command: str,
    accelerator: str = "t4",
    branch: Optional[str] = None,
    subdir: Optional[str] = None,
    install: str = "pip install poetry --quiet && poetry install -q",
    runner: str = "poetry run",
    env: Optional[Dict[str, str]] = None,
    github_token_secret: Optional[str] = None,
    **kwargs: Any,
) -> NotebookSpec:
    """Build the usual clone -> install -> train notebook.

    Args:
        repo: https clone URL.
        command: what to run inside the repo, e.g. ``"python -m src.main --model clifford"``.
        install: dependency install step, run inside the repo directory.
        runner: prefix for ``command`` (``"poetry run"``, or ``""`` for plain python).
        env: environment variables exported before the command.
        github_token_secret: name of a Kaggle secret holding a PAT, for private repos.
        subdir: path inside the repo to install and run in, for a project in a subdirectory.
    """
    name = repo.rstrip("/").rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    clone_dir = "/kaggle/working/%s" % name
    work = os.path.join(clone_dir, subdir) if subdir else clone_dir
    secrets = list(kwargs.pop("required_secrets", []))

    clone_lines = ["import os, shutil, subprocess"]
    if github_token_secret:
        secrets.append(github_token_secret)
        clone_lines += [
            "from kaggle_secrets import UserSecretsClient",
            "_tok = UserSecretsClient().get_secret(%r)" % github_token_secret,
            "_url = %r.replace('https://', 'https://git:' + _tok + '@')" % repo,
        ]
    else:
        clone_lines.append("_url = %r" % repo)
    clone_lines += [
        "if os.path.exists(%r): shutil.rmtree(%r)" % (clone_dir, clone_dir),
        "os.chdir('/kaggle/working')",
        "_branch = %r" % (branch or ""),
        "_cmd = ['git', 'clone', '--depth', '1'] + (['--branch', _branch] if _branch else []) + [_url]",
        "assert subprocess.run(_cmd).returncode == 0, 'git clone failed'",
        "os.chdir(%r)" % work,
        "print('cloned into', os.getcwd())",
        "print(subprocess.run(['git', 'log', '-1', '--oneline'], capture_output=True, text=True).stdout.strip())",
    ]

    exports = " ".join("%s=%s" % (k, v) for k, v in (env or {}).items())
    full_command = " ".join(x for x in (exports, runner, command) if x).strip()
    cells = [
        "\n".join(clone_lines),
        "import os, subprocess\nos.chdir(%r)\n%s"
        % (work, "print(subprocess.run(%r, shell=True, text=True).returncode)" % install),
        "import os, subprocess, sys\n"
        "os.chdir(%r)\n" % work
        + "_p = subprocess.run(%r, shell=True, text=True)\n" % full_command
        + "print('exit code:', _p.returncode)\n"
        "sys.exit(_p.returncode) if _p.returncode else print('run finished cleanly')",
    ]
    return NotebookSpec(
        title=title, cells=cells, accelerator=accelerator, required_secrets=secrets, **kwargs
    )


# --------------------------------------------------------------------------------------
# a launched run
# --------------------------------------------------------------------------------------
def normalize_ref(ref: str, username: str = "", slug: str = "") -> str:
    """Reduce whatever ``/kernels/push`` returned to a bare ``owner/slug``.

    The API answers with a site path such as ``/code/owner/slug`` (sometimes url-quoted),
    which is not the ``owner/slug`` form every other endpoint expects.
    """
    ref = urllib.parse.unquote((ref or "").strip())
    for prefix in ("https://www.kaggle.com", "http://www.kaggle.com"):
        if ref.startswith(prefix):
            ref = ref[len(prefix):]
    ref = ref.strip("/")
    for prefix in ("code/", "kernels/"):
        while ref.startswith(prefix):
            ref = ref[len(prefix):]
    parts = [p for p in ref.split("/") if p]
    if len(parts) >= 2:
        return "%s/%s" % (parts[0], parts[1])
    return "%s/%s" % (username, slug) if username and slug else ref


@dataclass
class LaunchHandle:
    """A pushed kernel: poll it, wait on it, cancel it."""

    launcher: "KaggleLauncher"
    account: KaggleAccount
    ref: str
    version: Optional[int] = None
    spec: Optional[NotebookSpec] = None

    def __post_init__(self) -> None:
        self.ref = normalize_ref(
            self.ref, self.account.username, self.spec.slug if self.spec else ""
        )

    @property
    def url(self) -> str:
        return "https://www.kaggle.com/code/" + self.ref

    def status(self) -> str:
        owner, slug = self.ref.split("/", 1)
        try:
            payload = self.launcher.router.request(
                self.account, "/kernels/status", userName=owner, kernelSlug=slug
            )
        except FileNotFoundError:
            return "notRun"
        return payload.get("status", "unknown")

    def failure(self) -> str:
        owner, slug = self.ref.split("/", 1)
        try:
            payload = self.launcher.router.request(
                self.account, "/kernels/status", userName=owner, kernelSlug=slug
            )
        except Exception:
            return ""
        return payload.get("failureMessage", "") or ""

    def wait(self, timeout: float = 3600.0, poll: float = 20.0, verbose: bool = True) -> str:
        """Block until the kernel reaches a terminal state, the timeout, or Ctrl-C."""
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            state = self.status()
            if verbose and state != last:
                print("[%s] %s" % (time.strftime("%H:%M:%S"), state), flush=True)
                last = state
            if state in TERMINAL_STATES:
                return state
            time.sleep(poll)
        return last or "timeout"

    def log(self) -> str:
        """The kernel's execution log.

        Note: for a pushed (batch) kernel Kaggle only serves output **after the run
        finishes** - while the status is ``running`` this returns ``""``. There is no
        live log streaming in the public API; for progress on a long run, log to W&B
        from inside the notebook instead.
        """
        owner, slug = self.ref.split("/", 1)
        payload = self.launcher.router.request(
            self.account, "/kernels/output", userName=owner, kernelSlug=slug
        )
        log = payload.get("log") or ""
        if isinstance(log, str) and log.lstrip().startswith("["):
            try:  # the API returns a JSON array of {stream_name, time, data} entries
                log = json.loads(log)
            except ValueError:
                return log
        if isinstance(log, list):
            return "".join(
                entry.get("data", "") if isinstance(entry, dict) else str(entry) for entry in log
            )
        return log

    def output_files(self) -> List[Dict[str, Any]]:
        """Files the kernel wrote to /kaggle/working, as ``{fileName, url}`` dicts."""
        owner, slug = self.ref.split("/", 1)
        payload = self.launcher.router.request(
            self.account, "/kernels/output", userName=owner, kernelSlug=slug
        )
        return payload.get("files") or []

    def cancel(self) -> str:
        owner, slug = self.ref.split("/", 1)
        self.launcher.router.request(
            self.account, "/kernels/cancel", body={}, userName=owner, kernelSlug=slug
        )
        return self.status()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "account": self.account.name,
            "username": self.account.username,
            "ref": self.ref,
            "url": self.url,
            "version": self.version,
            "accelerator": self.spec.accelerator if self.spec else None,
        }


# --------------------------------------------------------------------------------------
# launcher
# --------------------------------------------------------------------------------------
class KaggleLauncher:
    """Route a :class:`NotebookSpec` to an account with spare quota and push it."""

    def __init__(self, router: KaggleRouter) -> None:
        self.router = router

    # -- routing ----------------------------------------------------------------------
    def pick_account(
        self,
        spec: NotebookSpec,
        min_hours: float = 1.0,
        require_idle: bool = False,
        account: Optional[str] = None,
    ) -> KaggleAccount:
        """Choose the target account, or raise with the reason none qualified."""
        statuses = self.router.statuses
        if account:
            for status in statuses:
                if account in (status.name, status.username):
                    if not status.ok:
                        raise RuntimeError("account %s is unreachable: %s" % (account, status.error))
                    return status.account
            raise KeyError("no account named %r in the pool" % account)

        if spec.resource == "cpu":
            candidates = [s for s in statuses if s.ok]
            if require_idle:
                candidates = [s for s in candidates if not s.busy]
            if not candidates:
                raise RuntimeError("no reachable%s account in the pool"
                                   % (" idle" if require_idle else ""))
            # CPU runs burn no accelerator quota, so rank by "least loaded": idle first,
            # then most GPU-hours left, so a CPU job avoids the accounts you want for GPU work.
            candidates.sort(key=lambda s: (s.busy, -s.gpu.remaining_h))
            return candidates[0].account

        candidates = self.router.available(
            min_gpu_hours=min_hours, resource=spec.resource, require_idle=require_idle
        )
        if not candidates:
            raise RuntimeError(
                "no account has >=%.1f spare %s-hours%s (pool total %.1fh)"
                % (
                    min_hours,
                    spec.resource,
                    " while idle" if require_idle else "",
                    self.router.total_remaining_hours(spec.resource),
                )
            )
        return candidates[0].account

    # -- launching --------------------------------------------------------------------
    def launch(
        self,
        spec: NotebookSpec,
        min_hours: float = 1.0,
        require_idle: bool = False,
        account: Optional[str] = None,
        dry_run: bool = False,
    ) -> LaunchHandle:
        """Push ``spec`` to the best available account and start it running."""
        target = self.pick_account(spec, min_hours, require_idle, account)
        body = spec.push_body(target.username)

        if dry_run:
            preview = dict(body)
            preview["text"] = "<%d chars of ipynb>" % len(body["text"])
            print(json.dumps({"target": target.username, "push_body": preview}, indent=2))
            return LaunchHandle(self, target, "%s/%s" % (target.username, spec.slug), spec=spec)

        response = self.router.request(target, "/kernels/push", body=body)
        errors = response.get("error") or ""
        if errors:
            raise RuntimeError("push rejected: %s" % errors)
        ref = response.get("ref") or "%s/%s" % (target.username, spec.slug)
        return LaunchHandle(
            self, target, ref, version=response.get("versionNumber"), spec=spec
        )

    def launch_many(self, specs: Sequence[NotebookSpec], **kwargs: Any) -> List[LaunchHandle]:
        """Spread several runs across the pool, skipping accounts already used here."""
        handles, used = [], set()
        for spec in specs:
            target = None
            for status in self.router.available(
                min_gpu_hours=kwargs.get("min_hours", 1.0), resource=spec.resource
            ):
                if status.username not in used:
                    target = status.username
                    break
            handle = self.launch(spec, account=target, **kwargs)
            used.add(handle.account.username)
            handles.append(handle)
        return handles

    def active(self) -> List[str]:
        """Refresh the pool and list every running/queued kernel ref."""
        self.router.probe_all()
        return [k.ref for k in self.router.running()]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Launch a notebook on the Kaggle pool.")
    parser.add_argument("--tokens-dir", default=os.environ.get("KAGGLE_TOKENS_DIR"), required=False)
    parser.add_argument("--title", required=True)
    parser.add_argument("--repo", help="git clone URL (clone -> install -> run notebook)")
    parser.add_argument("--branch")
    parser.add_argument("--command", help="command to run inside the repo")
    parser.add_argument("--runner", default="poetry run")
    parser.add_argument("--install", default="pip install poetry --quiet && poetry install -q")
    parser.add_argument("--notebook", help="push an existing .ipynb instead of --repo")
    parser.add_argument("--accelerator", default="t4", choices=sorted(ACCELERATORS))
    parser.add_argument("--dataset", action="append", default=[], help="owner/slug (repeatable)")
    parser.add_argument("--secret", action="append", default=[], help="required Kaggle secret")
    parser.add_argument("--github-token-secret")
    parser.add_argument("--public", action="store_true", help="make the kernel public")
    parser.add_argument("--no-internet", action="store_true")
    parser.add_argument("--account", help="force a specific pool account")
    parser.add_argument("--min-hours", type=float, default=1.0)
    parser.add_argument("--idle-only", action="store_true")
    parser.add_argument("--wait", type=float, default=0.0, help="seconds to wait for completion")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.tokens_dir:
        parser.error("--tokens-dir is required (or set KAGGLE_TOKENS_DIR)")
    if not args.notebook and not (args.repo and args.command):
        parser.error("give --notebook, or both --repo and --command")

    common = dict(
        accelerator=args.accelerator,
        datasets=args.dataset,
        is_private=not args.public,
        enable_internet=not args.no_internet,
    )
    if args.notebook:
        spec = NotebookSpec(
            title=args.title,
            notebook_path=args.notebook,
            required_secrets=args.secret,
            **common
        )
    else:
        spec = git_run_spec(
            title=args.title,
            repo=args.repo,
            branch=args.branch,
            command=args.command,
            runner=args.runner,
            install=args.install,
            github_token_secret=args.github_token_secret,
            required_secrets=args.secret,
            **common
        )

    router = KaggleRouter.from_dir(args.tokens_dir)
    router.probe_all()
    launcher = KaggleLauncher(router)
    handle = launcher.launch(
        spec,
        min_hours=args.min_hours,
        require_idle=args.idle_only,
        account=args.account,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return 0

    print("launched %s on %s (%s)" % (spec.title, handle.account.name, spec.accelerator))
    print(handle.url)
    if args.wait:
        state = handle.wait(timeout=args.wait)
        print("final status:", state)
        if state == "error":
            print("failure:", handle.failure())
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
