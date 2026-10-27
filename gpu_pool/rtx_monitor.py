"""Sync no-internet (RTX / 4xL4 / TPU) Kaggle runs to W&B by tailing their logs.

A Kaggle kernel with internet disabled can't reach wandb.ai, so `pose3d`'s trainer prints one
grep-able JSON line per epoch instead (``WANDB_SYNC ...`` / ``WANDB_SYNC_FINAL ...``, see
``pose3d/pose3d/engine/tracking.py:log_offline_sync`` -- a no-op on a normal online run, which
still syncs to wandb.ai live and needs nothing from this script). This polls every account in the
pool, finds kernels running on a premium (competition-gated) accelerator, tails their logs with
``kaggle kernels logs``, and re-plays any new points into the *same* W&B run (by `run_name`), so
an offline run shows up in the dashboard exactly like a live one, just a few minutes behind.

CLI
---
    python gpu_pool/rtx_monitor.py --tokens-dir ~/.kaggle-accounts/pool           # loop forever
    python gpu_pool/rtx_monitor.py --tokens-dir DIR --once                       # single pass
    python gpu_pool/rtx_monitor.py --tokens-dir DIR --once --dry-run             # no wandb calls

Requirements: ``pip install wandb`` and a working `wandb login` (or `WANDB_API_KEY`) on this
machine -- this script authenticates to wandb.ai itself; the Kaggle kernel never does.

State: a small JSON file (default ``~/.kaggle-accounts/rtx_monitor_state.json``) maps each
kernel ref to the wandb run id it created and the last step synced, so re-polling the same
still-running kernel appends to one continuous run instead of creating a new one each pass.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:  # package import
    from .router import GATED_ACCELERATORS, KernelRun, PoolRouter, normalize_accelerator
except ImportError:  # run as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gpu_pool.router import GATED_ACCELERATORS, KernelRun, PoolRouter, normalize_accelerator

#: Accelerators this monitor watches. T4/P100/CPU runs sync to wandb.ai on their own and are
#: never touched here -- that's the "must not spoil T4 runs" requirement, satisfied by simply
#: never looking at them. Every competition-gated accelerator the router knows (RTX Pro 6000,
#: 4xL4, TPU) runs with internet disabled, so all of them are watched.
DEFAULT_ACCELERATORS = GATED_ACCELERATORS

_LINE_RE = re.compile(r"^(WANDB_SYNC_FINAL|WANDB_SYNC) (\{.*\})\s*$", re.MULTILINE)

DEFAULT_STATE_FILE = os.path.expanduser("~/.kaggle-accounts/rtx_monitor_state.json")


# --------------------------------------------------------------------------------------
# parsing -- pure functions, unit-testable without any credentials or network
# --------------------------------------------------------------------------------------
def _plain_log(log_text: str) -> str:
    """A finished kernel's log arrives as a JSON array of ``{"stream_name", "time", "data"}``
    records (quotes escaped), a running one as plain text; return plain text either way."""
    if not log_text.lstrip().startswith("["):
        return log_text
    try:
        records = json.loads(log_text)
    except ValueError:
        return log_text
    return "".join(r.get("data", "") for r in records if isinstance(r, dict))


def parse_sync_lines(log_text: str) -> List[Dict[str, Any]]:
    """Every ``WANDB_SYNC``/``WANDB_SYNC_FINAL`` line in ``log_text``, in file order.

    Malformed JSON on a matched line is skipped (a truncated log tail can cut a line in half);
    everything else in the log is ignored, so ordinary training prints and tqdm noise (or its
    absence) don't matter here.
    """
    out = []
    for kind, blob in _LINE_RE.findall(_plain_log(log_text or "")):
        try:
            payload = json.loads(blob)
        except ValueError:
            continue
        payload["final"] = kind == "WANDB_SYNC_FINAL" or bool(payload.get("final"))
        if out and (out[-1].get("step"), out[-1]["final"]) == (payload.get("step"), payload["final"]):
            continue  # a finished kernel's log repeats each line
        out.append(payload)
    return out


def new_points(payloads: Sequence[Dict[str, Any]], last_step: Optional[int]) -> List[Dict[str, Any]]:
    """Payloads with ``step`` beyond ``last_step`` (or all of them if nothing was synced yet).

    Sorted by step; a step already seen (e.g. the log tail overlaps the previous poll) is kept
    only the once. The final-evaluation line shares its step with the last epoch, so both the
    epoch and the final point at that step are kept (`wandb.log` will just overwrite the metric
    keys epoch logging didn't set; `run.summary.update` for the final line is a separate call).
    """
    seen_steps = set()
    out = []
    for payload in sorted(payloads, key=lambda p: (p.get("step", 0), p.get("final", False))):
        step = payload.get("step", 0)
        if last_step is not None and step < last_step:
            continue
        if step == last_step and not payload.get("final"):
            continue  # the non-final point at last_step was already synced last poll
        key = (step, payload.get("final", False))
        if key in seen_steps:
            continue
        seen_steps.add(key)
        out.append(payload)
    return out


# --------------------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------------------
@dataclass
class RunState:
    wandb_run_id: Optional[str] = None
    last_step: Optional[int] = None
    finished: bool = False
    run_name: Optional[str] = None


class MonitorState:
    """Persists ``RunState`` per kernel ref across separate invocations of this script."""

    def __init__(self, path: str = DEFAULT_STATE_FILE) -> None:
        self.path = os.path.expanduser(path)
        self._data: Dict[str, Dict[str, Any]] = {}
        if os.path.isfile(self.path):
            try:
                with open(self.path) as fh:
                    self._data = json.load(fh)
            except (OSError, ValueError):
                self._data = {}

    def get(self, ref: str) -> RunState:
        row = self._data.get(ref, {})
        return RunState(
            wandb_run_id=row.get("wandb_run_id"),
            last_step=row.get("last_step"),
            finished=row.get("finished", False),
            run_name=row.get("run_name"),
        )

    def refs(self) -> List[str]:
        return list(self._data)

    def set(self, ref: str, state: RunState) -> None:
        self._data[ref] = {
            "wandb_run_id": state.wandb_run_id,
            "last_step": state.last_step,
            "finished": state.finished,
            "run_name": state.run_name,
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self._data, fh, indent=1)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------------------
# wandb sink
# --------------------------------------------------------------------------------------
class WandbSink:
    """Replays parsed points into wandb.ai. Swappable for a dry-run printer in tests/CLI."""

    def push(self, payloads: List[Dict[str, Any]], state: RunState) -> RunState:
        import wandb  # local import: only needed on the real (non-dry-run) path

        if not payloads:
            return state
        entity, project = payloads[0]["wandb_entity"], payloads[0]["wandb_project"]
        run_name = payloads[0]["run_name"]
        run = wandb.init(
            entity=entity, project=project,
            id=state.wandb_run_id, name=run_name if not state.wandb_run_id else None,
            resume="allow",
        )
        try:
            for payload in payloads:
                if payload["final"]:
                    run.summary.update(payload["metrics"])
                else:
                    run.log(payload["metrics"], step=payload["step"])
                state.last_step = payload["step"]
                state.finished = state.finished or payload["final"]
        finally:
            state.wandb_run_id = run.id
            run.finish()
        return state


class DryRunSink:
    """Prints what would be pushed; touches neither the network nor wandb credentials."""

    def push(self, payloads: List[Dict[str, Any]], state: RunState) -> RunState:
        for payload in payloads:
            print(
                "[dry-run] would push run=%r step=%s final=%s metrics=%s"
                % (payload["run_name"], payload["step"], payload["final"], payload["metrics"])
            )
            state.last_step = payload["step"]
            state.finished = state.finished or payload["final"]
        return state


# --------------------------------------------------------------------------------------
# one pass over the pool
# --------------------------------------------------------------------------------------
def premium_kernels(router: PoolRouter, accelerators: Sequence[str]) -> List[tuple]:
    """``(account, KernelRun)`` for every active kernel on one of ``accelerators``."""
    wanted = {normalize_accelerator(a) for a in accelerators}
    out = []
    for status in router.statuses:
        if not status.ok:
            continue
        for kernel in status.active_kernels:
            if kernel.accelerator in wanted:
                out.append((status.account, kernel))
    return out


def run_once(
    router: PoolRouter,
    state: MonitorState,
    sink: Any,
    accelerators: Sequence[str] = DEFAULT_ACCELERATORS,
    verbose: bool = True,
) -> int:
    """Probe the pool once and sync any new points. Returns how many kernels had new data."""
    router.probe_all(check_kernels=True, with_accelerators=True)
    targets = premium_kernels(router, accelerators)
    if verbose:
        print(
            "%d premium-accelerator kernel(s) active: %s"
            % (len(targets), ", ".join("%s(%s)" % (k.ref, k.accelerator) for _, k in targets))
            if targets else "no premium-accelerator kernels active"
        )

    synced = 0
    for account, kernel in targets:
        run_state = state.get(kernel.ref)
        # Plain `kaggle kernels logs` returns nothing until the kernel finishes (see
        # kernel_logs' docstring), which would make this only ever fire once, at the very
        # end, defeating the point of polling a still-running kernel -- so follow it instead.
        log_text = router.kernel_logs(account, kernel.ref, follow=(kernel.status == "running"))
        parsed = parse_sync_lines(log_text)
        if parsed and run_state.wandb_run_id and (
                parsed[-1]["run_name"] != run_state.run_name
                or (run_state.finished and not any(p["final"] for p in parsed))):
            # A new version of the same notebook: state is keyed by kernel ref, so without
            # this the new run would be appended to the previous run's W&B id and its first
            # steps skipped. (Rows written before run_name was stored count as a new run.)
            run_state = RunState()
        if run_state.finished:
            continue
        payloads = new_points(parsed, run_state.last_step)
        if not payloads:
            continue
        run_state = sink.push(payloads, run_state)
        run_state.run_name = payloads[0]["run_name"]
        state.set(kernel.ref, run_state)
        synced += 1
        if verbose:
            print(
                "%s: synced %d point(s) up to step %s%s"
                % (kernel.ref, len(payloads), run_state.last_step,
                   " (final)" if run_state.finished else "")
            )

    # A kernel that finished between two polls is no longer active, so the loop above never
    # reads its last epochs or the final evaluation. Read the full log of every tracked,
    # unfinished run once it drops out of the active list, then close it either way (a
    # crashed run has no final line to wait for).
    active = {kernel.ref for _, kernel in targets}
    by_user = {status.username.lower(): status for status in router.statuses}
    for ref in state.refs():
        run_state = state.get(ref)
        if ref in active or run_state.finished or not run_state.run_name:
            continue
        status = by_user.get(ref.split("/")[0].lower())
        if status is None or not status.ok:
            continue  # unknown or unreachable this poll: try again next time
        parsed = [p for p in parse_sync_lines(router.kernel_logs(status.account, ref, follow=False))
                  if p["run_name"] == run_state.run_name]
        payloads = new_points(parsed, run_state.last_step)
        if payloads:
            run_state = sink.push(payloads, run_state)
            synced += 1
        run_state.finished = True
        state.set(ref, run_state)
        if verbose:
            print("%s: no longer active, synced its last %d point(s), closed"
                  % (ref, len(payloads)))
    state.save()
    return synced


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tail no-internet (RTX/4xL4/TPU) Kaggle runs and sync them to W&B.",
    )
    parser.add_argument("--tokens-dir", default=os.environ.get("KAGGLE_TOKENS_DIR"))
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    parser.add_argument(
        "--accelerator", dest="accelerators", action="append",
        help="premium accelerator to watch (repeatable); default: %s" % ", ".join(DEFAULT_ACCELERATORS),
    )
    parser.add_argument("--interval", type=float, default=240.0, help="seconds between polls")
    parser.add_argument("--once", action="store_true", help="poll once and exit")
    parser.add_argument("--dry-run", action="store_true", help="print, don't touch wandb")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not args.tokens_dir:
        parser.error("--tokens-dir is required (or set KAGGLE_TOKENS_DIR)")
    args.accelerators = args.accelerators or list(DEFAULT_ACCELERATORS)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    router = PoolRouter.from_dir(args.tokens_dir)
    state = MonitorState(args.state_file)
    sink = DryRunSink() if args.dry_run else WandbSink()

    if args.once:
        run_once(router, state, sink, args.accelerators, verbose=not args.quiet)
        return 0

    print("watching the pool every %.0fs (ctrl-C to stop)" % args.interval)
    try:
        while True:
            try:
                run_once(router, state, sink, args.accelerators, verbose=not args.quiet)
            except Exception as exc:  # a transient CLI/network failure must not end the watch
                print("poll failed, retrying next interval: %s: %s" % (type(exc).__name__, exc))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
