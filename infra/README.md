# Infra: Kaggle resource router

`kaggle_router.py` answers one question for the team: **whose Kaggle account can run this
notebook right now?** It reads a directory of `kaggle.json` files (one per account), then for
each account reports the weekly accelerator quota left and whether a session is already running.

Stdlib only, read-only against the Kaggle public API. API keys are never printed or serialized.

## Credentials layout

```
~/kaggle_tokens/
├── account1_alice.json      # {"username": "...", "key": "..."}
├── account2_bob.json
└── ...
```

Point the tool at that directory with `--tokens-dir`, or export `KAGGLE_TOKENS_DIR`.
Keep the directory outside the repo (`.gitignore` blocks `kaggle*.json` as a backstop).

## CLI

```bash
export KAGGLE_TOKENS_DIR=~/kaggle_tokens

python infra/kaggle_router.py                           # full table
python infra/kaggle_router.py --running                 # only active kernels
python infra/kaggle_router.py --best --idle-only \
                              --min-gpu-hours 6         # pick one account to use
python infra/kaggle_router.py --no-kernels              # quota only (~1s, skips status calls)
python infra/kaggle_router.py --json                    # machine readable
```

Example:

```
ACCOUNT                  KAGGLE USER      GPU LEFT   TPU LEFT   RUNNING  STATE
------------------------------------------------------------------------------
account5_alice           alice             30.0/30h   20.0/20h  0        FREE
account1_bob             bob               21.1/30h   20.0/20h  1        BUSY
```

## Library

```python
from infra.kaggle_router import KaggleRouter

router = KaggleRouter.from_dir("~/kaggle_tokens")
router.probe_all()

pick = router.best(min_gpu_hours=6, require_idle=True)   # AccountStatus or None
print(pick.username, pick.gpu.remaining_h)

pick.account.write_kaggle_json()        # install as ~/.kaggle/kaggle.json
env = pick.account.env()                # {"KAGGLE_USERNAME": ..., "KAGGLE_KEY": ...}

router.running()                        # every running/queued kernel in the pool
router.total_remaining_hours("gpu")
```

## What the numbers mean

| Field | Source | Notes |
|---|---|---|
| `gpu` / `tpu` quota | `GET /kernels/quota` | Weekly budget: 30 GPU-h, 20 TPU-h per account |
| `remaining_h` | `allowed - used - reserved` | `reserved` is quota held by a session running now |
| `quota_refresh` | `quotaRefreshTime` | Weekly reset timestamp (UTC) |
| `active_kernels` | `/kernels/list` + `/kernels/status` | Statuses `running`, `queued`, `cancelRequested` |
| `free_slots` | `--max-concurrent` (default 2) | Kaggle does not expose the concurrency cap; tune this |

Only the `--recent N` (default 10) most recently run kernels per account are status-checked, so a
session started from a kernel outside that window can be missed — raise `--recent` if needed.

## Caveats

- Kaggle's Terms of Service treat accounts as personal and prohibit sharing credentials. This
  tool coordinates a pool of accounts; using it to run work on someone else's account is between
  you and that ToS.
- Quota figures are per account and reset weekly (`quotaRefreshTime`), not per notebook.

---

# Launching runs: `kaggle_launcher.py`

`KaggleLauncher` turns "run this notebook somewhere with a T4" into a push, choosing the account
via `KaggleRouter`. Accelerator choice rides in the push metadata as `machineShape` — the legacy
`kaggle.json` basic-auth path carries it fine, no new CLI needed.

## CLI

```bash
python infra/kaggle_launcher.py     --tokens-dir ~/kaggle_tokens     --title clifford-flow-default     --repo https://github.com/optimist1938/Clifford-Flow-Matching.git     --command "python -m src.main"     --accelerator t4 --min-hours 6 --idle-only --wait 3600
```

`--dry-run` prints the chosen account and the exact push body without touching Kaggle.

## Library

```python
from infra.kaggle_router import KaggleRouter
from infra.kaggle_launcher import KaggleLauncher, git_run_spec

router = KaggleRouter.from_dir("~/kaggle_tokens"); router.probe_all()
launcher = KaggleLauncher(router)

spec = git_run_spec(
    title="clifford-flow-default",
    repo="https://github.com/optimist1938/Clifford-Flow-Matching.git",
    command="python -m src.main",      # defaults => --model clifford
    accelerator="t4",
)
handle = launcher.launch(spec, min_hours=6, require_idle=True)
print(handle.url, handle.wait(timeout=3600))
```

`git_run_spec` generates a four-cell notebook: preflight (python version + `nvidia-smi`, plus a
secret check when secrets are declared) -> `git clone --depth 1` -> install -> run. For an existing
notebook use `NotebookSpec(title=..., notebook_path="run.ipynb", accelerator="l4")`.

`launch_many([...])` spreads several specs across distinct accounts.

## Accelerators

| Alias | `machineShape` | Notes |
|---|---|---|
| `cpu` / `none` | *(omitted)* | No accelerator quota consumed |
| `t4` | `NvidiaTeslaT4` | Confirmed in use across the pool |
| `p100` | `NvidiaTeslaP100` | Default image's torch may lack `sm_60` kernels |
| `l4` | `NvidiaL4` | Confirmed |
| `rtx6000` | `NvidiaRtxPro6000` | Confirmed |
| `tpu-v3` / `tpu-v5e` | `Tpu1VmV38` / `TpuV5E8` | **Accepted but silently non-TPU** ([#1197](https://github.com/Kaggle/kaggle-cli/issues/1197)) |

There is no value for the editor's "GPU T4 ×2" ([#1196](https://github.com/Kaggle/kaggle-cli/issues/1196)).

## Secrets are per-account and cannot be provisioned via the API

There is no secrets endpoint. If a notebook calls `UserSecretsClient` (e.g. `wandb_api_key`, a
GitHub PAT for a private repo), each account must have that secret added by hand under
**Add-ons > Secrets**, or the run fails on a routed account. Declare them via
`required_secrets=[...]` / `--secret` so the preflight cell fails in seconds instead of hours.

The Clifford-Flow-Matching default run needs **no secrets**: the repo is public, and `run_name`
defaults to `None`, which makes `wandb_create_run` a no-op.

## No live logs

`handle.log()` and `handle.output_files()` only return content **after** the kernel finishes —
Kaggle serves a pushed kernel's output on completion, and the public API has no log streaming.
While a run is `running`, expect an empty log. `handle.status()` is the only live signal
(`queued` -> `running` -> `complete` / `error` / `cancelAcknowledged`).

For progress on a multi-hour run, log to W&B from inside the notebook: pass `--run_name` to
`src.main` and add the `wandb_api_key` secret to the target account.

## Tests

```bash
python3 infra/test_kaggle_launcher.py     # 11 offline tests, no network or credentials
```
