"""Offline tests for the launcher: no network, no credentials needed.

    python3 infra/test_kaggle_launcher.py      # or: pytest infra/test_kaggle_launcher.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.kaggle_launcher import (
    ACCELERATORS,
    KaggleLauncher,
    NotebookSpec,
    git_run_spec,
    normalize_ref,
)
from infra.kaggle_router import AccountStatus, KaggleAccount, KaggleRouter, Quota

CLIFFORD_REPO = "https://github.com/optimist1938/Clifford-Flow-Matching.git"


def _spec(**kw):
    kw.setdefault("title", "clifford-flow-default")
    kw.setdefault("repo", CLIFFORD_REPO)
    kw.setdefault("command", "python -m src.main")
    return git_run_spec(**kw)


def test_accelerator_maps_to_machine_shape():
    assert _spec(accelerator="t4").machine_shape == "NvidiaTeslaT4"
    assert _spec(accelerator="l4").machine_shape == "NvidiaL4"
    assert _spec(accelerator="rtx6000").machine_shape == "NvidiaRtxPro6000"
    assert _spec(accelerator="cpu").machine_shape is None
    assert _spec(accelerator="tpu-v5e").machine_shape == "TpuV5E8"


def test_resource_and_flags_follow_accelerator():
    gpu = _spec(accelerator="t4")
    assert (gpu.is_gpu, gpu.is_tpu, gpu.resource) == (True, False, "gpu")
    tpu = _spec(accelerator="tpu-v3")
    assert (tpu.is_gpu, tpu.is_tpu, tpu.resource) == (False, True, "tpu")
    cpu = _spec(accelerator="cpu")
    assert (cpu.is_gpu, cpu.is_tpu, cpu.resource) == (False, False, "cpu")


def test_push_body_shape():
    body = _spec(accelerator="t4").push_body("someuser")
    # "username/kernel-slug" goes in `slug`; `id` is the numeric kernel id and must be
    # absent when creating a new kernel (the API rejects a string there with HTTP 400).
    assert body["slug"] == "someuser/clifford-flow-default"
    assert "id" not in body
    assert NotebookSpec(title="x", cells=["pass"], kernel_id=123).push_body("u")["id"] == 123
    assert body["machineShape"] == "NvidiaTeslaT4"
    assert body["enableGpu"] is True and body["enableTpu"] is False
    assert body["kernelType"] == "notebook" and body["language"] == "python"
    assert body["isPrivate"] is True
    # cpu runs must not advertise an accelerator at all
    assert "machineShape" not in _spec(accelerator="cpu").push_body("someuser")


def test_generated_notebook_is_valid_and_runs_the_default_model():
    nb = _spec(accelerator="t4").to_ipynb()
    assert nb["nbformat"] == 4 and len(nb["cells"]) == 4
    assert all(c["cell_type"] == "code" for c in nb["cells"])
    json.dumps(nb)  # must be serializable for the push body
    source = "\n".join(c["source"] for c in nb["cells"])
    assert "nvidia-smi" in source                      # preflight
    assert "git clone" in source and CLIFFORD_REPO in source
    assert "poetry run python -m src.main" in source    # defaults => --model clifford
    assert "--model" not in source                      # no override of the default


def test_secrets_produce_a_preflight_guard():
    spec = _spec(github_token_secret="github_token", required_secrets=["wandb_api_key"])
    source = spec.to_ipynb()["cells"][0]["source"]
    assert "UserSecretsClient" in source
    assert "wandb_api_key" in source and "github_token" in source
    assert "missing Kaggle secret" in source
    # a public repo needs no token, so no secret guard at all
    assert "UserSecretsClient" not in _spec().to_ipynb()["cells"][0]["source"]


def test_slugify_and_validation():
    assert _spec(title="Clifford Flow: DW4 run #2").slug == "clifford-flow-dw4-run-2"
    for bad in (dict(accelerator="h100"), dict()):
        try:
            NotebookSpec(title="x", **bad) if bad else NotebookSpec(title="x")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % bad)


# -- routing, against a stubbed pool ----------------------------------------------------
def _fake_router(specs):
    """specs: list of (label, gpu_remaining_h, busy)"""
    router = KaggleRouter([KaggleAccount(username=n, key="x", label=n) for n, _, _ in specs])
    statuses = []
    for (name, hours, busy), account in zip(specs, router.accounts):
        status = AccountStatus(account=account, ok=True, max_concurrent=2)
        status.gpu = Quota("gpu", used_s=(30 - hours) * 3600, allowed_s=30 * 3600)
        status.tpu = Quota("tpu", allowed_s=20 * 3600)
        status.kernels_checked = True
        if busy:
            from infra.kaggle_router import KernelRun
            status.kernels = [KernelRun(ref="%s/live" % name, status="running")]
        statuses.append(status)
    router._statuses = statuses
    return router


def test_router_picks_the_richest_account():
    launcher = KaggleLauncher(_fake_router([("poor", 2.0, False), ("rich", 25.0, False)]))
    assert launcher.pick_account(_spec(accelerator="t4"), min_hours=1.0).username == "rich"


def test_idle_requirement_skips_busy_accounts():
    launcher = KaggleLauncher(_fake_router([("busy", 25.0, True), ("idle", 10.0, False)]))
    spec = _spec(accelerator="t4")
    assert launcher.pick_account(spec, min_hours=1.0).username == "busy"
    assert launcher.pick_account(spec, min_hours=1.0, require_idle=True).username == "idle"


def test_exhausted_pool_explains_itself():
    launcher = KaggleLauncher(_fake_router([("a", 1.0, False), ("b", 0.5, False)]))
    try:
        launcher.pick_account(_spec(accelerator="t4"), min_hours=6.0)
    except RuntimeError as exc:
        assert "no account has >=6.0 spare gpu-hours" in str(exc)
        assert "pool total 1.5h" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_explicit_account_overrides_routing():
    launcher = KaggleLauncher(_fake_router([("a", 1.0, False), ("b", 25.0, False)]))
    assert launcher.pick_account(_spec(accelerator="t4"), account="a").username == "a"
    try:
        launcher.pick_account(_spec(accelerator="t4"), account="nope")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


def test_credentials_never_leak_into_repr():
    account = KaggleAccount(username="u", key="SECRET_KEY_VALUE", label="l")
    assert "SECRET_KEY_VALUE" not in repr(account)
    assert account.env()["KAGGLE_KEY"] == "SECRET_KEY_VALUE"


def test_normalize_ref_handles_push_response_forms():
    # the push response uses a site path, not the owner/slug every other endpoint wants
    assert normalize_ref("/code/grig0ryz/clifford-flow-default") == "grig0ryz/clifford-flow-default"
    assert normalize_ref("code/u/s") == "u/s"
    assert normalize_ref("u/s") == "u/s"
    assert normalize_ref("https://www.kaggle.com/code/u/s") == "u/s"
    assert normalize_ref("/code/u/s/versions/3") == "u/s"
    assert normalize_ref("/code/u/my%20slug") == "u/my slug"
    assert normalize_ref("", "fallback", "slug") == "fallback/slug"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok   %s" % name)
        except Exception as exc:
            failed += 1
            print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    raise SystemExit(1 if failed else 0)
