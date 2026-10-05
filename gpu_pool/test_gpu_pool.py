"""Offline tests for the router and launcher: no network, no credentials, no CLI needed.

    python3 gpu_pool/test_gpu_pool.py      # or: pytest gpu_pool/test_gpu_pool.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_pool.launcher import (
    ACCELERATORS,
    PoolLauncher,
    NotebookSpec,
    _is_capacity_error,
    _ref_from_push_output,
    _version_from_push_output,
    git_run_spec,
    normalize_ref,
)
from gpu_pool.router import (
    AccountStatus,
    filter_progress,
    normalize_accelerator,
    requires_competition,
    shape_label,
    PoolAccount,
    PoolRouter,
    KernelRun,
    Quota,
    _normalize_status,
    _parse_hours,
)

CLIFFORD_REPO = "https://github.com/optimist1938/Clifford-Flow-Matching.git"
ARC = "arc-prize-2026-arc-agi-3"        # a competition, needed for L4 / RTX 6000 / TPU


def _spec(**kw):
    kw.setdefault("title", "clifford-flow-default")
    kw.setdefault("repo", CLIFFORD_REPO)
    kw.setdefault("command", "python -m src.main")
    return git_run_spec(**kw)


# -- spec / accelerators ----------------------------------------------------------------
def test_accelerator_maps_to_machine_shape():
    assert _spec(accelerator="t4").machine_shape == "NvidiaTeslaT4"
    assert _spec(accelerator="cpu").machine_shape is None
    # competition-gated shapes need a competition cited
    assert _spec(accelerator="l4", competitions=[ARC]).machine_shape == "NvidiaL4"
    assert _spec(accelerator="rtx6000", competitions=[ARC]).machine_shape == "NvidiaRtxPro6000"
    assert _spec(accelerator="tpu-v5e", competitions=[ARC]).machine_shape == "TpuV5E8"


def test_resource_and_flags_follow_accelerator():
    gpu = _spec(accelerator="t4")
    assert (gpu.is_gpu, gpu.is_tpu, gpu.resource) == (True, False, "gpu")
    tpu = _spec(accelerator="tpu-v3", competitions=[ARC])
    assert (tpu.is_gpu, tpu.is_tpu, tpu.resource) == (False, True, "tpu")
    cpu = _spec(accelerator="cpu")
    assert (cpu.is_gpu, cpu.is_tpu, cpu.resource) == (False, False, "cpu")


def test_kernel_metadata_shape():
    meta = _spec(accelerator="t4").metadata("someuser")
    # the CLI's kernel-metadata.json is snake_case, and `id` is owner/slug (unlike the REST body)
    assert meta["id"] == "someuser/clifford-flow-default"
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["enable_gpu"] is True and "enable_tpu" not in meta
    assert meta["kernel_type"] == "notebook" and meta["language"] == "python"
    assert meta["is_private"] is True
    assert meta["code_file"] == "clifford-flow-default.ipynb"
    cpu = _spec(accelerator="cpu").metadata("someuser")
    assert "machine_shape" not in cpu and cpu["enable_gpu"] is False
    assert _spec(accelerator="tpu-v3", competitions=[ARC]).metadata("u")["enable_tpu"] is True


def test_write_bundle_is_what_the_cli_expects():
    spec = _spec(accelerator="t4", datasets=["owner/ds"])
    with tempfile.TemporaryDirectory() as d:
        spec.write_bundle("someuser", d)
        assert sorted(os.listdir(d)) == ["clifford-flow-default.ipynb", "kernel-metadata.json"]
        meta = json.load(open(os.path.join(d, "kernel-metadata.json")))
        assert meta["dataset_sources"] == ["owner/ds"]
        nb = json.load(open(os.path.join(d, meta["code_file"])))
        assert nb["nbformat"] == 4 and len(nb["cells"]) == 4


def test_generated_notebook_runs_the_default_model():
    nb = _spec(accelerator="t4").to_ipynb()
    assert all(c["cell_type"] == "code" for c in nb["cells"])
    source = "\n".join(c["source"] for c in nb["cells"])
    assert "nvidia-smi" in source
    assert "git clone" in source and CLIFFORD_REPO in source
    assert "poetry run python -m src.main" in source
    assert "--model" not in source          # the default is --model clifford


def test_subdir_installs_and_runs_inside_the_subdirectory():
    spec = _spec(accelerator="t4", subdir="3D Pose experiemtns")
    cells = [c["source"] for c in spec.to_ipynb()["cells"]]
    assert "shutil.rmtree('/kaggle/working/Clifford-Flow-Matching')" in cells[1]   # clone root
    assert cells[2].count("/kaggle/working/Clifford-Flow-Matching/3D Pose experiemtns") == 1
    assert "3D Pose experiemtns" in cells[3]


def test_secrets_produce_a_preflight_guard():
    spec = _spec(github_token_secret="github_token", required_secrets=["wandb_api_key"])
    source = spec.to_ipynb()["cells"][0]["source"]
    assert "UserSecretsClient" in source
    assert "wandb_api_key" in source and "github_token" in source
    assert "missing Kaggle secret" in source
    assert "UserSecretsClient" not in _spec().to_ipynb()["cells"][0]["source"]


def test_slugify_and_validation():
    assert _spec(title="Clifford Flow: DW4 run #2").slug == "clifford-flow-dw4-run-2"
    try:
        NotebookSpec(title="x", cells=["pass"], accelerator="h100")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown accelerator")
    try:
        NotebookSpec(title="x")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError with neither cells nor notebook_path")


# -- CLI output parsing -----------------------------------------------------------------
def test_parse_hours_from_cli_strings():
    assert _parse_hours("0.14h") == 0.14
    assert _parse_hours("29.86h") == 29.86
    assert _parse_hours("30.00h") == 30.0
    assert _parse_hours(None) == 0.0
    assert _parse_hours(12) == 12.0


def test_normalize_status_from_cli_text():
    # `kaggle kernels status` prints the enum repr, not our vocabulary
    assert _normalize_status('x/y has status "KernelWorkerStatus.QUEUED"') == "queued"
    assert _normalize_status('x/y has status "KernelWorkerStatus.RUNNING"') == "running"
    assert _normalize_status('x/y has status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert _normalize_status('x/y has status "KernelWorkerStatus.ERROR"') == "error"
    assert _normalize_status('has status "CANCEL_ACKNOWLEDGED"') == "cancelAcknowledged"
    assert _normalize_status("") == "unknown"


def test_push_output_parsing():
    out = ("Kernel version 3 successfully pushed. Please check progress at "
           "https://www.kaggle.com/code/grig0ryz/clifford-flow-default")
    assert _ref_from_push_output(out) == "grig0ryz/clifford-flow-default"
    assert _version_from_push_output(out) == 3
    assert _ref_from_push_output("no url here") is None
    assert _version_from_push_output("no version here") is None


def test_normalize_ref_handles_url_forms():
    assert normalize_ref("/code/grig0ryz/clifford-flow-default") == "grig0ryz/clifford-flow-default"
    assert normalize_ref("code/u/s") == "u/s"
    assert normalize_ref("u/s") == "u/s"
    assert normalize_ref("https://www.kaggle.com/code/u/s") == "u/s"
    assert normalize_ref("/code/u/s/versions/3") == "u/s"
    assert normalize_ref("", "fallback", "slug") == "fallback/slug"


def test_capacity_error_is_recognised():
    assert _is_capacity_error("push rejected: Maximum batch GPU session count of 2 reached.")
    assert not _is_capacity_error("Could not convert string to integer")
    assert not _is_capacity_error("")


# -- credentials ------------------------------------------------------------------------
def test_token_and_legacy_credentials_both_load():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "account1_alice"))
        with open(os.path.join(d, "account1_alice", "token"), "w") as fh:
            fh.write("TOKEN_VALUE_SECRET\n")
        with open(os.path.join(d, "account2_bob.json"), "w") as fh:
            json.dump({"username": "bob", "key": "KEY_VALUE_SECRET"}, fh)
        router = PoolRouter.from_dir(d)
        by_label = {a.label: a for a in router.accounts}
        assert sorted(by_label) == ["account1_alice", "account2_bob"]

        alice = by_label["account1_alice"]
        assert alice.uses_token and alice.env() == {"KAGGLE_API_TOKEN": "TOKEN_VALUE_SECRET"}
        bob = by_label["account2_bob"]
        assert not bob.uses_token
        assert bob.env() == {"KAGGLE_USERNAME": "bob", "KAGGLE_KEY": "KEY_VALUE_SECRET"}
        # secrets must not surface through repr
        assert "TOKEN_VALUE_SECRET" not in repr(alice)
        assert "KEY_VALUE_SECRET" not in repr(bob)


def test_access_token_credentials_load():
    # `kaggle auth login` writes ~/.kaggle/access_token; a directory of those (one per account,
    # or a symlink to ~/.kaggle itself) must load like the `token` layout.
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "account9"))
        with open(os.path.join(d, "account9", "access_token"), "w") as fh:
            fh.write("ACCESS_TOKEN_SECRET\n")
        router = PoolRouter.from_dir(d)
        (acct,) = router.accounts
        assert acct.label == "account9" and acct.uses_token
        assert acct.env() == {"KAGGLE_API_TOKEN": "ACCESS_TOKEN_SECRET"}
        assert acct.path.endswith("access_token")
        assert "ACCESS_TOKEN_SECRET" not in repr(acct)


def test_docker_image_lands_in_metadata():
    # Pinning the image that ran a previous job keeps its Python version (the offline wheels
    # are cp312); the CLI forwards docker_image + docker_image_pinning_type on push.
    image = "gcr.io/kaggle-private-byod/python@sha256:" + "0" * 64
    meta = _spec(accelerator="t4", docker_image=image).metadata("u")
    assert meta["docker_image"] == image and meta["docker_image_pinning_type"] == "original"
    plain = _spec(accelerator="t4").metadata("u")
    assert "docker_image" not in plain and "docker_image_pinning_type" not in plain


def test_quota_from_cli_row():
    q = Quota.from_row({"resource": "GPU", "used": "0.14h", "remaining": "29.86h",
                        "total": "30.00h", "refreshAt": "2026-10-03T00:00:00"})
    assert (q.kind, q.used_h, q.remaining_h, q.allowed_h) == ("gpu", 0.14, 29.86, 30.0)
    assert round(q.utilization, 4) == round(0.14 / 30.0, 4)
    assert Quota("gpu", allowed_h=0).utilization == 1.0


# -- routing, against a stubbed pool ----------------------------------------------------
def _fake_router(specs):
    """specs: list of (label, gpu_remaining_h, busy)"""
    router = PoolRouter([PoolAccount(label=n, token="t") for n, _, _ in specs])
    statuses = []
    for (name, hours, busy), account in zip(specs, router.accounts):
        status = AccountStatus(account=account, ok=True, max_concurrent=2)
        status.gpu = Quota("gpu", used_h=30 - hours, remaining_h=hours, allowed_h=30)
        status.tpu = Quota("tpu", remaining_h=20, allowed_h=20)
        status.kernels_checked = True
        if busy:
            status.kernels = [KernelRun(ref="%s/live" % name, status="running")]
        statuses.append(status)
    router._statuses = statuses
    return router


def test_router_picks_the_richest_account():
    launcher = PoolLauncher(_fake_router([("poor", 2.0, False), ("rich", 25.0, False)]))
    assert launcher.pick_account(_spec(accelerator="t4"), min_hours=1.0).label == "rich"


def test_idle_requirement_skips_busy_accounts():
    launcher = PoolLauncher(_fake_router([("busy", 25.0, True), ("idle", 10.0, False)]))
    spec = _spec(accelerator="t4")
    assert launcher.pick_account(spec, min_hours=1.0).label == "busy"
    assert launcher.pick_account(spec, min_hours=1.0, require_idle=True).label == "idle"


def test_cpu_runs_prefer_idle_then_most_gpu_left():
    launcher = PoolLauncher(
        _fake_router([("busy", 30.0, True), ("lean", 5.0, False), ("spare", 25.0, False)])
    )
    # CPU burns no accelerator quota, so it should avoid accounts wanted for GPU work
    assert launcher.pick_account(_spec(accelerator="cpu")).label == "spare"


def test_exhausted_pool_explains_itself():
    launcher = PoolLauncher(_fake_router([("a", 1.0, False), ("b", 0.5, False)]))
    try:
        launcher.pick_account(_spec(accelerator="t4"), min_hours=6.0)
    except RuntimeError as exc:
        assert "no account has >=6.0 spare gpu-hours" in str(exc)
        assert "pool total 1.5h" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_explicit_account_overrides_routing():
    launcher = PoolLauncher(_fake_router([("a", 1.0, False), ("b", 25.0, False)]))
    assert launcher.pick_account(_spec(accelerator="t4"), account="a").label == "a"
    try:
        launcher.pick_account(_spec(accelerator="t4"), account="nope")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


def test_next_candidate_skips_refused_accounts():
    launcher = PoolLauncher(
        _fake_router([("a", 25.0, False), ("b", 20.0, False), ("c", 10.0, False)])
    )
    spec = _spec(accelerator="t4")
    assert launcher._next_candidate(spec, 1.0, []).label == "a"
    assert launcher._next_candidate(spec, 1.0, ["a"]).label == "b"
    assert launcher._next_candidate(spec, 1.0, ["a", "b"]).label == "c"
    assert launcher._next_candidate(spec, 1.0, ["a", "b", "c"]) is None
    assert launcher._next_candidate(spec, 22.0, ["a"]) is None


def test_free_slots_and_busy_track_active_kernels():
    router = _fake_router([("one", 25.0, True)])
    status = router.statuses[0]
    assert status.busy and status.free_slots == 1
    status.kernels.append(KernelRun(ref="one/second", status="queued"))
    assert status.free_slots == 0
    status.kernels.append(KernelRun(ref="one/done", status="complete"))
    assert status.free_slots == 0 and len(status.active_kernels) == 2


def test_shape_label_handles_the_clis_literal_none():
    assert shape_label("NvidiaTeslaT4") == "t4"
    assert shape_label("NvidiaL4") == "l4"
    assert shape_label("NvidiaRtxPro6000") == "rtx6000"
    # `kaggle kernels pull -m` writes the STRING "None" for a CPU kernel
    assert shape_label("None") == "cpu"
    assert shape_label("none") == "cpu"
    assert shape_label("") == "cpu"
    assert shape_label(None) == "cpu"
    assert shape_label("SomethingNew") == "SomethingNew"   # unknown shapes pass through


def test_accelerators_seen_rolls_up_per_account():
    router = _fake_router([("a", 25.0, False)])
    status = router.statuses[0]
    status.kernels = [
        KernelRun("a/1", "complete", machine_shape="NvidiaTeslaT4"),
        KernelRun("a/2", "complete", machine_shape="NvidiaTeslaT4"),
        KernelRun("a/3", "running", machine_shape="NvidiaL4"),
        KernelRun("a/4", "complete", machine_shape="None"),      # CPU
        KernelRun("a/5", "complete"),                             # not collected -> ignored
    ]
    assert status.accelerators_seen == {"t4": 2, "cpu": 1, "l4": 1}
    assert status.active_accelerators == ["l4"]
    assert status.has_run_on("l4") and status.has_run_on("t4")
    assert not status.has_run_on("rtx6000")


def test_available_filters_by_accelerator_eligibility():
    router = _fake_router([("has_l4", 10.0, False), ("t4_only", 30.0, False)])
    router.statuses[0].kernels = [KernelRun("x/1", "complete", machine_shape="NvidiaL4")]
    router.statuses[1].kernels = [KernelRun("y/1", "complete", machine_shape="NvidiaTeslaT4")]
    # L4 is competition-gated: with no competition data, observed history is the evidence
    assert [s.name for s in router.available(accelerator="l4")] == ["has_l4"]
    # T4 needs no competition, so it never narrows the pool
    assert len(router.available(accelerator="t4")) == 2
    assert router.available(accelerator="rtx6000") == []
    assert len(router.available()) == 2


def test_competition_gated_accelerators():
    # Kaggle offers only T4/P100 outside a competition
    assert not requires_competition("t4")
    assert not requires_competition("p100")
    assert not requires_competition("cpu")
    assert requires_competition("l4")
    assert requires_competition("rtx6000")
    assert requires_competition("tpu-v5e")
    # accepts Kaggle's own shape spelling too
    assert requires_competition("NvidiaL4")
    assert not requires_competition("NvidiaTeslaT4")
    assert normalize_accelerator("NvidiaRtxPro6000") == "rtx6000"
    assert normalize_accelerator("L4") == "l4"


def test_can_use_needs_a_competition_for_premium_gpus():
    router = _fake_router([("entered", 25.0, False), ("bare", 25.0, False)])
    entered, bare = router.statuses
    for st in (entered, bare):
        st.competitions_checked = True
    entered.competitions = ["arc-prize-2026-arc-agi-3"]

    assert entered.can_use("t4") and bare.can_use("t4")        # always available
    assert entered.can_use("l4") and entered.premium_accelerators
    assert not bare.can_use("l4") and not bare.premium_accelerators
    assert [s.name for s in router.available(accelerator="l4")] == ["entered"]
    assert len(router.available(accelerator="t4")) == 2


def test_can_use_falls_back_to_history_when_competitions_unknown():
    router = _fake_router([("seen_l4", 25.0, False)])
    status = router.statuses[0]
    assert status.competitions_checked is False
    assert not status.can_use("l4")                             # no evidence either way
    status.kernels = [KernelRun("x/1", "complete", machine_shape="NvidiaL4")]
    assert status.can_use("l4")                                 # it demonstrably ran on one


def test_spec_refuses_premium_accelerator_without_a_competition():
    try:
        _spec(accelerator="l4")
    except ValueError as exc:
        assert "only available inside a competition" in str(exc)
    else:
        raise AssertionError("expected ValueError for l4 with no competition")
    # with the competition cited it is fine, and it lands in the metadata
    spec = _spec(accelerator="l4", competitions=["arc-prize-2026-arc-agi-2"])
    meta = spec.metadata("someuser")
    assert meta["machine_shape"] == "NvidiaL4"
    assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-2"]


def test_premium_routing_only_considers_eligible_accounts():
    router = _fake_router([("rich_bare", 30.0, False), ("lean_entered", 8.0, False)])
    for st in router.statuses:
        st.competitions_checked = True
    router.statuses[1].competitions = ["arc-prize-2026-arc-agi-3"]
    launcher = PoolLauncher(router)
    spec = _spec(accelerator="l4", competitions=["arc-prize-2026-arc-agi-3"])
    # the richest account cannot reach an L4 at all, so the lean eligible one wins
    assert launcher.pick_account(spec, min_hours=1.0).label == "lean_entered"
    # and a T4 spec still prefers the richest
    assert launcher.pick_account(_spec(accelerator="t4"), min_hours=1.0).label == "rich_bare"


def test_cli_json_treats_no_results_as_empty():
    router = _fake_router([("a", 1.0, False)])
    import gpu_pool.router as mod

    class _Proc:
        returncode = 0
        stderr = ""
        def __init__(self, out): self.stdout = out

    router.run_cli = lambda account, *args, **kw: _Proc(_Proc.out)   # type: ignore
    _Proc.out = "No competitions found"
    assert router.cli_json(router.accounts[0], "competitions", "list") == []
    _Proc.out = ""
    assert router.cli_json(router.accounts[0], "competitions", "list") == []
    _Proc.out = '[{"ref": "x"}]'
    assert router.cli_json(router.accounts[0], "competitions", "list") == [{"ref": "x"}]
    _Proc.out = "unexpected garbage"
    try:
        router.cli_json(router.accounts[0], "competitions", "list")
    except mod.CliError:
        pass
    else:
        raise AssertionError("expected CliError on unparseable output")


def test_filter_progress_keeps_only_informative_lines():
    log = "\n".join([
        "python 3.12.13",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB",
        " 62%|#######   | 356/575 [00:23<00:14, 14.97it/s]",
        " 63%|#######   | 360/575 [00:24<00:14, 14.98it/s]",
        "   |     ",
        "[timing] train: loaded 18371 samples from cache pascal_train.pt in 15.1s",
        "Training on cuda:0 epoch 1 / 100.",
        "Training on cuda:0 epoch 1 / 100.",          # duplicate, dropped
        "Median rotation error 12.01 (46s)",
    ])
    kept = filter_progress(log)
    assert kept == [
        "python 3.12.13",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB",
        "[timing] train: loaded 18371 samples from cache pascal_train.pt in 15.1s",
        "Training on cuda:0 epoch 1 / 100.",
        "Median rotation error 12.01 (46s)",
    ]
    assert filter_progress(log, keep_last=2) == kept[-2:]
    assert filter_progress("") == []


def test_kernel_logs_follow_uses_the_streaming_flag():
    router = _fake_router([("a", 1.0, False)])
    calls = []

    class _Proc:
        returncode = 0
        stdout = "live output"
        stderr = ""

    def fake_run_cli(account, *args, **kw):
        calls.append((args, kw))
        return _Proc()

    router.run_cli = fake_run_cli            # type: ignore
    acct = router.accounts[0]

    router.kernel_logs(acct, "u/s")
    assert calls[-1][0] == ("kernels", "logs", "u/s")
    assert "partial_on_timeout" not in calls[-1][1]

    router.kernel_logs(acct, "u/s", follow=True, follow_seconds=12)
    args, kw = calls[-1]
    assert args == ("kernels", "logs", "-f", "u/s")
    assert kw["partial_on_timeout"] is True and kw["timeout"] == 12


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
