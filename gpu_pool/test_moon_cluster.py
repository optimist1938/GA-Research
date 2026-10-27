"""Offline tests for the Moon Cluster page: no network, no credentials, no CLI needed.

    python3 gpu_pool/test_moon_cluster.py      # or: pytest gpu_pool/test_moon_cluster.py
"""

import copy
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_pool.moon_cluster import (
    changes,
    load_snapshot,
    main,
    page_data,
    parse_label,
    render,
    scan_tokens_dir,
    spread,
)

ARC3, ARC2, GEMMA = "arc-prize-2026-arc-agi-3", "arc-prize-2026-arc-agi-2", "gemma-4-developer-agent"


def _acct(n, name, h, routes=None, runs=None, status="live"):
    return {"label": "account%d_%s" % (n, name), "n": n, "name": name, "user": name.lower(), "status": status,
            "h": h, "allowed": 30.0, "runs": runs or [], "routes": routes or {}}


def _snapshot():
    return {
        "version": 1, "title": "Moon Cluster", "subtitle": None,
        "taken_at": "2026-10-27T16:22:00Z", "quota_reset": "2026-10-31T00:00:00Z", "max_concurrent": 2,
        "accounts": [
            _acct(10, "AwesomeArina", 24.0, {GEMMA: {"hw": "l4", "how": "seen"}}, [["l4", "symsol-l4x4"]]),
            _acct(3, "saudabubin", 30.0, {ARC3: {"hw": "rtx6000", "how": "seen"}, ARC2: {"hw": "l4", "how": "rules"}}),
            _acct(5, "GrigoryZ", 20.5, {ARC3: {"hw": "rtx6000", "how": "seen"}}, [["rtx6000", "a"], ["rtx6000", "b"]]),
            _acct(4, "saudic", 24.9),
            {"label": "account12_GloriousStefany", "n": 12, "name": "GloriousStefany", "status": "pending", "runs": [], "routes": {}},
        ],
        "competitions": {ARC3: {"deadline": "2026-11-02T23:59:00Z"}, ARC2: {"deadline": "2026-11-02T23:59:00Z"},
                         GEMMA: {"deadline": "2026-12-02T23:59:00Z"}},
        "notes": [],
    }


def _embedded(page):
    match = re.search(r'<script type="application/json" id="moon-data">(.*?)</script>', page, re.S)
    return json.loads(match.group(1))


def test_parse_label_reads_number_name_and_tag():
    assert parse_label("account7_CityOfTver[suspended]") == (7, "CityOfTver", "suspended")
    assert parse_label("Account13_GreatestNarek") == (13, "GreatestNarek", None)
    assert parse_label("legacy") == (None, "legacy", None)


def test_render_fills_every_placeholder_and_embeds_the_snapshot():
    page = render(_snapshot())
    assert "{{" not in page
    assert "<title>Moon Cluster</title>" in page
    data = _embedded(page)
    assert data["snapshot"]["accounts"][0]["name"] == "AwesomeArina"
    assert set(data["hardware"]) == {"rtx6000", "l4", "open"}
    assert "1 more awaiting token" in data["meta"]["live_note"]


def test_layout_spaces_accounts_and_competitions():
    data = page_data(_snapshot())
    angles = [a["n"] for a in data["accounts"]]
    assert len(angles) == 5 and angles[-1] == 12                  # pending accounts go last
    comps = sorted(c["angle"] for c in data["competitions"])
    gaps = [(b - a) % 360 for a, b in zip(comps, comps[1:] + comps[:1])]
    assert min(gaps) >= 35.9
    assert data["competitions"][0]["days"] in (6, 36)
    hw = sorted(data["hardware_angles"].values())
    assert (hw[1] - hw[0]) % 360 >= 79.9


def test_spread_keeps_far_angles_and_separates_close_ones():
    out = spread({"a": 10.0, "b": 12.0, "c": 180.0}, min_gap=40)
    assert abs(((out["b"] - out["a"]) % 360) - 40) < 1e-6
    assert out["c"] == 180.0


def test_hardware_note_names_the_competition_and_the_quota_rate():
    hw = page_data(_snapshot())["hardware"]
    assert "ARC-AGI-3, which closes 2 Nov" in hw["rtx6000"]["note"]
    assert "Billed at 2× quota" in hw["l4"]["note"] and hw["l4"]["rate"] == 2.0


def test_changes_against_a_previous_snapshot():
    prev = _snapshot()
    now = copy.deepcopy(prev)
    prev["accounts"] = [a for a in prev["accounts"] if a["n"] != 10]              # 10 is new
    prev["accounts"].append(_acct(7, "CityOfTver", 24.8))                          # 7 disappears
    now["accounts"][2]["routes"][GEMMA] = {"hw": "l4", "how": "seen"}              # 5 joins Gemma 4
    now["accounts"].append({"label": "account7_CityOfTver", "n": 7, "name": "CityOfTver",
                            "status": "removed", "reason": "deleted", "runs": [], "routes": {}})
    cards = {c["kicker"]: c for c in changes(now, prev)}
    assert cards["new account"]["title"] == "10 · AwesomeArina"
    assert "4xL4 through Gemma 4" in cards["new account"]["text"]
    assert cards["new route"]["title"] == "Account 5 joined Gemma 4"
    assert "open to 3 accounts, up from 1" in cards["new route"]["text"]
    assert cards["removed"]["title"] == "Account 7 deleted" and "24.8 h" in cards["removed"]["text"]
    assert cards["awaiting tokens"]["title"] == "Account 12"
    assert cards["ending soon"]["title"] == "RTX Pro 6000 ends 2 Nov"


def test_notes_come_first():
    snap = _snapshot()
    snap["notes"] = [{"kicker": "note", "title": "Maintenance", "text": "Pool paused tonight.", "tone": "--l4"}]
    assert changes(snap)[0]["title"] == "Maintenance"


def test_a_rendered_page_is_its_own_previous_snapshot():
    page = render(_snapshot())
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "page.html")
        with open(path, "w") as fh:
            fh.write(page)
        assert load_snapshot(path) == _snapshot()


def test_text_is_escaped_and_cannot_close_the_data_tag():
    snap = _snapshot()
    snap["notes"] = [{"kicker": "x", "title": "<b>bold</b>", "text": "</script><script>alert(1)</script>"}]
    page = render(snap)
    assert "<b>bold</b>" not in page and "&lt;b&gt;bold&lt;/b&gt;" in page
    assert page.count("</script>") == page.count("<script")             # nothing closes a tag early
    assert _embedded(page)["snapshot"]["notes"][0]["text"].startswith("</script>")


def test_scan_tokens_dir_sorts_live_pending_and_removed():
    with tempfile.TemporaryDirectory() as tmp:
        for name, token in (("account1_a", "t1"), ("account7_b[suspended]", "t7"), ("account9_c", "t9"),
                            ("account12_d", None)):
            os.makedirs(os.path.join(tmp, name))
            if token:
                with open(os.path.join(tmp, name, "token"), "w") as fh:
                    fh.write(token)
        accounts, pending, removed = scan_tokens_dir(tmp, {"account9": "deleted"})
    assert [a.label for a in accounts] == ["account1_a"]
    assert pending == ["account12_d"]
    assert removed == {"account7_b[suspended]": "suspended", "account9_c": "deleted"}


def test_cli_renders_from_a_snapshot_with_notes_and_removals():
    with tempfile.TemporaryDirectory() as tmp:
        snap_path, out = os.path.join(tmp, "s.json"), os.path.join(tmp, "p.html")
        with open(snap_path, "w") as fh:
            json.dump(_snapshot(), fh)
        assert main(["--from-snapshot", snap_path, "--out", out, "--subtitle", "night shift",
                     "--note", "note|Heads up|RTX closes Sunday", "--removed", "account4=deleted"]) == 0
        page = open(out).read()
    assert "<em>night shift</em>" in page and "Heads up" in page and "Account 4 deleted" in page
    assert all(a["n"] != 4 for a in _embedded(page)["accounts"])


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
