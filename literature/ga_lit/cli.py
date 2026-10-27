"""Command-line interface: ``python -m ga_lit <command>`` (run from ``literature/``).

Commands
--------
update   fetch new candidates from all sources since the last run (or ``--since``), triage them into the
         inbox, save the database, record the run in ``state.yaml`` and re-render the views. The raw
         candidates are cached in ``.cache/candidates.json``; ``update --from-cache`` re-triages them
         without touching the network (useful when tuning ``config.py``).
render   re-render ``site/index.html`` and ``catalog.md`` from the database.
inbox    list inbox items (``--markdown`` prints a checklist suitable for a pull-request body).
accept   ``accept ID --relevance core --topics hep,theory [--summary "..."]`` moves an item to accepted.
reject   ``reject ID [--note "..."]`` moves an item to rejected (kept so it is never re-proposed).
stats    print counts by status, relevance and topic.
validate check the database for schema problems and duplicate keys.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import json

from .db import LIT_DIR, RELEVANCES, PaperDB, load_state, save_state
from .fetch import SOURCES, fetch_all
from .render import render_all
from .triage import triage

OVERLAP_DAYS = 7  # re-scan this many days before the last run so late-indexed papers are not missed


def cmd_update(args: argparse.Namespace) -> int:
    db = PaperDB.load()
    state = load_state()
    until = date.fromisoformat(args.until) if args.until else date.today()
    if args.since:
        since = date.fromisoformat(args.since)
    elif state.get("last_run"):
        since = date.fromisoformat(state["last_run"]) - timedelta(days=OVERLAP_DAYS)
    else:
        since = date(2026, 1, 1)
    sources = args.sources.split(",") if args.sources else list(SOURCES)
    cache = LIT_DIR / ".cache" / "candidates.json"
    if args.from_cache:
        cached = json.loads(cache.read_text())
        since, until = date.fromisoformat(cached["since"]), date.fromisoformat(cached["until"])
        candidates = cached["candidates"]
    else:
        candidates = fetch_all(since, until, sources)
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps({"since": since.isoformat(), "until": until.isoformat(), "candidates": candidates}))
    counts = triage(db, candidates, min_score=args.min_score)
    db.save()
    state["last_run"] = until.isoformat()
    state.setdefault("runs", []).append({"date": date.today().isoformat(), "window": [since.isoformat(), until.isoformat()],
                                         "sources": sources, **counts})
    state["runs"] = state["runs"][-50:]
    save_state(state)
    render_all(db)
    print(f"update done: window {since}..{until}, {counts['inbox_new']} new inbox items, "
          f"{len(db.by_status('inbox'))} in inbox total")
    return 0


def cmd_render(_: argparse.Namespace) -> int:
    db = PaperDB.load()
    problems = db.validate()
    for p in problems:
        print("WARNING", p, file=sys.stderr)
    for path in render_all(db):
        print("wrote", path)
    return 0


def cmd_inbox(args: argparse.Namespace) -> int:
    db = PaperDB.load()
    items = sorted(db.by_status("inbox"), key=lambda r: -(r.get("triage") or {}).get("score", 0))
    if args.markdown:
        print(f"### Literature inbox ({len(items)} items)\n")
        for r in items:
            link = next(iter((r.get("links") or {}).values()), "")
            score = (r.get("triage") or {}).get("score", "?")
            print(f"- [ ] `{r['id']}` (score {score}) **{r['title']}** — {r.get('venue', '')}, {r.get('date', '')}. {link}")
    else:
        for r in items:
            score = (r.get("triage") or {}).get("score", "?")
            print(f"{score:>3}  {r['id']:<36} {str(r.get('date', '')):<10} {r['title'][:90]}")
        print(f"{len(items)} inbox items")
    return 0


def cmd_accept(args: argparse.Namespace) -> int:
    db = PaperDB.load()
    rec = db.get(args.id)
    if rec is None:
        print("no such record", args.id, file=sys.stderr)
        return 1
    rec["status"] = "accepted"
    rec["relevance"] = args.relevance
    if args.topics:
        rec["topics"] = args.topics.split(",")
    if args.summary:
        rec["summary"] = args.summary
    rec.pop("triage", None)
    problems = [p for p in db.validate() if p.startswith(rec["id"])]
    for p in problems:
        print("ERROR", p, file=sys.stderr)
    if problems:
        return 1
    db.save()
    render_all(db)
    print("accepted", rec["id"])
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    db = PaperDB.load()
    rec = db.get(args.id)
    if rec is None:
        print("no such record", args.id, file=sys.stderr)
        return 1
    rec["status"] = "rejected"
    if args.note:
        rec["notes"] = args.note
    rec.pop("triage", None)
    db.save()
    render_all(db)
    print("rejected", rec["id"])
    return 0


def cmd_stats(_: argparse.Namespace) -> int:
    db = PaperDB.load()
    from collections import Counter

    print("status   ", dict(Counter(r.get("status") for r in db.records)))
    print("relevance", dict(Counter(r.get("relevance") for r in db.records if r.get("status") == "accepted")))
    topics = Counter(t for r in db.records if r.get("status") == "accepted" for t in r.get("topics") or [])
    for t, n in topics.most_common():
        print(f"  {n:3d}  {db.topics.get(t, {}).get('name', t)}")
    return 0


def cmd_validate(_: argparse.Namespace) -> int:
    db = PaperDB.load()
    problems = db.validate()
    for p in problems:
        print(p)
    print(f"{len(problems)} problems, {len(db.records)} records")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ga_lit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("update")
    p.add_argument("--since", help="YYYY-MM-DD; default: last run minus 7 days (or 2026-01-01)")
    p.add_argument("--until", help="YYYY-MM-DD; default: today")
    p.add_argument("--sources", help=f"comma-separated subset of {','.join(SOURCES)}")
    p.add_argument("--min-score", type=int, default=3)
    p.add_argument("--from-cache", action="store_true", help="re-triage the candidates of the previous fetch")
    p.set_defaults(func=cmd_update)

    sub.add_parser("render").set_defaults(func=cmd_render)

    p = sub.add_parser("inbox")
    p.add_argument("--markdown", action="store_true")
    p.set_defaults(func=cmd_inbox)

    p = sub.add_parser("accept")
    p.add_argument("id")
    p.add_argument("--relevance", choices=RELEVANCES, default="core")
    p.add_argument("--topics", help="comma-separated topic ids from topics.yaml")
    p.add_argument("--summary")
    p.set_defaults(func=cmd_accept)

    p = sub.add_parser("reject")
    p.add_argument("id")
    p.add_argument("--note")
    p.set_defaults(func=cmd_reject)

    sub.add_parser("stats").set_defaults(func=cmd_stats)
    sub.add_parser("validate").set_defaults(func=cmd_validate)

    args = ap.parse_args(argv)
    return args.func(args)
