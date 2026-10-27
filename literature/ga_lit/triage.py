"""Turn raw candidates into inbox records.

The *triage score* of a candidate is a small integer summarising how much it looks like a Clifford /
geometric-algebra machine-learning paper:

    score = (number of distinct GA terms matched)
          + (2 if a machine-learning term matched)
          - (3 if a name-collision term matched)
          - (1 for an arXiv paper with no cs.*, stat.*, eess.* or q-bio category)

computed on title + abstract (see ``config``). A candidate enters the inbox when it has a machine-learning
term **and** either ``score >= MIN_SCORE`` or a GA term in its title. Pure mathematics or physics papers about
Clifford algebras therefore stay out, however many GA terms they use. Candidates already in the database only
contribute new identifiers and ``found_by`` sources to the existing record.
"""

from __future__ import annotations

import sys
from datetime import date

from . import config
from .db import PaperDB

MIN_SCORE = 3
ML_CATEGORY_PREFIXES = ("cs.", "stat.", "eess.", "q-bio", "econ.")


def score(candidate: dict) -> tuple[int, dict]:
    text = f"{candidate.get('title', '')} {candidate.get('abstract', '')}"
    ga = sorted({m.group(0).lower() for m in config.GA_TERMS.finditer(text)})
    ml = bool(config.ML_TERMS.search(text))
    excluded = sorted({m.group(0).lower() for m in config.EXCLUDE_TERMS.finditer(text)})
    title_hit = bool(config.GA_TERMS.search(candidate.get("title", "")))
    cats = candidate.get("categories") or []
    off_topic_arxiv = bool(cats) and not any(c.startswith(ML_CATEGORY_PREFIXES) for c in cats)
    s = len(ga) + (2 if ml else 0) - (3 if excluded else 0) - (1 if off_topic_arxiv else 0)
    return s, {"ga_terms": ga, "ml_context": ml, "excluded_terms": excluded, "title_hit": title_hit}


def qualifies(s: int, details: dict, min_score: int = MIN_SCORE) -> bool:
    return details["ml_context"] and (s >= min_score or details["title_hit"])


def suggest_topics(candidate: dict) -> list[str]:
    text = f"{candidate.get('title', '')} {candidate.get('abstract', '')}"
    return [t for t, rx in config.TOPIC_HINTS.items() if rx.search(text)] or ["architecture"]


def to_inbox_record(candidate: dict, s: int, details: dict) -> dict:
    rec = {
        "id": candidate["id"],
        "title": candidate["title"],
        "authors": candidate.get("authors") or [],
        "date": candidate.get("date"),
        "venue": candidate.get("venue") or "",
        "venue_type": candidate.get("venue_type") or "other",
        "links": {k: v for k, v in (candidate.get("links") or {}).items() if v},
        "topics": suggest_topics(candidate),
        "status": "inbox",
        "relevance": None,
        "summary": (candidate.get("abstract") or "")[:400],
        "triage": {"score": s, **details},
        "found_by": sorted(set(candidate.get("found_by") or [])),
        "added": date.today().isoformat(),
    }
    return rec


def triage(db: PaperDB, candidates: list[dict], min_score: int = MIN_SCORE) -> dict:
    """Merge ``candidates`` into ``db``. Returns counts of what happened."""
    counts = {"candidates": len(candidates), "known": 0, "known_updated": 0, "inbox_new": 0, "below_threshold": 0}
    for cand in candidates:
        existing = db.find(cand)
        if existing is not None:
            counts["known"] += 1
            if db.merge_alias(existing, cand):
                counts["known_updated"] += 1
            continue
        s, details = score(cand)
        if not qualifies(s, details, min_score):
            counts["below_threshold"] += 1
            continue
        db.add(to_inbox_record(cand, s, details))
        counts["inbox_new"] += 1
    print(f"triage: {counts}", file=sys.stderr)
    return counts
