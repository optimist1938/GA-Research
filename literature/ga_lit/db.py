"""Load, index and save the paper database (``literature/papers.yaml``).

The database is a YAML list; each element is one paper record with the fields documented in
``literature/README.md``. Records are matched to incoming candidates by any of their identifiers
(arXiv id without version, DOI, OpenReview id) or by a normalised title, so the same paper arriving from
two sources is never stored twice.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

LIT_DIR = Path(__file__).resolve().parent.parent
PAPERS_PATH = LIT_DIR / "papers.yaml"
TOPICS_PATH = LIT_DIR / "topics.yaml"
STATE_PATH = LIT_DIR / "state.yaml"

STATUSES = ("accepted", "inbox", "rejected")
RELEVANCES = ("core", "adjacent", "context")
VENUE_TYPES = ("arxiv", "conference", "journal", "openreview", "preprint-server", "book", "other")


def norm_title(title: str) -> str:
    """Lower-case, alphanumeric-only prefix of a title, used as a fuzzy duplicate key."""
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())[:80]


def norm_arxiv(aid: str) -> str:
    """Strip the version suffix and any URL prefix from an arXiv identifier."""
    aid = aid.strip()
    aid = re.sub(r"^.*arxiv\.org/(abs|pdf)/", "", aid)
    aid = re.sub(r"v\d+$", "", aid)
    return aid


def norm_doi(doi: str) -> str:
    doi = doi.strip().lower()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)


def keys_of(rec: dict) -> set[str]:
    """All duplicate-detection keys of a record: identifiers, aliases and the normalised title."""
    keys = {rec["id"]}
    keys.update(rec.get("aliases") or [])
    links = rec.get("links") or {}
    if links.get("arxiv"):
        keys.add("arxiv:" + norm_arxiv(links["arxiv"]))
    if links.get("doi"):
        keys.add("doi:" + norm_doi(links["doi"]))
    if links.get("openreview"):
        keys.add("openreview:" + re.sub(r"^.*id=", "", links["openreview"]))
    if rec.get("title"):
        keys.add("title:" + norm_title(rec["title"]))
    return keys


class PaperDB:
    def __init__(self, records: list[dict], topics: dict):
        self.records = records
        self.topics = topics
        self._index: dict[str, dict] = {}
        for rec in records:
            self._register(rec)

    # ---- persistence --------------------------------------------------------------------------
    @classmethod
    def load(cls, papers_path: Path = PAPERS_PATH, topics_path: Path = TOPICS_PATH) -> "PaperDB":
        records = yaml.safe_load(papers_path.read_text()) or [] if papers_path.exists() else []
        topics = yaml.safe_load(topics_path.read_text()) or {} if topics_path.exists() else {}
        return cls(records, topics)

    def save(self, papers_path: Path = PAPERS_PATH) -> None:
        self.records.sort(key=lambda r: (str(r.get("date") or ""), r["id"]), reverse=True)
        papers_path.write_text(
            "# Paper database of the GA-Research literature sweep. Edit by hand or with `python -m ga_lit`.\n"
            "# Field reference: literature/README.md\n"
            + yaml.safe_dump(self.records, sort_keys=False, allow_unicode=True, width=110)
        )

    # ---- lookup -------------------------------------------------------------------------------
    def _register(self, rec: dict) -> None:
        for k in keys_of(rec):
            self._index.setdefault(k, rec)

    def find(self, candidate: dict) -> dict | None:
        """Return the stored record matching any identifier or the title of ``candidate``."""
        for k in keys_of(candidate):
            if k in self._index:
                return self._index[k]
        return None

    def get(self, rec_id: str) -> dict | None:
        return self._index.get(rec_id)

    def add(self, rec: dict) -> None:
        rec.setdefault("added", date.today().isoformat())
        self.records.append(rec)
        self._register(rec)

    def merge_alias(self, existing: dict, candidate: dict) -> bool:
        """Attach the identifiers of ``candidate`` to ``existing``; return True if anything changed."""
        changed = False
        links = existing.setdefault("links", {})
        for name, url in (candidate.get("links") or {}).items():
            if url and not links.get(name):
                links[name] = url
                changed = True
        for src in candidate.get("found_by") or []:
            fb = existing.setdefault("found_by", [])
            if src not in fb:
                fb.append(src)
                changed = True
        if candidate["id"] != existing["id"] and candidate["id"] not in (existing.get("aliases") or []):
            existing.setdefault("aliases", []).append(candidate["id"])
            changed = True
        if changed:
            self._register(existing)
        return changed

    # ---- views --------------------------------------------------------------------------------
    def by_status(self, status: str) -> list[dict]:
        return [r for r in self.records if r.get("status") == status]

    def validate(self) -> list[str]:
        problems = []
        seen: dict[str, str] = {}
        for rec in self.records:
            rid = rec.get("id", "<missing id>")
            if rec.get("status") not in STATUSES:
                problems.append(f"{rid}: status must be one of {STATUSES}")
            if rec.get("status") == "accepted" and rec.get("relevance") not in RELEVANCES:
                problems.append(f"{rid}: accepted papers need relevance in {RELEVANCES}")
            if rec.get("venue_type") not in VENUE_TYPES:
                problems.append(f"{rid}: venue_type must be one of {VENUE_TYPES}")
            for t in rec.get("topics") or []:
                if t not in self.topics:
                    problems.append(f"{rid}: unknown topic '{t}'")
            if rec.get("status") == "accepted" and not rec.get("topics"):
                problems.append(f"{rid}: accepted papers need at least one topic")
            for k in keys_of(rec):
                if k in seen and seen[k] != rid:
                    problems.append(f"{rid}: shares key {k} with {seen[k]}")
                seen[k] = rid
        return problems


def load_state() -> dict:
    return yaml.safe_load(STATE_PATH.read_text()) or {} if STATE_PATH.exists() else {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(yaml.safe_dump(state, sort_keys=False))
