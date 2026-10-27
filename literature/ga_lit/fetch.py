"""Fetchers for the four sources that are reachable without an API key.

Each fetcher yields *candidates*: plain dicts with the keys

    id, title, authors, date (YYYY-MM-DD), venue, venue_type, abstract, links, found_by

restricted to papers first published inside ``[since, until]``. Network errors are retried a few times
and then logged; a failing source never aborts the whole run.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from typing import Iterable

from . import config
from .db import norm_arxiv

_ATOM = {"a": "http://www.w3.org/2005/Atom", "os": "http://a9.com/-/spec/opensearch/1.1/"}


def _get(url: str, retries: int = 4, accept: str = "application/json") -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": config.CONTACT, "Accept": accept})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - we want to keep going whatever the failure
            wait = 4 * (attempt + 1)
            print(f"  retry {attempt + 1}/{retries} after error: {exc} ({url[:90]})", file=sys.stderr)
            time.sleep(wait)
    return None


def _in_window(day: str | None, since: date, until: date) -> bool:
    if not day:
        return False
    try:
        d = date.fromisoformat(day[:10])
    except ValueError:
        return False
    return since <= d <= until


# --------------------------------------------------------------------------------------------------
# arXiv
# --------------------------------------------------------------------------------------------------

def fetch_arxiv(since: date, until: date, queries: Iterable[str] = config.ARXIV_QUERIES) -> list[dict]:
    window = f"submittedDate:[{since:%Y%m%d}0000 TO {until:%Y%m%d}2359]"
    out: dict[str, dict] = {}
    for q in queries:
        start = 0
        while True:
            url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                {"search_query": f"({q}) AND {window}", "start": start, "max_results": 200,
                 "sortBy": "submittedDate", "sortOrder": "descending"})
            data = _get(url, accept="application/atom+xml")
            if not data:
                break
            root = ET.fromstring(data)
            entries = root.findall("a:entry", _ATOM)
            for e in entries:
                aid = norm_arxiv(e.find("a:id", _ATOM).text)
                rec = out.setdefault(aid, {
                    "id": f"arxiv:{aid}",
                    "title": " ".join(e.find("a:title", _ATOM).text.split()),
                    "authors": [a.find("a:name", _ATOM).text for a in e.findall("a:author", _ATOM)],
                    "date": e.find("a:published", _ATOM).text[:10],
                    "venue": "arXiv",
                    "venue_type": "arxiv",
                    "abstract": " ".join(e.find("a:summary", _ATOM).text.split()),
                    "links": {"arxiv": f"https://arxiv.org/abs/{aid}"},
                    "categories": [c.get("term") for c in e.findall("a:category", _ATOM)],
                    "found_by": [],
                })
                rec["found_by"].append(f"arxiv:{q}")
            total = int(root.find("os:totalResults", _ATOM).text or 0)
            start += 200
            if start >= total or not entries:
                break
            time.sleep(config.SLEEP["arxiv"])
        print(f"  arxiv  {len(out):4d} cumulative  <- {q}", file=sys.stderr)
        time.sleep(config.SLEEP["arxiv"])
    return list(out.values())


# --------------------------------------------------------------------------------------------------
# OpenReview
# --------------------------------------------------------------------------------------------------

def fetch_openreview(since: date, until: date, terms: Iterable[str] = config.OPENREVIEW_TERMS) -> list[dict]:
    out: dict[str, dict] = {}
    for term in terms:
        for offset in range(0, 2000, 200):
            url = "https://api2.openreview.net/notes/search?" + urllib.parse.urlencode(
                {"term": term, "limit": 200, "offset": offset, "content": "all", "source": "forum"})
            data = _get(url)
            if not data:
                break
            notes = json.loads(data).get("notes", [])
            for n in notes:
                c = n.get("content", {})

                def val(k):
                    x = c.get(k)
                    return x.get("value") if isinstance(x, dict) else x

                ts = n.get("pdate") or n.get("cdate") or n.get("tcdate")
                day = datetime.fromtimestamp(ts / 1000, timezone.utc).date().isoformat() if ts else None
                if not _in_window(day, since, until):
                    continue
                inv = (n.get("invitations") or [""])[0]
                venue = val("venue") or inv.split("/-/")[0].replace("/", " ")
                authors = val("authors") or []
                if authors and isinstance(authors[0], dict):
                    authors = [a.get("fullname", "") for a in authors]
                rec = out.setdefault(n["id"], {
                    "id": f"openreview:{n['id']}",
                    "title": (val("title") or "").strip(),
                    "authors": authors,
                    "date": day,
                    "venue": venue,
                    "venue_type": "openreview",
                    "abstract": (val("abstract") or "").strip(),
                    "links": {"openreview": f"https://openreview.net/forum?id={n['id']}"},
                    "found_by": [],
                })
                rec["found_by"].append(f"openreview:{term}")
            if len(notes) < 200:
                break
            time.sleep(config.SLEEP["openreview"])
        print(f"  openreview {len(out):4d} cumulative  <- {term}", file=sys.stderr)
        time.sleep(config.SLEEP["openreview"])
    return list(out.values())


# --------------------------------------------------------------------------------------------------
# Crossref
# --------------------------------------------------------------------------------------------------

_CR_TYPES = {"journal-article": "journal", "proceedings-article": "conference", "book-chapter": "book",
             "posted-content": "preprint-server"}


def fetch_crossref(since: date, until: date, terms: Iterable[str] = config.CROSSREF_TERMS) -> list[dict]:
    out: dict[str, dict] = {}
    for term in terms:
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
            {"query.bibliographic": term, "rows": 100,
             "filter": f"from-pub-date:{since.isoformat()},until-pub-date:{until.isoformat()}",
             "select": "DOI,title,author,issued,container-title,abstract,URL,type", "mailto": "noreply@example.org"})
        data = _get(url)
        if not data:
            continue
        for w in json.loads(data)["message"]["items"]:
            parts = (w.get("issued") or {}).get("date-parts", [[None]])[0]
            day = "-".join(f"{p:02d}" if i else f"{p:04d}" for i, p in enumerate(parts) if p is not None)
            if len(parts) == 1:
                day = f"{parts[0]}-01-01"
            elif len(parts) == 2:
                day = f"{parts[0]}-{parts[1]:02d}-01"
            rec = out.setdefault(w["DOI"].lower(), {
                "id": f"doi:{w['DOI'].lower()}",
                "title": re.sub(r"\s+", " ", (w.get("title") or [""])[0]),
                "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in (w.get("author") or [])],
                "date": day,
                "venue": (w.get("container-title") or [""])[0],
                "venue_type": _CR_TYPES.get(w.get("type"), "other"),
                "abstract": re.sub(r"<[^>]+>", "", w.get("abstract") or "").strip(),
                "links": {"doi": f"https://doi.org/{w['DOI']}"},
                "found_by": [],
            })
            rec["found_by"].append(f"crossref:{term}")
        print(f"  crossref {len(out):4d} cumulative  <- {term}", file=sys.stderr)
        time.sleep(config.SLEEP["crossref"])
    return list(out.values())


# --------------------------------------------------------------------------------------------------
# INSPIRE-HEP (citation graph of the GA transformer papers + title search)
# --------------------------------------------------------------------------------------------------

def _inspire_query(q: str, since: date, until: date, found_by: str, out: dict[str, dict]) -> None:
    page = 1
    while True:
        url = "https://inspirehep.net/api/literature?" + urllib.parse.urlencode(
            {"q": q, "size": 250, "page": page, "sort": "mostrecent",
             "fields": "titles,arxiv_eprints,earliest_date,authors.full_name,abstracts,dois"})
        data = _get(url)
        if not data:
            return
        hits = json.loads(data).get("hits", {}).get("hits", [])
        oldest = "9999"
        for h in hits:
            m = h["metadata"]
            day = m.get("earliest_date", "")
            oldest = min(oldest, day or "9999")
            if not _in_window(day if len(day) == 10 else (day + "-01-01")[:10], since, until):
                continue
            eprints = [e["value"] for e in m.get("arxiv_eprints", [])]
            dois = [d["value"] for d in m.get("dois", [])]
            key = f"arxiv:{eprints[0]}" if eprints else (f"doi:{dois[0].lower()}" if dois else f"inspire:{h['id']}")
            links = {"inspire": f"https://inspirehep.net/literature/{h['id']}"}
            if eprints:
                links["arxiv"] = f"https://arxiv.org/abs/{eprints[0]}"
            if dois:
                links["doi"] = f"https://doi.org/{dois[0]}"
            rec = out.setdefault(key, {
                "id": key,
                "title": m["titles"][0]["title"],
                "authors": [a["full_name"] for a in m.get("authors", [])],
                "date": (day + "-01-01")[:10],
                "venue": "arXiv" if eprints else "INSPIRE record",
                "venue_type": "arxiv" if eprints else "other",
                "abstract": (m.get("abstracts") or [{}])[0].get("value", ""),
                "links": links,
                "found_by": [],
            })
            rec["found_by"].append(found_by)
        if len(hits) < 250 or oldest < since.isoformat():
            return
        page += 1
        time.sleep(config.SLEEP["inspire"])


def fetch_inspire(since: date, until: date) -> list[dict]:
    out: dict[str, dict] = {}
    for name, recid in config.INSPIRE_SEEDS.items():
        _inspire_query(f"refersto:recid:{recid}", since, until, f"inspire:cites {name}", out)
        print(f"  inspire {len(out):4d} cumulative  <- cites {name}", file=sys.stderr)
        time.sleep(config.SLEEP["inspire"])
    for q in config.INSPIRE_TITLE_QUERIES:
        _inspire_query(q, since, until, f"inspire:{q}", out)
        print(f"  inspire {len(out):4d} cumulative  <- {q}", file=sys.stderr)
        time.sleep(config.SLEEP["inspire"])
    return list(out.values())


SOURCES = {"arxiv": fetch_arxiv, "openreview": fetch_openreview, "crossref": fetch_crossref, "inspire": fetch_inspire}


def fetch_all(since: date, until: date, sources: Iterable[str] = SOURCES) -> list[dict]:
    candidates: list[dict] = []
    for name in sources:
        print(f"[{name}] {since} .. {until}", file=sys.stderr)
        try:
            candidates.extend(SOURCES[name](since, until))
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAILED: {exc}", file=sys.stderr)
    return candidates
