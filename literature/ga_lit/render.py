"""Render the database into the two human-facing views.

* ``literature/site/index.html`` - a single self-contained page (the template in ``site/template.html`` with
  the database embedded as JSON) that offers full-text search, topic / status / venue / year filters and
  an inbox tab for triage. It works from ``file://``, from GitHub Pages and as a shared artifact.
* ``literature/catalog.md`` - a Markdown catalogue of accepted papers grouped by topic, for people who
  prefer reading on GitHub.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from .db import LIT_DIR, PaperDB

SITE_DIR = LIT_DIR / "site"
TEMPLATE_PATH = SITE_DIR / "template.html"
INDEX_PATH = SITE_DIR / "index.html"
CATALOG_PATH = LIT_DIR / "catalog.md"


_SKELETON = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    '</head>\n<body>\n{fragment}\n</body>\n</html>\n'
)


def render_fragment(db: PaperDB) -> str:
    """The page body with the database embedded; this is what gets published as an artifact."""
    payload = {"generated": date.today().isoformat(), "topics": db.topics, "papers": db.records}
    data = json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")
    return TEMPLATE_PATH.read_text().replace("/*__DATA__*/null", data)


def render_site(db: PaperDB, out: Path = INDEX_PATH) -> Path:
    """Full HTML document for local viewing and GitHub Pages."""
    out.write_text(_SKELETON.format(fragment=render_fragment(db)))
    return out


def _fmt_links(links: dict) -> str:
    order = ["arxiv", "doi", "openreview", "pdf", "code", "inspire", "url"]
    parts = [f"[{k}]({links[k]})" for k in order if links.get(k)]
    parts += [f"[{k}]({v})" for k, v in links.items() if k not in order and v]
    return " · ".join(parts)


def _fmt_authors(authors: list[str], limit: int = 4) -> str:
    if not authors:
        return "anonymous"
    short = [a.split()[-1] if a else a for a in authors[:limit]]
    return ", ".join(short) + (" et al." if len(authors) > limit else "")


def render_catalog(db: PaperDB, out: Path = CATALOG_PATH) -> Path:
    accepted = [r for r in db.records if r.get("status") == "accepted"]
    years = sorted({str(r.get("date") or "")[:4] for r in accepted}, reverse=True)
    lines = [
        "# Clifford / geometric algebra deep learning: paper catalogue",
        "",
        f"Generated from `literature/papers.yaml` on {date.today().isoformat()} by `python -m ga_lit render`. "
        "Do not edit by hand; edit the YAML or use the CLI.",
        "",
        f"**{len(accepted)} accepted papers** ({sum(1 for r in accepted if r.get('relevance') == 'core')} core, "
        f"{sum(1 for r in accepted if r.get('relevance') == 'adjacent')} adjacent, "
        f"{sum(1 for r in accepted if r.get('relevance') == 'context')} context), "
        f"{len(db.by_status('inbox'))} waiting in the inbox. Browse interactively: `literature/site/index.html`.",
        "",
        "Relevance: **core** = geometric algebra inside the model or method; **adjacent** = GA-related machine "
        "learning or a GA method next to a learned component; **context** = cites or benchmarks against GA "
        "models without using one.",
        "",
    ]
    for year in years:
        lines += [f"## {year}", ""]
        by_topic: dict[str, list[dict]] = defaultdict(list)
        for r in accepted:
            if str(r.get("date") or "")[:4] != year:
                continue
            primary = (r.get("topics") or ["other"])[0]
            by_topic[primary].append(r)
        for topic_id in [t for t in db.topics if t in by_topic] + [t for t in by_topic if t not in db.topics]:
            meta = db.topics.get(topic_id, {"name": topic_id, "description": ""})
            lines += [f"### {meta['name']}", ""]
            if meta.get("description"):
                lines += [meta["description"], ""]
            for r in sorted(by_topic[topic_id], key=lambda x: str(x.get("date")), reverse=True):
                tag = "" if r.get("relevance") == "core" else f" *({r.get('relevance')})*"
                cross = [t for t in (r.get("topics") or [])[1:]]
                cross_s = f" — also: {', '.join(db.topics.get(t, {}).get('name', t) for t in cross)}" if cross else ""
                lines.append(
                    f"- **{r['title']}**{tag} — {_fmt_authors(r.get('authors') or [])}. "
                    f"*{r.get('venue', '')}, {r.get('date', '')}.* {_fmt_links(r.get('links') or {})}"
                )
                if r.get("summary"):
                    lines.append(f"  {r['summary']}{cross_s}")
            lines.append("")
    out.write_text("\n".join(lines))
    return out


def render_all(db: PaperDB) -> list[Path]:
    return [render_site(db), render_catalog(db)]
