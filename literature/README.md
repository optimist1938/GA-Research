# Literature database: Clifford / geometric algebra deep learning

A small, scriptable knowledge base of papers that use Clifford algebra (also called geometric algebra, GA)
in machine learning, plus the pipeline that keeps it fresh. Everything lives in this folder.

| Path | What it is |
|---|---|
| `papers.yaml` | **The database.** One YAML record per paper (schema below). This is the only file you edit by hand. |
| `topics.yaml` | The topic taxonomy: id, display name and a one-line description of each shelf. |
| `site/index.html` | **The browser.** A self-contained page with full-text search, topic / relevance / venue / year filters and an inbox tab. Live copy: <https://chagrygoris.github.io/GA-Research/> (GitHub Pages, redeployed on every change to `main`); also works opened locally. Generated. |
| `catalog.md` | A Markdown catalogue of accepted papers grouped by year and topic, for reading on GitHub. Generated. |
| `ga_lit/` | The Python package: `config.py` (search terms and keyword vocabularies), `fetch.py` (arXiv, OpenReview, Crossref, INSPIRE-HEP), `triage.py` (scoring), `db.py` (load / dedupe / validate), `render.py`, `cli.py`. |
| `state.yaml` | When the last sweep ran and what it found. Generated. |
| `../.github/workflows/literature-update.yml` | The daily GitHub Actions job that runs the sweep, commits the result to `main` and redeploys the page. |

## Daily use

```bash
cd literature
pip install pyyaml                      # the only dependency
python -m ga_lit render                 # rebuild site/index.html and catalog.md from papers.yaml
python -m ga_lit inbox                  # what is waiting for a decision, best score first
python -m ga_lit accept arxiv:2611.01234 --relevance core --topics architecture,vision --summary "One line."
python -m ga_lit reject arxiv:2611.04321 --note "Clifford circuits, not geometric algebra"
python -m ga_lit update --since 2026-11-01   # run the sweep yourself for a custom window
python -m ga_lit stats                  # counts by status, relevance and topic
python -m ga_lit validate               # schema check and duplicate detection
```

The browser (`site/index.html`) has an **Inbox** tab. Each inbox card shows the keyword evidence and a
"copy accept command" / "copy reject command" button so that triage is paste-and-run.

## How the sweep works

1. **Fetch.** `fetch.py` queries four sources for papers first published inside a date window
   (default: from the last run minus 7 days of overlap until today):
   * arXiv API with ~35 keyword queries (`config.ARXIV_QUERIES`);
   * OpenReview full-text search (ICLR, ICML, NeurIPS, CVPR, TMLR and workshop submissions);
   * Crossref bibliographic search (journals, proceedings, SSRN / Research Square / Preprints.org);
   * INSPIRE-HEP: every new paper citing the Lorentz-equivariant geometric algebra transformer (L-GATr)
     line and the Clifford Group Equivariant Neural Network paper, plus title searches.
   Semantic Scholar, OpenAlex and dblp are not used because they rate-limit anonymous clients; adding one
   needs an API key stored as a repository secret and a fetcher in `fetch.py`.
2. **Dedupe.** `db.py` matches candidates to existing records by arXiv id (version stripped), DOI,
   OpenReview id or normalised title. Known papers only gain new links and `found_by` sources.
3. **Triage.** `triage.py` computes a *triage score*: the number of distinct GA terms matched in title and
   abstract, plus two if a machine-learning term matched, minus three if a name-collision term matched
   (quantum "Clifford gates", the Clifford torus, multi-vector retrieval, rotor machines, ...), minus one for
   an arXiv paper filed outside the cs / stat / eess / q-bio categories. A candidate enters the **inbox**
   with suggested topics when it has a machine-learning term and either scores at least 3 or has a GA term
   in its title; pure mathematics or physics papers about Clifford algebras stay out. Nothing is accepted
   automatically; rejected records are kept so they are never proposed again. The raw candidates of the last
   fetch are cached in `.cache/candidates.json`, so `python -m ga_lit update --from-cache` re-runs triage
   offline after a change to `config.py`.
4. **Render.** `render.py` regenerates the browser page and the Markdown catalogue.

## Automation

`.github/workflows/literature-update.yml` runs **every day at 05:23 UTC** (after arXiv's nightly
announcement) and on manual dispatch with optional `since` / `until` inputs. It runs `python -m ga_lit
update`, commits the updated database, page and catalogue straight to `main`, and redeploys the page to
GitHub Pages at <https://chagrygoris.github.io/GA-Research/>. New papers only ever enter the **inbox**, so
the automatic commit never changes what counts as accepted; the run summary in the Actions tab lists the
inbox as a checklist. `.github/workflows/pages.yml` additionally redeploys the page when a human pushes a
change to `literature/site/` on `main`. One-time repository setting for both: *Settings -> Pages -> Build
and deployment -> Source: GitHub Actions*.

Reviewing the inbox is the one manual step: open the **Inbox** tab of the live page, decide, and run the
copied `accept` / `reject` command (or edit the YAML), then `python -m ga_lit render` and commit. It can
be delegated to an assistant session with a prompt such as "triage the literature inbox: open each
candidate, decide core / adjacent / context or reject, run the commands and commit", which keeps a human in
the loop at commit review.

## Record schema (`papers.yaml`)

```yaml
- id: arxiv:2601.06793            # primary key: arxiv:<id> | doi:<doi> | openreview:<id> | url:<slug>
  title: CliffordNet: All You Need is Geometric Algebra
  authors: [Zhongping Ji]         # [] when anonymous (double-blind submission)
  date: 2026-01-11                # first public date (YYYY-MM-DD); journal issue dates may be in the future
  venue: arXiv                    # free text, e.g. "ICML 2026 (PMLR 306)"
  venue_type: arxiv               # arxiv | conference | journal | openreview | preprint-server | book | other
  links: {arxiv: ..., doi: ..., openreview: ..., code: ..., url: ..., inspire: ...}
  aliases: [arxiv:2608.24795]     # other ids of the same paper (re-postings)
  topics: [architecture, vision]  # ids from topics.yaml; the first one is the primary shelf
  status: accepted                # accepted | inbox | rejected
  relevance: core                 # core | adjacent | context (required when accepted)
  summary: One sentence saying what the paper does and why it matters here.
  notes: optional free text (for rejected items: why)
  triage: {score: 3, ga_terms: [...], ...}   # present on inbox items only, removed on accept/reject
  found_by: [arxiv:'all:"clifford algebra"', crossref:...]   # which queries surfaced it
  added: 2026-10-25
```

**Relevance** is the main navigation axis: *core* means geometric algebra is inside the model or method;
*adjacent* means GA-related machine learning or a GA method next to a learned component; *context* means the
paper cites or benchmarks against GA models without using one (kept for recall).
