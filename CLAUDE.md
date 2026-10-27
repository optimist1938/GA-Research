# Rules for AI agents working in this repository

This list will be updated. Read it fully before starting any task.

## 1. GPU resources

GPU resources in this project are managed with the Kaggle router in [`gpu_pool/`](gpu_pool/).
Two types of GPU are available in our pool:

| GPU | Use for |
|---|---|
| RTX cards | main experiments |
| 2xT4 | side quests |

* **Always calculate an approximate ETA** of a GPU run before launching it, and state it.
* **Any operation that takes more than 30 minutes in total must be approved by a human**
  before it is started. The pool does not support cancelling operations, so a launched run
  cannot be stopped.

## 2. Writing and mathematics

* **Never introduce a new entity without explaining what it is.** This applies to code
  (modules, classes, flags, config options), reports and documentation alike.
* **When writing mathematical formulas, define every symbol before it is used.**

## 3. Monitoring launched operations

After launching an operation, set up regular monitoring of it:

* **Every 15 minutes at the beginning**, to make sure the process actually started.
* **Then every 1 hour** for runs with a large ETA.

## 4. Literature database

Papers on Clifford / geometric-algebra machine learning are catalogued in [`literature/`](literature/)
(full guide: [`literature/README.md`](literature/README.md)).

* **`literature/papers.yaml` is the only source of truth.** `literature/site/index.html` (the searchable
  browser page), `literature/catalog.md` (the Markdown catalogue) and `literature/state.yaml` are
  generated; never edit them by hand. After any change to the YAML run `python -m ga_lit render`
  from `literature/` and commit the regenerated files together with it.
* **Use the CLI, not ad-hoc edits**, so records stay valid: `python -m ga_lit update` (sweep the
  sources for a date window), `inbox` (what awaits a decision), `accept ID --relevance core|adjacent|context
  --topics a,b --summary "..."`, `reject ID --note "..."`, `validate`, `stats`. Topic ids come from
  `literature/topics.yaml`; add a topic there before using it.
* **Before citing or surveying literature in a report**, search the database first (the browser page or
  `grep` on the YAML) and add any paper you use that is missing, with a one-line summary.
* **Triage rules.** Relevance *core* means geometric algebra is inside the model or method; *adjacent*
  means GA-related machine learning or a GA method next to a learned component; *context* means the
  paper cites or benchmarks against GA models without using one. Rejected items stay in the YAML with a
  `notes` reason so the sweep never proposes them again. Nothing is accepted automatically.
* **The daily sweep** (`.github/workflows/literature-update.yml`) commits new inbox candidates and the
  regenerated page straight to `main` and redeploys the live page. The Inbox tab of the page is the review
  queue; emptying it (accept or reject every item, then commit) is a task for a human or an agent asked to
  do it. Pull the latest `main` before editing `papers.yaml`, since the bot may have committed overnight.
