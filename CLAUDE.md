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
