---
name: moon-cluster
description: Refresh or change the Moon Cluster page, the night-sky dial of our shared accelerator pool (accounts, competition-gated GPUs, quota). Use when asked to update, redraw, refresh or annotate the Moon Cluster / pool visualization, or after the pool changes (new tokens, removed accounts, new competitions).
---

# Moon Cluster

The Moon Cluster page is a published Artifact: **https://claude.ai/artifact/Y7cgZp5Kdbe1or38mdDPZd**.
Always republish to that URL so the link people already have keeps working.

`gpu_pool/moon_cluster.py` builds it in two steps, described in its docstring and in
`gpu_pool/README.md` ("Moon Cluster page"):

1. a **snapshot**: plain JSON describing the pool (no credentials), from a live probe or a file;
2. **render**: the page, from the snapshot plus the previous snapshot for the "changes" cards.

The published page embeds its own snapshot, so the current page is always the `--previous` input.

## Steps

1. **Credentials.** Use `$KAGGLE_TOKENS_DIR`, a directory the user names, or an uploaded zip that you
   unpack into a fresh folder in your scratchpad. Never copy tokens into the repo or the page.
   Folders without a `token` show up as *awaiting token*; folders tagged like
   `account7_Name[suspended]` show up as removed.
2. **Previous page.** Read the artifact (`Artifact` tool, `action: "read"`, the URL above) and pass
   the saved HTML file as `--previous`. Skip this only if the read fails; the page then shows no
   "new account / new route" cards.
3. **Apply the instruction.**
   - a card or announcement: `--note "KICKER|Title|Text"`;
   - an account that is gone: `--removed account7=deleted`;
   - a new name or subtitle: `--title`, `--subtitle`;
   - anything else (hide an account, rename a competition, fix a number): run with
     `--dump-snapshot snap.json` first, edit the JSON, then render with `--from-snapshot snap.json`.
     Hardware styles and competition display names live in `HARDWARE_STYLE` and
     `COMPETITION_NAMES` in `gpu_pool/moon_cluster.py`. Change them there when the change should last.
4. **Render.** About a minute for a live probe of 10 accounts (one CLI call per kernel):

   ```bash
   python gpu_pool/moon_cluster.py --tokens-dir "$TOKENS" --previous previous.html \
       --out "$SCRATCH/moon_cluster.html" [--note ...] [--removed ...]
   ```

5. **Publish** with the `Artifact` tool: `file_path` = the rendered file, `url` = the URL above.
   Read the artifact first in this conversation if you have not (a publish to an unread artifact
   is refused).
6. **Report** in a few lines: spare GPU-hours, live accounts, which premium GPUs are reachable and
   until when, and the change cards the page now shows.

## Do not

- Hand-edit the published HTML: the next render overwrites it. Change the snapshot, the flags, or
  the template (`gpu_pool/moon_cluster_template.html`) instead.
- Launch anything on the pool for this: the page only reads quota and kernel state.
