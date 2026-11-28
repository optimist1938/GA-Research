# chile3d-pyr7-dit-geobias-grid14-s100-e180-l4x4

Kaggle kernel `stefanychaik/chile3d-pyr7-dit-geobias-grid14-s100-e180-l4x4`, launched 2026-11-28 07:45 UTC on 4xL4
(account12, competition `gemma-4-developer-agent`) with a human's approval. ETA ~2-2.5 h.

**Question.** Does the DiT vector field with an explicit geometric attention bias (`dit_geobias`) get past the ~13 deg
plateau on the full-3D SYMSOL-CHILE 7-gonal pyramid (`--symsol_set 14`, "pyr7"), when trained with the recipe under which
cosine-attention GATr learned the bottom-view 7-gon (0.40 deg / 0.46 deg at start scale 60 / 100)?

**Baseline.** `cheesewaffler/chile3d-pyr7-cosdir-grid14-s100-e180-seed0-t4x2` (GATr-cosdir, same recipe): 12.7 deg median
error, seed 1: 13.1 deg. The guessed-turn level is 90/7 = 12.9 deg.

**Differences from the baseline.** Vector field `dit_geobias` (width 21, 8 heads, 4 blocks, about 97k parameters, as in
`stefanychaik/symsol-dit-spherecloud-geobias-h8-step30-l4x4`) with a fixed bias gain s = 100 (GATr's start scale was 100
and learnable); 4xL4 instead of 2xT4 (same global batch 64); 8 data workers.

**Code.** Not on a branch of this repository. The kernel ships it as a tarball built from:
1. `chagrygoris/GA-Research`, branch `claude/symsol-symsol2-differences-2dukos` (SYMSOL-CHILE renderer, resumable training);
2. `pyr_patches.py`: `apply`, `apply_grid`, `apply_chile3d` copied from the GATr pyr7 kernel (`--dircloud_nside`,
   `--dircloud_grid`, `--symsol_set 14`); the GATr-only cosine-attention edits are not applied;
3. `geobias_graft.py`: `dit_geobias.py` from the SYMSOL-I geobias kernel's code, its gain read from
   `POSE3D_GEOBIAS_SCALE`, plus the wiring (`DiTBlock(attention=...)`, config literal, builder, read-out init).

`make_nb.py` writes the notebook from those pieces; it expects the geobias kernel's notebook (for the GATr wheels and the
dependency cell) and the code tarball in its working directory. The kernel tests itself before training: CPU tests
(`test_symsol_chile`, `test_resume`, `test_spherecloud`, `test_dircloud`), then a GPU build of the exact model (field type,
94k-98k field parameters, gain 100, 369 condition tokens, a finite 2-step prediction).

## Result (2026-11-28, kernel complete 09:5x UTC)

Final evaluation, 32 samples per image (geodesic medoid), 1,000 test images: **median 2.00 deg, acc@15 0.982,
acc@30 1.000**. Per-epoch single-sample median error: ~13 deg (the guessed-turn level) from epoch ~25 to ~100, then a
steady fall to 2.35 deg at epoch 180. The GATr-cosdir runs with the same recipe never left the plateau (final 12.74 and
13.11 deg). Single seed. Chart: `pyr7_geobias_vs_gatr.png` in the repository root (`plot_pyr7.py`).
