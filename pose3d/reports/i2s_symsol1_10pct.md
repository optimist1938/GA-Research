# Image2Sphere on SYMSOL I with 10 % of the data: the baseline the flows are compared with

Question: the Image2Sphere paper (Klee et al., ICLR 2023) reports only the log-likelihood on SYMSOL (Table 3: 2.98
with 10k training views per shape, 3.41 with the public 50k), never a rotation error, and nothing at our 4,500-view
setting. What does the published model reach under the protocol of this repo's `symsol1_10pct_*` runs?

Runs: `pepeschneyne/i2s-symsol1-10pct-sgd` and `pepeschneyne/i2s-symsol1-10pct` (Kaggle T4, one process), W&B
`i2s_symsol1_10pct_sgd` (19jdfmfk) and `i2s_symsol1_10pct_adamw` (7ki9qi3c). `--model i2s_real` (upstream's code, ResNet-50
ImageNet init), SYMSOL I, 4,500 of 45,000 training views per shape, one model for the five shapes, no augmentation, no EMA,
40 epochs, batch 64, argmax on the level-5 SO(3) HEALPix grid (2.36M rotations, 1.875 deg). Scored on the first 1,000 test
views per shape, like the flow runs: rotation error to the **nearest equivalent pose** and upstream's grid log-likelihood
(`log(softmax[nearest cell] N / pi^2)`, 8 equivalents per image). Notebook: `notebooks/i2s-symsol-runner.ipynb`.

| run | recipe | median error (deg) | mean of per-shape medians (deg) | Acc@15 | log-likelihood |
|---|---|---|---|---|---|
| `i2s_symsol1_10pct_adamw` | AdamW lr 1e-4 x sqrt(2), 5 warmup + cosine (the flow runs' schedule) | **3.07** | **3.21** | 1.000 | **3.48** |
| `i2s_symsol1_10pct_sgd` | the paper's: SGD + Nesterov 0.9, lr 1e-3, x0.1 at epochs 15 and 30, unclipped | 4.23 | 4.51 | 0.995 | 2.55 |
| `symsol1_10pct_gatr_spherecloud_harm6_i2s_adamw` (best flow run) | | 16.97 | 19.88 | 0.481 | 3.24 |
| `symsol1_10pct_gatr_spherecloud_small_i2s_adamw` | | 26.63 | 25.62 | 0.418 | 1.24 |

Per shape (AdamW run): median error tet 4.44, cube 4.40, icosa 4.75, cone 1.34, cyl 1.14 deg; log-likelihood tet 3.69,
cube 3.37, icosa 2.84, cone 4.06, cyl 3.46. SGD run: tet 6.64, cube 5.65, icosa 5.22, cone 2.61, cyl 2.45 deg;
log-likelihood tet 2.79, cube 2.42, icosa 1.71, cone 3.21, cyl 2.61.

## Reading

* Image2Sphere resolves all five symmetry groups from 4,500 views per shape: every polyhedron sits at 4 to 7 degrees,
  within a few grid spacings of the 1.875 deg argmax grid, and Acc@15 is 1.0. The flows at the same budget are at 17 to
  27 degrees median error because they only partly resolve the polyhedra (see `symsol_p1_stall.md`).
* The log-likelihoods are above the paper's 10k-view 2.98 (AdamW even above its 50k-view 3.41). Same histogram as
  upstream's `evaluate_ll`, but averaged over 8 equivalents per image on 1,000 views instead of one random equivalent on
  5,000, so the estimate is less noisy rather than the model better; and the Adam schedule simply trains the same network
  further than the paper's SGD one in 40 epochs (loss 5.5 vs 6.6 at the end).
* The flows' log-likelihood is not that far behind (3.24 vs 3.48 for the best run) while their median error is 6x worse:
  the flow puts mass near the right poses but its argmax-free sampling does not land on them, where I2S's grid argmax
  reads the mode off directly.

## A trap: gradient clipping

The trainer clipped the gradient norm to 1.0 unconditionally (a flow-recipe setting). Upstream I2S never clips, and
with SGD at lr 1e-3 the clip starved every update: after 40 epochs the loss was 10.14 against 10.51 for the uniform
distribution (33.7 deg class-mean median error, log-likelihood -1.90; kept in W&B as `i2s_symsol1_10pct_sgd_clipped`).
`--grad_clip` now exists (default 1.0, so the flow recipes are unchanged) and the I2S SGD recipe passes 0. Adam is nearly
insensitive to the clip (it rescales the step), which is why the AdamW run was fine with it on.

## Infrastructure notes

* Both DDP ranks on a Kaggle T4 x2 decode and hold the 47,500 images (7 GB each) and the host SIGKILLs one before the
  first epoch: the notebook runs `--no-ddp`. One T4 does 280 to 300 s per epoch, 3.5 h per run.
* `I2SReal` builds its evaluation grid lazily on the GPU in chunks, so the paper's level-5 grid (4.3 GB Wigner matrix)
  fits where upstream's whole-matrix CPU build did not; `grid_logits` feeds `engine/grid_model_likelihood.py`.
* `gpu_pool/rtx_monitor.py` keys its state by kernel ref: a notebook that runs two `run_name`s in sequence makes it
  re-push the first run's points into a fresh W&B run at every poll (16 duplicates, deleted). Replay such kernels
  per run_name by hand, or launch one run per kernel.
