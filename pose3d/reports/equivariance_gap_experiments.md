# Experiments that would show an equivariant-vs-plain gap on Pascal3D+, if one exists

Branch `so2-cond-head` @ dc62901 (P1 / GATr side) and `dit-flow-baseline` @ 8ffa6db (DiT side), October 2026.
Design only; nothing was trained. Numbers marked **[computed]** come from the existing per-image
evaluation files in `/Users/optimist1938/rfeval/failviz/out/fail/*.npz` (scripts in `/tmp/eq_research_3/`,
each < 1 min CPU). The reference is Brehmer, Behrends, de Haan, Cohen, *Does equivariance matter at
scale?*, arXiv:2410.23179 ("the paper" below).

---

## 0. Summary

1. **Our current comparison cannot resolve the effect size it is looking for.** With 1,309 validation
   images the stratified image bootstrap gives the class-mean of per-class medians a 95 % CI half-width of
   **0.75–2.0 deg** per model (DiT: 1.96 deg, driven by diningtable with 7 images and sofa with 27), and
   paired GATr−DiT differences of **[−3.1, +0.9] deg** **[computed]**. Every reported "gap" (8.61 vs 8.85 vs
   8.95 vs 9.35) is inside that interval. On metrics that use all images, all six checkpoints are
   indistinguishable: mean log(1+err) paired CI ±0.04 (all within ±0.02 of each other), micro-median
   paired CI ±0.4 deg. The one consistent, near-significant signal is **acc@30: every GATr variant is
   +1.1 to +1.3 points above DiT (paired bootstrap P(GATr<DiT)=0.04–0.09)**, i.e. fewer gross failures
   (DiT 82 images > 90 deg, GATr 65–78). Re-evaluating the same 8.95 checkpoint with a different noise
   draw moved its class-mean by 0.17 deg (8.95 training-time vs 8.785 re-eval): evaluation noise alone is
   of the order of the claimed gaps.
2. **The paper's regime is not ours.** The paper matches *compute* (not parameters), uses a task whose
   symmetry is exact and fully exercised by the data, has the equivariant model see geometric inputs end
   to end, and finds (i) a roughly constant ~2x loss advantage across 10^16–10^19 FLOPs, (ii) a gap that
   *widens* as unique training data shrinks, and (iii) a non-equivariant model with symmetry augmentation
   that catches up in the small-data/many-epoch regime. Our pipeline has a 42.5 M non-equivariant
   ResNet-101 in front of a ~0.2–1.6 M head, data whose roll symmetry is barely exercised (jitter std 7.7
   deg; `reports/geometric_redesign.md`), and full augmentation on for both arms. Under those conditions
   the paper itself predicts no gap.
3. **The plan therefore moves to the regimes the paper says a gap lives in**, in order of cost: (A) head
   only, on frozen and D4-frame-averaged backbone features so the equivariant head is *exactly*
   equivariant end to end and a data-efficiency curve costs minutes per run; (B) the fine-tuned pipeline at
   low data and without augmentation; (C) evaluation on mirrored / rotated validation images, where
   exact equivariance guarantees invariance and the plain model has to have learned it. Everything is
   budgeted at ~50 RTX-hours plus a few T4-hours, with 3 seeds where it matters.
4. **Primary endpoints change.** Pre-register micro-median, acc@30 (failure rate) and mean log(1+err) with
   paired-by-image bootstrap CIs plus a seed-level CI (n = 3); report the class-mean of medians with its
   CI as the published secondary number, and never interpret a class-mean difference below ~1 deg.

---

## 1. Axes of scale in the paper, and what we can vary cheaply

| Axis | The paper | Here | Cost / status |
|---|---|---|---|
| Training compute | 10^16–10^19 FLOPs, model size N and tokens D traded off at each budget; compute-optimal fits L* ≈ 1.03 C^−0.268 (baseline) vs 0.14 C^−0.236 (equivariant); equivariant ≈ 2x lower loss throughout, gap does not shrink | `--n_epochs`, `--batch_size`, head size. Compute is dominated by the backbone (ResNet-101 fwd+bwd ≈ 23 GFLOPs/img vs < 0.1 GFLOPs for any head), so "compute" in the paper's sense is only variable with a frozen backbone | Cheap only in stage A (frozen features) |
| Unique training data | 2·10^6 – 2·10^11 tokens at fixed 10^18 FLOPs; gap widens with less data; baseline + symmetry augmentation catches up with many epochs; no gain from augmentation in the single-pass regime | Fraction of the 18,371 real images (`--real_fraction`, **new flag**, §5), with or without the 2.38 M synthetic renders (`--no-use_synth`), `--max_synth` to cap the synthetic pool | Cheap: a real-only epoch is 1/4 of the current one |
| Model size | One size parameter n scales depth/width/heads; equivariant model narrower (4n scalar + n multivector vs 64n channels); compute-optimal N* ∝ C^0.68 (equivariant) vs C^0.29 (baseline) | Head only: P1 `--dircloud_scalars/--gatr_mv_channels/--gatr_s_channels/--gatr_blocks`, DiT `--dit_dim/--dit_blocks`. Parameter ladder **[computed]**: P1 46 k (16/4/16/2), 116 k (32/8/32/2), **232 k (64/8/32/4 = the current P1-small = the defaults)**, 524 k (64/16/64/4), **1.07 M (64/16/96/6 = the current P1)**; DiT 126 k (dim 32), **249 k (dim 48 = DiT-small)**, 416 k (64), 886 k (96), **1.19 M (112)**, 2.37 M (160) | Cheap |
| Parameter vs compute matching | Nominal FLOPs; the equivariant model costs ~10x FLOPs per parameter (ξ ≈ 61 vs 6) | We match parameters; P1-small is 1.6x slower per epoch than DiT-small (175 vs 111 s). An equal-wall-clock DiT would be dim 64–96 | Cheap (one extra DiT run) |
| Augmentation | Baseline ± symmetry augmentation (2-D translations + rotations about gravity, resampled each epoch) | `--use_warp` = roll jitter (std 7.7 deg) + horizontal flip + bbox jitter, exactly the task's symmetry group; `--no-use_warp --no-use_synth` = zero augmentation (the flip is tied to `use_warp` in `datasets/packed.py::PackedReal`; synthetic renders always flip + jitter, so "no augmentation" requires `--no-use_synth` too) | Cheap, flags exist |
| Backbone frozen vs fine-tuned | n/a (no pretrained backbone) | `--freeze_encoder` exists (`Features.freeze_encoder`, passed as `freeze_backbone` to CliffordFlow and DiTFlow; BatchNorm pinned to eval by `encoders.freeze_encoder`) | Cheap; with cached features, very cheap |
| Exact end-to-end equivariance | Yes (GATr on raw geometry) | Only with frozen + D4 frame-averaged features (`--fa_d4`, **new**, §5): f̄(x) = ⅛ Σ_g g⁻¹·f(g·x) over the 8 rot90/flip transforms of the 7x7 map makes the per-cell scalars exactly D4-equivariant, so P1's Pin-equivariance becomes exact end to end; DiT-cells stays non-equivariant | Moderate code, cheap compute when cached |
| Input resolution | n/a | 224 is hard-wired (7x7 map; `DirectionCloudHead` asserts the grid) | Not cheap; skip |
| Early stopping / tuning | Early stopping everywhere, lr tuned per budget | EMA + cosine, no early stopping; per-epoch validation already logged, so "best epoch" is free to report | Free |
| Uncertainty | Bootstrap (10^4) only on scaling-law coefficients; seeds not stated | Per-image errors saved by `eval_fail.py` → paired image bootstrap is free; seeds via `--seed` | Free / seeds cost runs |

---

## 2. Ranked, budgeted plan (~50 RTX Pro 6000 hours + ~5 T4 hours)

Timing basis: DiT-small 111 s/epoch and P1-small 175 s/epoch at 73,484 samples/epoch (real + synth).
A real-only epoch (18,371 samples) is ≈ 28 s / 44 s plus per-epoch validation (1,309 images, one draw,
20 Euler steps; ≈ 10–15 s). Final 32-sample evaluation: ≈ 100 s (DiT) / 390 s (P1) on a T4.
Head-only steps on cached 2048x7x7 features (stage A) have no backbone cost: estimated 15–40 ms/step
for DiT-small/P1-small at batch 64 x 8 time samples **[reasoned from token counts; verify on the first run]**.

Hypothesis H: *an exactly equivariant head is more data-efficient than a parameter-matched plain
transformer head on inputs that transform under the task's symmetry, and the gap shrinks with data and
with symmetry augmentation.* Each run below states what confirms and what refutes H.

### Stage 0: no training (T4, ≈ 3–4 h total)

| # | Run | Config | Time | Confirms H | Refutes H |
|---|---|---|---|---|---|
| 0a | Paired bootstrap on the six existing per-image files | `/tmp/eq_research_3/boot.py` (CPU) | done, §3 | — | — |
| 0b | **Symmetry-consistency evaluation of the six existing checkpoints** (gatr895, gatr_s3, v4, v5b, p1, dit): for T ∈ {flip, roll ±10, ±30, 90 deg} report (i) error on T(val), (ii) equivariance error e_T = geodesic(pred(T x), T·pred(x)), (iii) the gain from T-TTA (medoid over the union of the 32 samples of x and the mapped samples of T x) | `eval_fail.py --transform ...` (**new option**, §5); same noise seed for all | 6 models x 6 transforms x 2–7 min ≈ 3 h T4 | P1's e_flip ≪ DiT's (e.g. < 5 deg vs > 15 deg) and P1's error on flipped val is unchanged while DiT's rises: the head's exactness survives the backbone (the backbone learned approximate flip equivariance from the augmentation) | All models degrade alike under T and P1's e_T is as large as DiT's: the backbone destroys the symmetry before the head sees it; a gap can then only appear in stage A |

For ±10/±30 deg use the loader itself (`desired_up = (cos φ, sin φ, 0)`, no bbox jitter, no flip): it
produces exactly the training-time distribution at small φ. Verified on this machine
(`/tmp/eq_research_3/rotlabel.py`): it gives `R' = Rz(−φ)·R` in the label convention, and at φ = 90 deg the
crop-to-crop map is the exact pixel rotation (x, y) → (y, 224 − x) with `R' = D·R`,
`D = [[0,1,0],[−1,0,0],[0,0,1]]`; at other φ the loader also rescales the crop (its focal fit uses
max |x|,|y|), so for a pure rotation use pixel rotation with a circular mask applied to *both* the φ = 0
and the rotated image. The flip is `R' = F R F`, `F = diag(−1,1,1)` (`reports/geometric_redesign.md` §1.3).

### Stage A: head-only on frozen, D4-averaged features (the paper's regime; ≈ 10–12 h)

Needs the feature cache and `--fa_d4` (§5). Backbone: ImageNet ResNet-101, frozen, no fine-tuning;
features precomputed once for train (18,371 x 8 forwards) and val. Absolute errors will be far worse
than 9 deg (frozen ImageNet features); that is fine, the question is the *gap* and its slope.

| # | Run | Config (both arms) | Time | Confirms H | Refutes H |
|---|---|---|---|---|---|
| A0 | Feature extraction + upload as a Kaggle dataset | `--freeze_encoder --fa_d4 --no-use_warp --no-use_synth --feature_cache_save_dir ...` | 0.3 h | — | — |
| A1 | **Data-efficiency curve.** `real_fraction ∈ {0.02, 0.05, 0.1, 0.25, 1.0}` (stratified per class, 367 / 918 / 1,837 / 4,593 / 18,371 images) x {P1-small 232 k, DiT-cells-small 249 k} x 3 seeds, fixed budget of 20 k optimizer steps (batch 64, `--n_epochs` = 20 k / steps-per-epoch, `--eval_every` so that ≤ 50 validations happen) | `--feature_cache_dir ... --batch_size 64 --no-ema`(or ema, same for both) `--seed {1,2,3}`; P1: `--cond_tokens dircloud --pose_tokens frame_pin --condition_head none`; DiT: `--model dit_flow --dit_tokens cells --dit_dim 48` | 30 runs x ≈ 8–12 min ≈ 5 h | Gap (DiT − P1) ≥ 2 deg micro-median at 2–5 % data, shrinking monotonically to ≈ 0 at 100 %; seed CI excludes 0 at the two smallest fractions | P1 ≤ DiT nowhere, or gap flat in data: equivariance buys no data efficiency even when exact |
| A1' | **Augmentation control.** DiT-cells with exact D4 augmentation of the cached features (rot90/flip of the 7x7 map + the matching label map, free because the features are D4-equivariant) at the same fractions, 3 seeds | `--feat_aug d4` (**new**, trivial with the cache) | 15 runs ≈ 2.5 h | DiT+aug closes most of the gap at 20 k steps (the paper's finding); P1 still ahead at the smallest fractions | DiT+aug beats P1 everywhere: the symmetry is learnable from augmentation and the hard-wired version is a handicap |
| A2 | **Model-size sweep** at 10 % and 100 % real: P1 {46 k, 116 k, 232 k, 524 k, 1.07 M} vs DiT-cells {126 k, 249 k, 416 k, 886 k, 1.19 M}, 2 seeds, 20 k steps | size flags from §1 | 40 runs x ≈ 6–15 min ≈ 5 h | The paper's pattern: P1 better at every size with a roughly constant ratio; at 10 % data the gap grows for small models | Curves cross or coincide |

### Stage B: fine-tuned pipeline, the regimes where a gap is most likely (≈ 20 h)

Both arms see the same 49-cell input (P1 vs **DiT-cells**, §5); the pooled 8.95 / DiT pair is kept only
as a reference. Real images only (`--no-use_synth`), so an epoch is ≈ 45–60 s.

| # | Run | Config | Time | Confirms H | Refutes H |
|---|---|---|---|---|---|
| B1 | **Low data, fine-tuned.** 10 % real (1,837 images), warp on, 300 epochs (≈ 8.6 k steps), `--eval_every 10`, 3 seeds each | `--real_fraction 0.1 --no-use_synth --use_warp --n_epochs 300 --warmup_epochs 15 --eval_every 10 --seed s` + arm flags | 6 runs x ≈ 0.5 h ≈ 3 h | P1 − DiT ≤ −1.5 deg micro-median (or ≥ +4 pts acc@30) with the 3-seed CI excluding 0 | |CI| contains 0 or DiT ahead |
| B2 | **Full real data, warp on, no synth**, 100 epochs, 3 seeds each | `--no-use_synth --seed s` + arm flags | 6 runs x ≈ 1.3–1.7 h ≈ 9 h | Gap ≤ −0.5 deg or ≥ +2 pts acc@30, seed CI excluding 0, and smaller than in B1 (data-dependence) | No gap: at 18 k images with augmentation the plain head has caught up, consistent with the paper |
| B3 | **No augmentation at all**, full real data, 2 seeds each | `--no-use_warp --no-use_synth --seed s` (uses the pre-built un-augmented tensors) | 4 runs x ≈ 1.3 h ≈ 5.5 h | Gap wider than in B2 (DiT loses the symmetry it learned from the flip/roll augmentation; P1's head keeps it) | Gap unchanged or P1 worse: the symmetry is not what distinguishes the heads |
| B4 | Stage-0b evaluation of every B checkpoint (flip, ±10, ±30, 90 deg; equivariance error) | T4 | ≈ 2 h T4 | B3-DiT degrades sharply under flip/roll, B3-P1 does not; B2 narrows that difference | Both degrade alike |

### Stage C: optional, with whatever remains (≈ 8 h)

| # | Run | Config | Time | Purpose |
|---|---|---|---|---|
| C1 | Equal-wall-clock DiT-cells (dim 64, 416 k) on the B2 recipe, 2 seeds | `--dit_dim 64` | 2 x 1.5 h | Removes the "parameter matching favours GATr" objection: the paper matches compute |
| C2 | P1 vs DiT-cells at 1 % real (184 images), warp on, 1,000 epochs with `--eval_every 50`, 3 seeds | `--real_fraction 0.01 --n_epochs 1000` | 6 x 0.4 h | The extreme of the data axis; also the regime where DiT+augmentation is expected to be closest |

Budget: stage 0 ≈ 3 h (T4) · A ≈ 12.5 h · B ≈ 17.5 h (+ 2 h T4) · C ≈ 8 h → ≈ 38 h RTX + 5 h T4,
leaving ≈ 12 h of the 50 h for re-runs of Kaggle sessions that never start (`memory: Kaggle silent
sessions`) and for the one-off feature-extraction and merge-verification runs.

Account split: A fits in one account's week (many short runs; keep each kernel ≤ 3 h by chaining runs in
one notebook); B1/B2 on the second account; B3/C the following week.

---

## 3. Statistics: what can and cannot be resolved

**What the existing data says [computed]** (stratified-by-class image bootstrap, B = 5,000, same val
order and noise for all six models):

| metric | typical per-model 95 % CI half-width | paired GATr(8.95) − DiT | paired v4 − DiT | paired P1 − DiT |
|---|---|---|---|---|
| class-mean of per-class medians | 0.75–1.96 deg | −0.15 [−3.13, +0.85] | −0.26 [−3.24, +0.55] | +0.29 [−2.78, +2.06] |
| 10-class trimmed class-mean (drop diningtable, sofa) | 0.6–0.8 deg | +0.23 [−0.54, +0.80] | −0.24 [−1.01, +0.31] | +0.24 [−0.61, +0.88] |
| micro-median | 0.4–0.55 deg | +0.24 [−0.19, +0.75] | +0.24 [−0.18, +0.64] | −0.05 [−0.43, +0.34] |
| acc@30 | 1.6–1.8 pts | +1.1 [−0.5, +2.7], P(<)=0.09 | +1.2 [−0.2, +2.8], P(<)=0.04 | +0.5 [−1.0, +2.1] |
| acc@15 | 2.2 pts | +0.2 [−1.7, +2.1] | +1.4 [−0.5, +3.3] | +1.2 [−0.7, +3.1] |
| mean log(1+err) | 0.05 | +0.001 [−0.038, +0.039] | −0.002 [−0.039, +0.034] | −0.009 [−0.047, +0.026] |

Two GATr seeds of the same config: class-mean 8.785 vs 8.823 (re-eval) but micro-median 7.92 vs 7.51
(paired CI [−0.14, +0.90]); diningtable's median alone ranges 4.2–8.8 deg across models. The published
number is the noisiest thing we measure.

**Protocol.**

1. *Endpoints, pre-registered.* Primary: micro-median error, acc@30 (= 1 − gross-failure rate) and mean
   log(1 + err), each with (a) a paired-by-image stratified bootstrap CI (free, from `eval_fail.py`
   outputs, pooled over seeds by concatenating the seed-wise per-image errors) and (b) a seed-level CI
   (mean ± t_{0.975, n−1}·sd/√n over n = 3 seeds of the per-run metric). Secondary, reported but not
   used for decisions: class-mean of per-class medians with its CI; the 10-class trimmed class-mean; per-class
   medians for boat, bicycle, motorbike, chair, tvmonitor (the classes where the probe shows the failures).
2. *Decision rule.* Declare a gap only if both the pooled paired-image CI and the seed CI exclude 0, and
   the point estimate exceeds the pre-registered minimum effect: 0.5 deg micro-median or 2 pts acc@30 at
   full data (B2/B3), 1.5 deg / 4 pts at ≤ 10 % data (A1/B1/C2). Below that, report "no detectable gap
   at this scale", which is a legitimate answer.
3. *Seeds.* 3 per arm wherever a decision is taken (A1, B1, B2), 2 for the sweeps (A2, B3, C1), and the
   same seed set for both arms so the comparison is paired by seed as well. With a seed-to-seed sd of
   ≈ 0.3 deg on micro-median (two GATr seeds: 0.4 deg apart), n = 3 resolves ≈ 1 deg at 95 %; 0.3 deg
   would need ≈ 15 seeds per arm and is out of budget, hence the shift to regimes with larger expected
   effects.
4. *Evaluation noise.* The 32-sample medoid with one noise seed moves the class-mean by ≈ 0.15 deg
   (8.95 → 8.785 on the same weights). Evaluate every checkpoint with `--samples 64` or two
   `--seed` values and average; always use the same eval seed for both arms.
5. *Curves, not points.* For A1/A2 report the gap as a function of data / size with seed error bars and
   fit log(err) = a + b·log(n_data) per arm; H predicts b_P1 < b_DiT in magnitude (a flatter curve for the
   equivariant head) and a positive intercept difference. A regression on all 30 runs is more powerful
   than any single pairwise test.
6. *Consistency metrics (stage 0b/B4)* need no ground truth and are therefore far less noisy than the
   error: e_T has a per-image CI of a fraction of a degree. Report its median and its 90th percentile.
7. *Multiple comparisons.* There are ~6 pre-registered decisions; report everything, do not cherry-pick a
   metric per run, and mention that at α = 0.05 one false positive among six is expected.

---

## 4. Zero-GPU and cheap first steps (do these before any RTX run)

1. **Done [computed]:** the paired bootstrap above (`/tmp/eq_research_3/boot.py`; re-run against any new
   `eval_fail.py` output by editing the `names` list). Conclusion to carry forward: the present
   "GATr does not beat DiT" result is itself not established either way at 0.3 deg; only acc@30 hints at
   fewer gross failures for the GATr family.
2. **T4, ≈ 3 h:** stage 0b, the flip / roll consistency evaluation of the six existing checkpoints. It
   decides whether the backbone passes any exact symmetry through (if P1's e_flip is small, the fine-tuned
   stage B is worth its 17 h; if not, stage A is the only place a clean gap can appear and B should be cut
   to B1 only).
3. **T4, ≈ 1 h:** re-evaluate the six checkpoints with a second noise seed (`--seed 1`) to measure the
   evaluation-noise floor per metric directly and to make the seed-level CIs honest.
4. **CPU, minutes:** verify the new `--transform` label maps on the loader alone (extend
   `/tmp/eq_research_3/rotlabel.py` to the flip and to pixel rotation of a real crop) before trusting any
   number from 0b; the sign convention for roll was checked only for the loader path.
5. **CPU, minutes:** after the merge in §5, run the existing unit tests
   (`pose3d/tests/test_dircloud.py`, `test_gatr_denoiser.py`, `test_so2_head.py`) plus the two new ones in §5 to
   confirm that `--fa_d4` makes the P1 model exactly D4-equivariant end to end on a random ResNet (float64,
   `rot90`/flip of the input image, D·R / F R F on the output) and that DiT-cells is not.

---

## 5. Code changes each run needs (all small; file paths absolute)

Nothing here exists yet on `so2-cond-head` unless marked *exists*.

| Change | Where | Size | Needed by |
|---|---|---|---|
| **Merge the DiT baseline into the P1 branch** (`git merge dit-flow-baseline` or cherry-pick 8ffa6db: `models/dit_flow.py`, `models/__init__.py::_dit_flow`, `config.py` `dit_*` flags, `tests/test_dit_flow.py`). The branches diverge only in `config.py`, `models/__init__.py` and the two model files; conflicts are additive | `/Users/optimist1938/code/GA-Research/pose3d/pose3d/` | merge | everything with both arms in one kernel |
| **`--dit_tokens pooled\|cells`**: a `DiTConditionHead` over the 49 cells of the 7x7 map (1x1 conv 2048 → `dit_dim`, learned 2-D position embedding, plus one axis token with its own flag so the token set matches P1's), instead of the 256 pooled chunks. At dim 48 the head is ≈ 98 k (conv) + 2 blocks ≈ 250 k, matching P1-small; same DiT vector field as now | `models/dit_flow.py` (`DiTConditionHead`, `DiTFlow.__init__/condition`), `config.py::FlowConfig.dit_tokens`, `models/__init__.py::_dit_flow` | ≈ 40 lines | A, B, C (the fair control; `reports/geometric_redesign.md` §3.1 experiment 3) |
| **`--real_fraction f` (+ `--subset_seed`)**: stratified-per-class subset of the real training images. Packed path: `PackedReal.__init__` draws `self.sel` from `self.idx['cls']` and indexes through it (`__len__`, `__getitem__`); `PackedPascal3D` is unchanged (synthetic draws still 3 x real per epoch). Cached / un-augmented path (`--no-use_warp --no-use_synth` → `_load_cached`): wrap the train `InMemoryDataset` in `torch.utils.data.Subset` with the same stratified draw, using the class labels `_attach_eval_classes` already reads | `config.py::DataConfig`, `datasets/packed.py`, `datasets/pascal.py::_train_dataset` | ≈ 30 lines | A1, A2, B1, C2 |
| **`--eval_every k`**: validate and log only when `(epoch+1) % k == 0` or on the last epoch (the per-epoch validation costs 10–15 s, which dominates at 1,837 images/epoch) | `config.py::TrainConfig`, `engine/trainer.py::train` | ≈ 8 lines | A, B1, C2 |
| **`--fa_d4`**: in `ImageToMultivectors.forward`, with `frozen_backbone`, run the backbone on the 8 D4 images (`torch.rot90(x, k, (2,3))`, `x.flip(3)`), undo the transform on the 7x7 map (`rot90(fmap, −k, (2,3))`, `flip(3)`), average. Only meaningful with `--freeze_encoder` (raise otherwise). Also the pooled path benefits (pooled features become D4-invariant) | `models/clifford_flow.py::ImageToMultivectors`, `config.py::Features` | ≈ 20 lines | A (and a 0b variant: D4-TTA of any frozen model) |
| **Feature cache**: `--feature_cache_save_dir` writes `(N, 2048, 7, 7)` fp16 + `rot` + `cls` for train and val after `--freeze_encoder [--fa_d4]` (3.7 GB train, fits RAM; upload as a Kaggle dataset); `--feature_cache_dir` loads it and the dataset yields `fmap` instead of `img`; `CliffordFlow.condition` / `DiTFlow.condition` skip the backbone when the input is a 2048-channel map. `--feat_aug d4`: random rot90/flip of the cached map with the label map `D^k·R` / `F R F` (exact for D4-averaged features) | new `datasets/features.py` (≈ 60 lines), `models/clifford_flow.py`, `models/dit_flow.py`, `engine/trainer.py::_compute_loss` (key `fmap`), `config.py::DataConfig` | ≈ 100 lines | A0–A2 |
| **`eval_fail.py --transform flip\|rot90:k\|roll:φ`** + output of `pred_T`, `e_T` and the TTA medoid; `roll:φ` through the loader (`Pascal3DReal` with `desired_up=(cos φ, sin φ, 0)`, `use_warp=False`, so no jitter/flip: needs a small subclass that overrides `desired_up`) or pixel rotation + circular mask | `/Users/optimist1938/rfeval/failviz/eval_fail.py` (outside the repo) | ≈ 50 lines | 0b, B4 |
| Unit tests: `test_fa_d4_makes_p1_exactly_equivariant` (float64, random ResNet weights, `rot90`/flip of the image → `D·R` / `F R F` of the integrated pose) and `test_dit_cells_is_not` (negative control) | `pose3d/tests/test_dircloud.py` | ≈ 40 lines | A |
| *Exists:* `--freeze_encoder`, `--no-use_warp`, `--no-use_synth`, `--max_synth`, `--n_epochs`, `--warmup_epochs`, `--seed`, `--batch_size`, all head-size flags, `--no-ema`, `--eval_samples`; P1 = `--cond_tokens dircloud --pose_tokens frame_pin --condition_head none` (+ `--gatr_mv_channels 16 --gatr_s_channels 96 --gatr_blocks 6` for the 1.07 M version); DiT-small = `--model dit_flow --dit_dim 48` | `config.py` | — | — |

Caveats to keep in the write-up: (i) with `--no-use_warp --no-use_synth` the training crops come from the
pre-built tensors (`pre_cache`), i.e. one fixed un-augmented pass, which is the intended "no augmentation";
(ii) `--fa_d4` is exact only for the 8 grid symmetries, not for the continuous roll, so stage A tests
D4-equivariance, the strongest exact statement available with a 7x7 grid; (iii) the fine-tuned stage B
models are never exactly equivariant (the backbone is not), so there H can only show up as a *learned*
advantage, which is exactly what the paper's data-axis figure is about; (iv) all GATr runs must keep
`join_reference="data"` (`reports/geometric_redesign.md` §0.5).

---

## 6. What each outcome would mean

* **Gap in A only.** Equivariance helps a head on equivariant inputs, but a fine-tuned ResNet learns the
  same thing from the augmentation. Honest conclusion: on Pascal3D+ with a pretrained CNN, GATr's symmetry is
  redundant; the paper's result is reproduced in the regime where its assumptions hold.
* **Gap in A and B1 but not B2/B3.** Equivariance is a low-data effect here, as in the paper's Figure 2;
  the project's headline run (full data + 2.38 M renders, augmentation on) is exactly the regime where the
  paper predicts no gap.
* **Gap in B3 but not B2.** The symmetry is the active ingredient and augmentation substitutes for it; the
  right claim is "equivariance replaces augmentation", not "equivariance beats transformers".
* **No gap anywhere, including 0b.** The backbone and the medoid dominate; the remaining GATr signal is the
  +1 pt acc@30 (fewer 180-degree flips), which should then be tested as its own hypothesis with the
  per-class failure counts rather than with the median.
