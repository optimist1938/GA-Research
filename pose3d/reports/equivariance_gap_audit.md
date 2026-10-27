# Why the equivariant GATr does not beat the DiT on Pascal3D+: a code and pipeline audit

Branch `so2-cond-head` (HEAD dc62901), DiT on `dit-flow-baseline` (8ffa6db). October 2026.
Scope: where in *our* pipeline the symmetry is broken, never exercised, or handed to the baseline for free.
Nothing was trained. Every number marked **[verified]** comes from the scripts in `/tmp/eq_research_2/`
(`warp_conventions.py`, `params.py`, `equiv_e2e.py`, `equiv_roll.py`; each < 30 s on CPU) run on the two
checkpoints (`clifford_flow_gatr_warp_synth_ema_b64.pth` = 8.95 baseline, `clifford_flow_gatr_p1_dircloud.pth` = P1)
and on 6-12 Pascal3D+ validation thumbnails (`rfeval/failviz/out/fail/thumbs.npz`, 128 px upsampled to 224; the
dataset itself is not local). **[reasoned]** = from the code or the literature only.

Results being explained (class-mean of per-class median error, deg): v4 8.61 (~1.6M head params), DiT 8.85 (1.19M),
8.95 baseline (1.13M; seeds 8.93-9.12), P1 9.35 (1.07M), DiT-small 9.09 (0.249M), P1-small 9.41 (0.232M).

---

## 0. The short version

Brehmer et al. (arXiv 2410.23179) see equivariant models win at small scale because (a) the inputs are geometric
objects that really transform, (b) the data is spread over the group, and (c) the whole network is equivariant.
None of the three holds here, and two of them fail inside our own code:

| | what the paper assumes | what our pipeline does | evidence |
|---|---|---|---|
| input representation | points / vectors that transform under the group | 8.95 baseline, v4, DiT: a globally pooled 2048-vector cut into 256 "multivectors" that transform under nothing; P1: directions on a 7x7 grid that transform only under 90 deg turns and the flip | §1, §2 **[verified]** |
| data spans the group | yes (E(3)) | roll jitter std 7.7 deg, max 38 deg, never 90 deg; flip 50 %; test crops upright | §2 **[verified]** |
| equivariant end to end | yes | 97.4 % of the parameters (ResNet-101) are not equivariant; a 90 deg turn of the image moves the predicted pose by ~120 deg for *both* models | §1 **[verified]** |
| the non-equivariant control cannot learn the symmetry cheaply | trained without augmentation in the paper's main setting | the warp/flip augmentation teaches the baseline the flip to 2.2 deg and 10 deg rolls to 2.0 deg, as well as P1 (2.6 / 2.4 deg) | §1, §2 **[verified]** |

On top of that the "equivariant" entries in the results table are not equivariant to anything the data does
(the 8.95 baseline and v4 use a rotor token that GATr *conjugates* while the data *left-multiplies*; the pooled
tokens never transform at all), the one honest model (P1) was compared against a DiT that sees a different input
(pooled 256x8 vs 49 cells x 64 scalars) and gets 14x more conditioning bandwidth than the baseline, and the DiT
has better time conditioning (sinusoidal + adaLN-Zero vs a raw scalar). All differences are within 1-2 seed spreads.

---

## 1. Findings, ranked by likely impact

### F1. The exact symmetry of the equivariant head is one the data never performs, and the symmetry the data does perform is only approximate for every model **[verified]**

P1's head (`dircloud_head.py` + `GATrCloudField`) is exactly equivariant to the 90 deg turn and the left-right
flip of the 7x7 map: with the *feature map* transformed and the pose mapped (R -> R_z(-90) R, resp. F R F), the
body velocity changes by 3e-7 (rot90) and 2.5e-7 (flip) relative (`equiv_e2e.py`, trained weights, fp32). That is
the symmetry group C4 x Z2 on the grid, plus whatever the ResNet gives. What the data realises
(`packed.py::_warp_to_224`, `desired_up = [3,0,0] + N(0, 0.4)`):

* roll angle std **7.71 deg**, 99 % within 20 deg, max **38 deg** in 200k draws, **P(|roll| > 45 deg) = 0**; a 90 deg
  roll never occurs in training and the test crops (`desired_up = [1,0,0]`) keep the photo's roll (upright).
* the horizontal flip with probability 1/2, label `F R F` to machine precision (residual 0.0e+00); no vertical flip.
* a 10 deg roll of the crop by the loader is, to resampling noise, `R -> R_z(-10) R` (residual 4.8e-5) and the image
  turned CCW on screen; `torch.rot90(x, 1, dims=(2,3))` <-> `R_z(-90) R` as `test_so2_head.py` assumes (image diff 0.0017).

So the only part of P1's exact group that the data uses is the flip. A 10 deg roll (what the augmentation draws) is
*not* in C4: on the 7x7 grid the cells do not permute, so the head is only approximately equivariant to it, exactly
like the DiT. Pose-level check (12 val thumbnails, 8-sample medoid, noise matched by `r0 -> G r0`, error between
`G pred(x)` and `pred(G x)`; "no reaction" would score the roll angle, perfect equivariance the resampling floor of
~0.8 deg):

| roll of the image | 5 deg | 10 deg | 20 deg | 45 deg (outside the augmentation) | 90 deg |
|---|---|---|---|---|---|
| 8.95 baseline (no image symmetry at all) | 2.10 | 1.99 | 3.62 | 8.25 (mean 19.7) | 99 (mean 115) |
| P1 (C4 x flip exact head) | 1.77 | 2.35 | 1.60 | 7.11 (mean 28.6) | 121 (mean 129) |

The augmentation gives the baseline the same roll behaviour as the exact head; at 90 deg, where P1's head is exact,
both models are wrecked because the ResNet is not (F2). **The equivariance that was built is never exercised and the
equivariance that is exercised was never built.**

*Fix.* Either make the head exact for the transformations the data has (continuous roll: resample the map onto a
polar / HEALPix hemisphere grid as the design report's P1-b, or canonicalise the roll with a learned C4-steerable
dipole; Kaba et al.) or make the data exercise the symmetry the head has (train with `rot90` augmentation and the
label `R_z(-90 k) R`: free, exact, and it would be the first regime where P1 could win against a DiT that then has
to learn four orientations). The cheapest diagnostic is D4 test-time augmentation on the existing checkpoints
(8 backbone passes, pool the mapped-back samples with the medoid): if neither model gains, exactness has no value
on this task at all.

### F2. 97.4 % of the model is a non-equivariant ResNet-101 and it destroys C4 completely; fine-tuning under the two designs pushes it in opposite directions **[verified]**

`params.py`: 8.95 baseline 43,747,104 params, of which the backbone is 42,605,510 (97.39 %); P1 43,677,536, backbone
97.55 %. Per-cell equivariance of the 2048x7x7 map (12 val thumbnails, cells permuted to undo the transform, cosine
between original and transformed features):

| backbone | flip: per-cell cos | flip: pooled cos | rot90: per-cell cos | rot90: pooled cos |
|---|---|---|---|---|
| ImageNet ResNet-101, not fine-tuned | 0.615 | 0.973 | 0.417 | 0.814 |
| fine-tuned inside the 8.95 baseline (pooled tokens) | **0.349** | 0.613 | 0.126 | 0.542 |
| fine-tuned inside P1 (direction cloud) | **0.746** | 0.963 | 0.130 | 0.698 |

Two things follow. (i) Under a 90 deg turn the fine-tuned features of a cell are uncorrelated with those of the
cell it came from (cos 0.13): the exact C4 head sits on inputs that do not transform, so end to end the velocity
error under rot90 is 1.20 relative for P1 (the "different image" control is 0.88) and 1.47 for the baseline. (ii) The
flip is different: P1's backbone became *more* flip-equivariant than ImageNet (0.75 vs 0.62) because its head can
only use the features through a flip-equivariant map, whereas the baseline's backbone became *less* (0.35), because
with global pooling the only way to carry orientation is through non-equivariant channel content (F3). End to end
under the flip, P1's velocity error is 0.068 relative (baseline 0.296) and the medoid pose error 2.6 deg (baseline
2.2 deg median but one of six images flips to a 169 deg mode). So the equivariant head does propagate its structure
into the backbone, but only for the flip, and the baseline matches it at the pose level anyway (F1).

*Would freezing the backbone help?* No: a frozen ImageNet ResNet is less flip-equivariant (0.62) than the one P1
trained (0.75) and not C4-equivariant at all (0.42); the exact head's assumption ("per-cell scalars are invariant")
would be more wrong, and the DiT, which can undo any fixed feature transformation, would lose less. Freezing
also removes the 42.6M that both models mostly learn with (pretrained-vs-scratch is 10 deg vs 57 deg per `config.py`).
*An equivariant backbone* is the only way to make P1's exactness real end to end; no E(2)-steerable ResNet-101 with
ImageNet weights exists, so the realistic versions are frame averaging over D4 (8x backbone cost, exact, no training)
or a canonicaliser, both of which can be tried on the existing checkpoints first.

### F3. The model called "equivariant" in the 8.95-vs-8.85 comparison is not equivariant to anything in the data; it is a differently parametrised black box with an input bottleneck **[verified + reasoned]**

* `ImageToMultivectors(conv_adapter=False)` (`clifford_flow.py`): `AdaptiveAvgPool2d(1)` then `view(B, 256, 8)`. The
  256 "multivectors" are eight consecutive ResNet channels; nothing makes them rotate. Rotating the feature map by
  90 deg leaves the condition tokens unchanged to 7.6e-6 (`equiv_roll.py`): the conditioning is exactly
  *invariant* to turning the image, so the model can only know the image orientation through the backbone's
  non-equivariance. The DiT uses the same adapter and has the same property.
* The Clifford condition head and the GATr act O(3)-equivariantly on this fake representation. That constrains the
  function class on orbits the data never visits: no help, only a harder parametrisation. The published ablation
  `mlp_heads` (43.7 deg, `config.py`) that made the GA layers look essential was a broken MLP baseline; the DiT
  (8.85 deg) is what a competent non-equivariant head does with the same inputs.
* The rotor token is conjugated by GATr (`R -> G R G~`) while a camera rotation left-multiplies (`R -> G R`);
  `test_rotor_token_breaks_that_invariance` is the repo's own negative control. This also applies to **v4**
  (`cond_tokens=so2`, `pose_tokens=rotor`), the best run at 8.61: the so2 tokens rotate with the image but the
  pose token does not transform compatibly, so v4 is not equivariant to the data symmetry either. The best
  "equivariant" number in the table comes from a model whose symmetry is inconsistent, with ~40 % more head
  parameters than the DiT.
* Bandwidth: the baseline's vector field sees `n_cond_mv = 64` tokens x 8 components = **512 numbers** per image;
  the DiT sees 64 tokens x `dit_dim = 112` = **7168 numbers** (DiT-small: 3072); P1 sees 50 x 73 = 3650 (of which
  3200 are plain scalars). Token *count* was matched, token *width* was not. The baseline's 1.04M condition head
  compresses the 2048-d image vector into 512 numbers before the vector field; the DiT's 0.2M condition head
  keeps 7168. That the baseline reaches 8.95 through a 14x narrower bottleneck says the bottleneck is not where
  the error is, but it is a parameter-matched comparison in name only.

*Fix.* Drop the pooled-token models from any "equivariant vs not" claim. The honest comparison is P1 against a
DiT over the same 49 per-cell tokens with 2-D positions (the design report's experiment 3), at matched token
width, matched epoch time and 3 seeds each. Until that is run, the table does not test the paper's claim.

### F4. Capacity inside the equivariant head is mostly in its invariant scalar path; the geometric part is a thin side channel **[verified]**

`params.py`, GATr internals: in P1's 940k vector field, 63.6 % of the parameters are scalar->scalar, 14.8 % mv->mv
(the only Pin-equivariant mixing), 11.5 % scalar->mv, 10.0 % mv->scalar; `gatr_mv_channels = 16` against
`gatr_s_channels = 128` and 64 scalar channels per token from the 1x1 conv. The 8.95 baseline's 98k vector field is
37.8 % scalar->scalar, 35.6 % mv->mv. In both, the multivector channel is a small fraction and all the image
content enters as invariant scalars; the geometry the mv channel carries in P1 is the fixed direction `s_j` of each
cell (a positional encoding) and the three pose axes. The "equivariant transformer" is a scalar transformer with a
16-channel geometric sidecar. That is a sensible design for this task, but it means parameter-matching to the DiT
mostly matches scalar MLP capacity, not geometric capacity, and the paper's regime (geometry-heavy inputs) is not
reproduced.

*Fix.* Report the split, and sweep `gatr_mv_channels` at fixed total: if 4 and 32 mv channels give the same
error, the geometric path is inert and the claim is moot. Add real geometric content (the design report's P2
depth-lifted points) before expecting equivariance to pay.

### F5. P1 hands the pose to the head as a fictional wide-angle geometry and gets no condition head; the DiT gets sinusoidal time and adaLN-Zero **[reasoned, with numbers from the code]**

* `dircloud_head.py` places cell directions on a virtual pinhole with 45 deg at the edge of the map, while the
  true bearing rays of a 224 px Pascal crop span < 5 deg (design report §1.2). The inner products `<R e_i, s_j>` the
  attention keys on have no physical meaning; they are a learned positional encoding with a Pin-equivariant
  parametrisation. Nothing in the data (object shape, depth) enters as a multivector. GATr's E(3) part (translations,
  the join) acts on nothing because no token has `e0` content; `join_reference="data"` is correct and harmless.
* P1 has `condition_head = none`: a single `Conv2d(2048, 64, 1)` + LayerNorm (131k params) is all the per-image
  processing; everything else runs inside the vector field 8x per image in training (`n_time_samples = 8`) and
  32 x 20 = 640x per image at evaluation. The DiT runs a 198k condition head once and a 991k vector field per
  step; the baseline runs a 1.04M condition head once and a 98k vector field per step. Three different
  allocations, all called "~1.1M".
* Time enters GATr as one raw scalar channel `t in [0, 1]` (`GATrVectorField.forward`, `GATrCloudField.forward`);
  the DiT embeds it with 256 sinusoidal frequencies, an MLP and adaLN-Zero modulation of every block, and also
  feeds the mean condition token through adaLN. This is a known-good conditioning mechanism that has nothing to
  do with equivariance and the GATr models do not have it.
* Cost: epoch time (one RTX Pro 6000) 215 s baseline, 230 s P1, 175 s P1-small, 117 s DiT, 112 s DiT-small
  (`doc_runs.json`, Kaggle logs). The DiT is ~2x cheaper per epoch at equal parameter count; a compute-matched
  comparison would favour it further.

*Fix.* Give the GATr models Fourier features of `t` in the scalar channels (invariant, so free), a per-image
condition stage of comparable size (P1 with the 1x1 conv widened or one GATr block run once over the cloud), and
report parameters *and* epoch time. The DiT control for P1 must see the same 49-cell tokens.

### F6. Everything else checked is consistent; the remaining asymmetries are shared or small **[verified / reasoned]**

* Label conventions: `e1` = columns right, `e2` = rows down, `e3` = optical axis match the loader (CCW turn on
  screen <-> `R_z(-90) R`; flip <-> `F R F` with `F = diag(-1, 1, 1)`), so the head, the tests and the data agree.
  The flip relabelling assumes bilaterally symmetric objects; the asymmetric instances (steering wheels, text) are
  label noise for every model.
* Crop centring: the warp is an exact virtual camera rotation; bbox jitter (+-10 %) is a translation of about
  0.7 cells, not a symmetry of any model; both learn it.
* `principal_point = (px, px)` (inherited from Mohlin et al.): a per-image tilt of the label frame of up to ~6 deg,
  identical in train and test and for every model; a floor on the metric, not a model difference.
* Medoid: geodesic distances are bi-invariant, so the medoid commutes with the group action (checked implicitly:
  the P1 flip test's medoid error, 2.6 deg, is within 2 deg of the per-sample error). EMA: averaging the weights of an
  equivariant architecture stays equivariant; `use_buffers=True` only averages ResNet BatchNorm statistics. Per-epoch
  metric: one fixed-noise draw for everyone; final metric 32-sample medoid for everyone.
* Normalisation: LayerNorm in `DirectionCloudHead` is per token (permutation-safe); GATr's EquiLayerNorm is
  invariant; the so2 head's RMS gain is invariant. BatchNorm lives only in the ResNet, in eval mode at test time.
* Precision: `GradScaler` is constructed but no `autocast` context exists in `engine/trainer.py`, so training is
  fp32 for every model (the scaler is a no-op); no fp16 damage to the geometric product.
* Optimiser: AdamW (default weight decay 0.01 on every parameter, EquiLinear weights included), lr 1e-4 x sqrt(2),
  5 warm-up epochs, cosine to 5 %, grad-clip 1.0 over all 43M parameters, EMA 0.999: the recipe was fixed on the
  baseline GATr run and copied unchanged to the DiT and P1 (`--batch_size 64 --lr_scaling sqrt`). Nothing was
  tuned per architecture; the DiT at least inherits the DiT paper's defaults (adaLN-Zero, zero-init output), the
  GATr models inherit a zero-init `linear_out` that stalled the frame-token runs twice (`gatr_denoiser.py` comments).
* Statistics: one seed for v4, DiT, P1 and the small models; the baseline's two seeds differ by 0.2 deg
  (8.93 / 9.12; `raw_cmre` of the last iterate differs from the EMA number by up to 0.19). Every gap in the table
  except P1-vs-DiT (0.5 deg) is within that spread, and P1-vs-DiT is confounded by F3/F5.

---

## 2. Answers to the five questions

**Q1. How equivariant is each model end to end?** `equiv_e2e.py` / `equiv_roll.py`, trained checkpoints, CPU.
Head-only (feature map transformed): P1 exact to 3e-7 for rot90 and flip; baseline 1.3 / 0.64 relative (not
designed to be). End to end (image transformed): rot90 fails for both (velocity error 1.20 P1 / 1.47 baseline, pose
error ~120 / ~100 deg); flip holds approximately for both (velocity 0.068 / 0.296; medoid pose error 2.6 / 2.2 deg,
one baseline image jumps to a mirrored mode); rolls of 5-20 deg are handled to ~2 deg by both, 45 deg to ~7-8 deg
median by both. The ResNet is the whole story: per-cell cosine 0.13 under rot90 for either fine-tuned backbone,
0.75 (P1) vs 0.35 (baseline) under the flip.

**Q2. Does the warp give the DiT the symmetry for free?** Yes for everything the data has: rolls with std 7.7 deg
(max 38 deg, never 90 deg) and the horizontal flip at p = 1/2, both relabelled exactly. The baseline, which has no
image symmetry in its architecture, is as roll- and flip-consistent as P1 at the pose level. Exact equivariance is
redundant on this data; the one direction where it would not be (90 deg turns, vertical flips) is never sampled
and is where the ResNet breaks everything.

**Q3. Silent symmetry breaks or mismatches?** None found in the head/loader conventions (axes, flip relabelling,
rot90 sign, medoid, EMA, norms, join reference, time/flag channels all consistent). The genuine mismatches are
architectural: (a) pooled tokens are exactly invariant to turning the map, so the baseline, v4's condition path
and the DiT can only read orientation from ResNet non-equivariance; (b) the rotor token (baseline, v4) transforms by
conjugation while the label transforms by left multiplication; (c) P1's C4 exactness vs the data's continuous
small rolls. Shared, model-independent noise: `principal_point = (px, px)`, flip labels on asymmetric instances.

**Q4. Where does capacity sit?** 97.4-97.5 % in the ResNet-101 for every model. Of the head budget, P1 puts
88 % in a vector field whose parameters are 64 % scalar->scalar and 15 % mv->mv (16 mv channels vs 128 + 64
scalar channels); the baseline puts 91 % in a Clifford condition head acting on a fake representation. Freezing
the backbone would make the exact head's assumption less true (ImageNet features are less flip-equivariant than
P1's fine-tuned ones) and remove most of the learnable capacity for both; an equivariant backbone is the only
change that would make the comparison test the paper's claim, and D4 frame averaging / flip TTA on the existing
checkpoints is the zero-training way to estimate what it would be worth.

**Q5. Other unfairness?** The DiT gets 14x the conditioning bandwidth of the baseline (7168 vs 512 numbers per
image), sinusoidal time + adaLN-Zero conditioning, and runs 2x faster per epoch; the training recipe (lr, batch,
schedule, EMA, clipping) was fixed on the baseline GATr and reused for everyone; the P1-vs-DiT comparison changes
the input representation and the head allocation at the same time; single seeds everywhere but the baseline; and
the earlier `mlp_heads` ablation (43.7 deg) that motivated the GA heads was not a competent baseline.

---

## 3. Concrete next steps, cheapest first

1. **D4 / flip test-time augmentation on the existing checkpoints** (baseline, DiT, P1; map samples back, medoid
   over 8 x 32): a zero-training measurement of what exact symmetry would be worth. If P1 gains nothing under the
   flip it already handles to 2.6 deg, and everyone gains the same under rolls, F1 is confirmed and the project's
   answer is "no residual value on Pascal3D+".
2. **Rot90 augmentation** (`torch.rot90` k in {0..3}, label `R_z(-90 k) R`) for P1 and the DiT-cells control: the
   one setting where P1's exactness is real and the DiT has to learn four orientations.
3. **The controlled pair**: P1 vs DiT over the same 49 cell tokens + 2-D positions, matched token width, Fourier
   time features in GATr's scalar channels, same epoch time budget, 3 seeds. Report the mv/scalar parameter split
   and an `gatr_mv_channels in {4, 16, 32}` sweep at fixed total.
4. Retire the pooled-token "equivariant" models from the symmetry claim; keep them as flow-matching baselines.
5. Only after 1-3: geometry that actually transforms (depth-lifted points, P2) and an equivariant or
   canonicalised backbone.

---

## Appendix: raw outputs

```
warp_conventions.py
warp roll angle: std 7.71 deg, 99% 20.10, 99.9% 25.96, max 38.10, P(|a|>30) = 2.15e-04, P(|a|>45) = 0.00e+00
up rotated by 90.0 deg: label R1 = Rz(-a) R (residual 8.18e-17); image: rot90(k=+1) diff 0.0017 (k=-1: 0.2402)
up rotated by 10.0 deg: label R1 = Rz(-a) R (residual 4.81e-05)
flip: |R_flip - F R F| = 0.00e+00, |R_flip - R| = 1.39; image: |flip(crop) - crop_flip| = 0.0000

params.py
8.95 baseline: 43,747,104 = backbone 42,605,510 (97.39 %) + condition head 1,043,328 + vector field 97,746
   vector field: scalar->scalar 37.8 %, mv->mv 35.6 %, scalar->mv 14.1 %, mv->scalar 12.5 %
P1: 43,677,536 = backbone 42,605,510 (97.55 %) + vector field 940,242 + token head 131,264
   vector field: scalar->scalar 63.6 %, mv->mv 14.8 %, scalar->mv 11.5 %, mv->scalar 10.0 %
DiT dim=112: condition head 198,336, vector field 990,867; DiT dim=48: 57,536 + 191,187

equiv_e2e.py (6 val thumbnails: boat, car, car, boat, aeroplane, train; velocity rel. err = |v' - expected| / |v|)
8.95 baseline  rot90: head-only 1.3e+00 | end-to-end 1.466 (other-image control 1.239) | ResNet cells 1.558, pooled cos 0.518
               pose: per-sample 109 deg, medoid-of-8 99 deg (floor 1.4)
               flip : head-only 6.4e-01 | end-to-end 0.296 | cells 1.112, pooled cos 0.781
               pose: per-sample median 2.5 deg (mean 17), medoid 2.2 deg [11.0, 0.6, 1.7, 168.8, 5.9, 2.2] (floor 1.0)
P1             rot90: head-only 3.0e-07 | end-to-end 1.199 (control 0.881) | cells 1.292, pooled cos 0.611
               pose: per-sample 114 deg, medoid 121 deg (floor 0.4)
               flip : head-only 2.5e-07 | end-to-end 0.068 | cells 0.692, pooled cos 0.970
               pose: per-sample median 3.0 deg (mean 6.3), medoid 2.6 deg [20.0, 0.9, 3.2, 2.6, 3.6, 1.0] (floor 0.4)

equiv_roll.py (12 val thumbnails; medoid-of-8 pose error between G pred(x) and pred(G x), deg)
backbone flip  | ImageNet (not fine-tuned) cell cos 0.615, pooled 0.973 | baseline 0.349 / 0.613 | P1 0.746 / 0.963
backbone rot90 | ImageNet 0.417 / 0.814 | baseline 0.126 / 0.542 | P1 0.130 / 0.698
baseline condition tokens with the fmap turned 90 deg: max |diff| 7.6e-06
8.95 baseline  floor 0.78 | roll 5: 2.10 | 10: 1.99 | 20: 3.62 | 45: 8.25 (mean 19.7)
P1             floor 0.80 | roll 5: 1.77 | 10: 2.35 | 20: 1.60 | 45: 7.11 (mean 28.6)

epoch times (s): baseline 215, P1 230, P1-small 175, DiT 117, DiT-small 112
```
