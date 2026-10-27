# Why P1 stopped learning on SYMSOL I (10 % data), and the fix

Runs: `samson12345423/symsol-p1-small-10pct` v4 (P1-small, `--no-gatr_zero_readout`) and
`grig0ryz/symsol-dit-small-10pct` (DiT-small), 4,500 train / 1,000 test views per shape, 100 epochs.
Diagnosis scripts: `~/rfeval/symsol/{diagnose,grad_diag,gate_diag}.py` on the final (EMA) checkpoints.

## Result

| log-likelihood (uniform -2.29) | avg | tet | cube | icosa | cone | cyl |
|---|---|---|---|---|---|---|
| DiT-small | 3.92 | 4.65 | 2.19 | -2.29 | 7.76 | 7.31 |
| P1-small | 1.95 | -2.29 | -2.29 | -2.29 | 8.68 | 7.96 |

P1's training loss went flat at ~epoch 20-30 at 4.30 and stayed there.

## Diagnosis

1. **The plateau is "cone + cylinder solved, polyhedra off".** Per shape, against the zero predictor and the
   Bayes-optimal (oracle) velocity E[xi | R_t] computed in closed form from the equivalent poses: P1 reaches 100 %
   of the achievable loss reduction on the cone and cylinder and 0 % on the tetrahedron, cube and icosahedron,
   where |v| is 0.004-0.006. (cone 2.35 + cyl 3.26 + 3 x ~5.30) / 5 = 4.30, the plateau. DiT: tet 52 %,
   cube 40 %, icosa 0 %.
2. **Why the polyhedra start with no signal.** With R1 drawn at random among 12 / 24 / 60 equivalents the
   targets cancel: the oracle is barely better than v = 0 until t ~ 0.5 (tet) or t ~ 0.75 (icosa). The cone and
   cylinder (a circle of equivalents) give a strong signal at every t, so they are learned first.
3. **How P1 turns them off: magnitude.** The pose token's hidden state before the read-out is ~15x smaller for
   polyhedra images (rotation bivector 0.86 vs 11.8; whole multivector 4.4 vs 75). GATr's layers are
   multiplicative (geometric products, gating), so small activations mean small gradients.
4. **Why it never recovers.** At the final weights the polyhedra gradients are 28-100x smaller than the
   cone / cylinder ones on the same shared weights (vector field 0.30 vs 11.3, backbone 0.10 vs 2.8). Adam scales
   each step by the weight's RMS gradient, now set by the cone / cylinder noise, so the polyhedra train at an
   effective learning rate ~30-100x lower. A descent direction exists (oracle-target gradient signal 0.55-0.59
   vs a 0.35 noise floor) but the steps are negligible. DiT's additive residual stream keeps polyhedra
   activations (and gradients, 11.9) large, so it kept learning tet and cube; it switched off only the icosahedron.
5. The earlier stall (v3, loss 5.29 from epoch 1) was a different, now-fixed problem: the zeroed read-out left
   one learnable direction and no gradient upstream (`FlowConfig.gatr_zero_readout`).

## Fix

`--symsol_nearest_coupling`: each training R0 flows to the nearest of the image's equivalent poses. The targets
no longer cancel at any t (zero-predictor loss ~1.15 instead of 5.29), so there is nothing to switch off; from a
uniform R0 every equivalent is still chosen equally often (tested), so the flow learns the same symmetric
distribution and the likelihood keeps its meaning. It applies to both models alike. At P1's collapsed weights it
does not help (gradients there are tiny), so it needs training from scratch.

Fallback if P1 still switches shapes off: normalise the pose token before the read-out, so "off" cannot be
implemented by shrinking magnitudes.
