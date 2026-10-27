# Spectral bias of MLP, CGENN and GATr on the sphere

Probe: `probes/spectral_bias_sphere/` (see its README for the target and the metric). Models of about 98k parameters (MLP, CGENN, GATr) plus
the paper's 6x256 MLP (330k), full-batch Adam at lr 1e-3, 10,000 steps, 2,000 training points, 3 target draws, T4 kernels.

## Task 1: `lambda(x) = sum_{l=1..8} c_l . Y_l(x)`, the Clifford models also see the constant frame

Steps until the seed-mean `rho_l` first reaches 0.9 (log-spaced checkpoints, so each value is good to about 25%). Equal amplitudes:

| model | l=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| MLP 6x256 (330k) | 11 | 22 | 44 | 70 | 89 | 113 | 180 | 229 |
| MLP (98k) | 22 | 35 | 70 | 89 | 113 | 180 | 289 | 289 |
| CGENN (99k) | 27 | 89 | 180 | 229 | 289 | 367 | 464 | 588 |
| GATr (98k) | 70 | 180 | 289 | 367 | 464 | 588 | 744 | 943 |

All four recover every degree by 10,000 steps (held-out MSE 1e-3 to 1e-2) and all show the spectral bias: low degrees first. CGENN is about
2x and GATr about 3x slower than the MLP at every degree, a uniform delay and not a cutoff. With amplitudes rising from 0.1 to 1,
MLP and CGENN keep the low-first order; GATr learns degrees 2 to 8 at nearly the same time (744 to 943 steps), after degree 1 (289 steps).
Wall clock on a T4 per seed: MLP 23 s, CGENN 9 min, GATr 30 min.

## Comparison with optimist1938's line experiment (`spectral-bias-cgenn-fast3`)

His setup: `sum_i sin(2 pi k_i z + phi_i)`, k = 5..50, 200 points, scalar input `z`, MLP ReLU 6x256 (329,729) vs CGENN (3 blocks of 104,
331,349; `z` as scalar or as `z e1`), Adam, **lr in {1e-4, 3e-4, 1e-3}**, 80,000 steps, 10 seeds. Steps to 90% at k=5 / k=50:

| lr | MLP ReLU | CGENN, scalar | CGENN, `z e1` |
|---|---|---|---|
| 1e-4 | 600 / not reached for k>=30 | 3,000 / 21,100 | 6,700 / 33,100 |
| 3e-4 | 500 / 36,400 | 1,300 / 12,800 | 3,700 / 21,600 |
| 1e-3 | 600 / 13,800 | 700 / 20,200 | 2,000 / 9,700 |

The geometric advantage appears at low learning rates, where the MLP slows sharply at high frequency. At 1e-3, which is the only rate used in
Task 1, his results are mixed too (MLP faster at low k, `z e1` CGENN faster at high k), in line with Task 1. Other differences: his CGENN
has a scalar input (nothing geometric to exploit), both of his models have 330k parameters (ours 98k), 8x more steps, 10 seeds.

## Task 2: pose-dependent target `lambda(x, R) = sum_l c_l . Y_l(R^T x)` (preliminary)

4,000 random `(x, R)` pairs, 10,000 steps, lr 1e-3. MLP (3 seeds), MLP 6x256 (3 seeds), CGENN (2 seeds; the third and GATr were interrupted):

- Both MLPs recover degrees 1 and 2 (`rho` 0.97 / 0.87) and degree 3 only partly (0.45 to 0.54); degrees 4 to 8 stay at 0.0 to 0.3. Held-out MSE 7.7 to 9.6, the total power being 8.
- CGENN has partly recovered all degrees at once (`rho` 0.4 to 0.7), seed-dependent (held-out MSE 6.9 and 3.8), not converged: a flat spectrum, not low-first.
- optimist1938's `gatr-frequency-probe` (GATr flow field regressing a velocity of pure pose frequency l) finds GATr stalls for l >= 4 on fixed data and l >= 3 on fresh samples without the 49 harmonics, and learns every l <= 8 with them.

A rerun of Task 2 with GATr and 3 seeds, and a batched lr sweep of Task 1 (`spectral_ens.py`, 10 seeds, MLP / MLP 6x256 / CGENN), were still running when this was written.


## Update: the lr sweep with 10 seeds (MLP, MLP 6x256, CGENN), the pose-dependent task

* The batched sweep (3 learning rates x 10 seeds per model, 1,000 training points, 80,000 steps for the MLPs, 12,000 for CGENN) is in
  `sphere_sweep_his_format.md`, in the layout of optimist1938's tables, and in `figures/spectral_bias/sphere_lr_sweep.png`,
  `compare_optimist.png` (side by side with his line experiment) and `compare_heatmap_his_style.png`. Every model recovers all eight
  degrees at every learning rate, low degrees first; CGENN needs 1.4 to 4 times more steps than the MLP of its size, so his result
  (MLP fails at high k for lr 1e-4, CGENN does not) does not reproduce on the sphere. His CGENN is also three times deeper (3 blocks of 104,
  331k parameters) than ours (one block of 84, 99k).
* The pose-dependent task and the effect of the pose encoding, width and depth: `pose_encoding_symmetry.md`.
