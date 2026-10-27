# How the pose is encoded decides what a GATr / CGENN can learn: the pose-dependent spherical-harmonic benchmark

Scripts: `probes/spectral_bias_sphere/` (`spectral_bias_pose.py`, `scale_pose*.py`, `spectral_ens.py`). Companion report on the
sphere task and the comparison with optimist1938's line experiment: `sphere_spectral_bias.md`. Figures: `figures/spectral_bias/`.

## 1. The benchmark and its symbols

* `x` in S^2 is a unit vector (the input direction); `R` in SO(3) is a rotation matrix (the pose); `a = R e1`, `b = R e2`, `c = R e3` are
  the three columns of `R`.
* `u = R^T x` is `x` written in the frame of `R`: `u_k = <x, R e_k>`, `<., .>` the inner product.
* `Y_l(u)` in R^(2l+1) is the vector of real spherical harmonics of degree `l`; `c_l` in R^(2l+1) a fixed random unit vector per degree
  (drawn once per seed); `A_l` the amplitude of degree `l` (all 1 in the "equal" setting).
* Target: `lambda(x, R) = sum_{l=1..8} A_l <c_l, Y_l(R^T x)> =: F(u)`. Training set: 4,000 random `(x, R)`; held-out: 3,000.
* `rho_l` is the recovered fraction of degree `l`: the model's outputs on held-out points are fitted by least squares with
  harmonics of `u` (degrees 0..11); `rho_l = <c_hat_l, A_l c_l> / A_l^2` with `c_hat_l` the fitted degree-`l` coefficients.
  `rho_l = 1`: the degree is fully learned; `0`: not at all. The target's variance is 8 (the MSE of always predicting 0).

The target is invariant under turning the point and the pose together, `(x, R) -> (g x, g R)` for every rotation or reflection `g`
(`<g x, g R e_k> = <x, R e_k>`). A model is only as good as the match between this action and the action its architecture imposes on
its input tokens: GATr and CGENN are equivariant under all of O(3) (rotations and reflections), with each token transforming
according to its grade.

## 2. Encodings and results

(Seed 0, about 98k parameters, lr 1e-3, equal amplitudes; GATr 8 multivector / 32 scalar channels, 4 blocks; CGENN: one block of
84 channels; MLP: 6 hidden layers. `e_i`, `e_ij`, `e_123` are the vector, bivector and pseudoscalar blades of Cl(3,0); `e_0` is the
extra null direction of GATr's 16-component algebra.)

| # | Encoding | Model symmetry on the inputs | Result on `lambda(x, R)` |
|---|---|---|---|
| 1 | MLP: the 12 numbers `(x, vec R)` | none | **fails**: `rho` = 1.08, 0.68, 0.35, 0.08, 0.03, 0.05, 0.00, 0.01 (degrees 1..8); held-out MSE 9.6 (> 8): it memorises |
| 2 | `x` a vector; pose `(a) e_123` as a bivector with `(e12, e13, e23) = (a3, -a2, a1)`, plus the vectors `b`, `c` (the SYMSOL model's `frame_pin`) | `x, b, c -> g x, g b, g c`; the bivector `-> det(g) g a e_123` (axial) | **capped**: `rho` = 0.28, 0.60, 0.21, 0.22, 0.48, 0.70, 0.67, 0.37 (GATr), 0.18, 0.67, 0.13, 0.14, 0.49, 0.71, 0.66, 0.30 (CGENN); held-out MSE 7.4 / 6.9 |
| 3 | pose as a rotor `q = q0 + q12 e12 + q13 e13 + q23 e23` (GATr's `embed_rotation`), `x` a vector | `x -> g x`, `q -> g q g^-1` (i.e. `R -> g R g^T`) | **fails**: `rho` about 0 on every degree (-0.04..0.07); held-out MSE 9.8 (CGENN), 12.5 (GATr) |
| 4 | rotor pose and `x` as GATr's point trivector (`embed_point`: `x -> e_123 - x1 e_023 + x2 e_013 - x3 e_012`) | as 3 | **fails** the same way: `rho` -0.01..0.06; held-out MSE 9.3 |
| 5 | `x`, `a`, `b`, `c` all as grade-1 vectors | `(x, a, b, c) -> (g x, g a, g b, g c)` | **succeeds**: CGENN `rho` = 1.00 on every degree, held-out MSE 0.003; GATr `rho` = 0.92, 0.94, 1.00, 1.00, 0.99, 1.00, 1.01, 1.00, held-out MSE 0.026; low degrees first |

Why:

* **Encoding 2.** Two inputs that differ only in the sign of `u_1 = <x, a>` (with `u_2`, `u_3` and `|x|` unchanged) are related by a
  reflection, so a model with reflection symmetry must give them the same output: it can only represent the part of `F` that is
  even in `u_1`. The fraction of each degree's energy in that part (computed for the seed-0 target) is
  0.24, 0.67, 0.19, 0.20, 0.53, 0.71, 0.67, 0.31, which is what the models reach. Width and depth do not move it (section 3).
* **Encodings 3 and 4.** Under the action of a rotor token, `lambda(g x, g R g^T) = F(g u)`. Averaged over `g`, `F(g u)` is the mean
  of `F` over the sphere, which is 0 for degrees >= 1: an exactly invariant model can only output 0. Checked numerically on
  untrained models: the output is invariant to ~1e-7 on the 78% of inputs where `matrix_to_rotor` keeps the sign of `q`, and differs on
  the 22% where it flips to `-q` (a small leak, not enough to learn anything).
* **Encoding 5.** `u_k` are plain inner products of grade-1 vectors; the target is invariant under exactly the symmetry the model has.

So: the encoding must make the symmetry of the target the symmetry of the model. This is a statement about this benchmark.

## 3. Width and depth (encoding 2, seed 0, 6,000 steps, about 4x the parameters)

| Model | width x4 held-out MSE / mean `rho` | depth x4 held-out MSE / mean `rho` |
|---|---|---|
| MLP | 10.0 -> 8.1 / 0.31 -> 0.31 | 10.0 -> 8.0 / 0.31 -> 0.00 (24 layers did not train) |
| CGENN | 6.3 -> 6.5 / 0.42 -> 0.43 | 6.3 -> 9.8 (2 blocks), 10.0 (3 blocks of 64) / 0.42 -> 0.42, 0.49 |
| GATr | 6.8 -> 7.6 / 0.44 -> 0.43 | 6.8 -> 6.6 (16 blocks) / 0.44 -> 0.42 |

Neither helps: the recovery per degree sits at the encoding-2 ceiling of section 2; more depth makes CGENN reach it sooner and GATr
later. These runs say nothing about capacity, because the encoding caps them.

## 4. What this does not show

* A single seed per configuration (the ceiling is a property of that seed's `c_l`; the 3-seed means of the base runs agree).
* The benchmark's symmetry is not that of single-view pose estimation. There the symmetries are the camera roll (`R -> r_z R`
  for rotations `r_z` about the optical axis, with the image rotated), the left-right image flip (`R -> F R F`, which makes `a`
  axial and `b`, `c` polar: the reason `frame_pin` uses the bivector for `a`) and the shape's own symmetry group acting on the right
  (`R -> R s`). The all-vectors frame (encoding 5) matches none of these, so its success here does not transfer.
* The experiment on SYMSOL images (the default CliffordFlow with `--pose_tokens frame_ch`, i.e. encoding 5) is in section 5.

## 5. SYMSOL I images, default CliffordFlow, `--pose_tokens frame_ch`

What was run. The default CliffordFlow (a Clifford-MLP condition head turns the pooled ResNet-50 features into 64 multivectors; a GATr
vector field reads them together with the pose and the flow time) with `--pose_tokens frame_ch`: the pose enters as the three vectors
`R e1, R e2, R e3` (the columns of the rotation matrix `R`) in three channels of one GATr token, instead of one rotor token, and the
velocity is read out in the camera frame (encoding 5 of section 2). Protocol of the repo's `symsol-stf-runner` (`notebooks/symsol-stf-runner.ipynb`,
variant `frame_ch`): SYMSOL I, 10 % of the training data (4,500 views per shape), the first 1,000 test views per shape, ResNet-50,
AdamW lr 1e-4 x sqrt(2), 5 warmup epochs + cosine, no EMA, batch 64, 40 epochs. `--no-symsol_likelihood` switches the flow log-likelihood off,
so the metrics are the rotation error to the nearest equivalent pose (acc@15 / acc@30: the share of test images whose error is below 15 /
30 degrees; acc@15 is the primary one). One seed, one RTX Pro 6000, 2,045 s. Kernel `syfry5suvzovvakmuj/symsol-framech-rtx2`, code
`04352e9` of this branch.

Result (final evaluation, 32 flow samples per image):

| | median error (deg) | mean of per-shape medians (deg) | acc@15 | acc@30 |
|---|---|---|---|---|
| `frame_ch` (this run) | 27.38 | 26.06 | 0.410 | 0.537 |

| shape | tetrahedron | cube | icosahedron | cone | cylinder |
|---|---|---|---|---|---|
| median error (deg) | 54.45 | 42.28 | 31.51 | 0.66 | 1.38 |
| acc@15 | 0.013 | 0.027 | 0.053 | 1.000 | 0.956 |

Reading. The cone and cylinder are learned to about a degree; the three polyhedra are not (acc@15 between 1 and 5 %), which is the
pattern of the older flow runs on these images. For reference, `reports/i2s_symsol1_10pct.md` lists the earlier runs on the same protocol:
the spherecloud GATr flow (`frame_pin`, the bivector for `R e1`): 26.63 deg / 25.62 deg / acc@15 0.418; the same with 49 pose harmonics
(`harm6`): 16.97 / 19.88 / 0.481; Image2Sphere: 3.07 / 3.21 / 1.000. So the all-vectors pose frame neither helps nor hurts here.

Caveats. (1) There is no matched control: a run with the default rotor token on this code, protocol and encoder was not made (the team's
`symsol1_10pct_gatr_control` was still running when this was written), so the comparison above is against different models. (2) One seed.
(3) The all-vectors frame matches the symmetry of our synthetic benchmark (section 1), not the symmetries of single-view pose estimation
(section 4); the image-flip symmetry, for instance, is what the bivector in `frame_pin` is built for. (4) The same encoding on SYMSOL
point clouds, run by another team member (`avgurpetrovich/pts2-gflow-sp10s-polar`, flag `--pin_polar`, 512 points, GATr flow), gave log-likelihood
1.412 against 1.47 for the bivector frame, with the tetrahedron, cube and icosahedron still at the uniform value -2.29: also no change.

History of the run (for the record). Two earlier launches (an RTX kernel and a T4 kernel) crashed at the end of their first epoch with
`AttributeError: 'RunConfig' object has no attribute 'viz_every'`: the merge of `origin/main` into this branch had dropped
`RunConfig.log_viz / viz_every / viz_images` and `I2SConfig.wigner_loss_*`. Kaggle kept reporting the crashed kernels as running for hours.
The fields were restored (commit `04352e9`) and `tests/test_config_fields.py` now checks that every `cfg.<group>.<field>` the code reads
exists. A duplicate of the final run on another account (`grig0ryz/symsol-framech-rtx2`) could not be cancelled and is expected to give the same numbers.
