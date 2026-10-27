# Spectral bias on the sphere

Rahaman et al. (2019, "On the spectral bias of neural networks", Fig. 1) fit a sum of sinusoids on a line and watch which
frequencies a ReLU MLP learns first. This probe does the same with spherical harmonics, and compares an MLP, the project's CGENN
(`TralaleroTralala`) and GATr. The line version for MLP vs CGENN is optimist1938's `probes/spectral_bias` (80k steps, lr sweep, 10 seeds).

Target on S^2: `lambda(x) = sum_{l=1..L} A_l c_l . Y_l(x)` (L = 8, random unit `c_l` per degree, real spherical harmonics in
`component` normalisation, so every degree has power `A_l^2`; amplitudes equal, or rising from 0.1 to 1). The measured quantity is
`rho_l(t)`, the fraction of the target's degree `l` that the model has recovered: least-squares harmonic coefficients of the model on a
held-out grid (degrees 0..11, so leakage into other degrees is separated), projected on `A_l c_l`.

| script | what it does |
|---|---|
| `spectral_bias_sphere.py` | the sphere version, one run at a time. MLP sees `x`; CGENN and GATr also see the constant frame `e1, e2, e3` (an equivariant network that sees only `x` can output only a constant on the sphere) |
| `spectral_bias_pose.py` | pose-dependent target `lambda(x, R) = sum_l A_l c_l . Y_l(R^T x)`: invariant under turning `x` and `R` together; CGENN and GATr see the pose as the frame `(R e1) e123, R e2, R e3` (P1's pose token), the MLP the 12 raw numbers. `<x, R e_k> = (R^T x)_k` gives them the coordinates of `R^T x` |
| `spectral_ens.py` | batched version (MLP, MLP 6x256, CGENN): all `lr x seed` runs of a model trained at once, one CUDA graph per step; the CGENN layers are re-implemented with a leading ensemble dim after optimist1938's `ensemble.py` and checked against the library in `tests/test_spectral_bias_sphere.py`. `--budget_s` cuts the steps to a time budget |
| `scale_pose.py`, `scale_pose_depth.py` | the pose task at about 4x the parameters by width (`scale_pose.py`: MLP 6x280, CGENN 160 channels, GATr 16/64) or by depth (`scale_pose_depth.py`: 24-layer MLP, CGENN 2 and 3 blocks, GATr 16 blocks); one seed, steps cut to a per-model time cap |
| `scale_pose_rotor.py` | the pose as a rotor (GATr's `embed_rotation`), with `x` as a vector or as GATr's point trivector (`embed_point`); `pose_encoding_symmetry.md` explains why this fails |
| `scale_pose_polar.py` | the pose as three ordinary vectors `R e1, R e2, R e3` (replaces `frame_mv`): the encoding that works |
| `plot_spectral.py` | heatmaps (Fig. 1 style), steps-to-90% and held-out MSE from the result JSONs |

```bash
python probes/spectral_bias_sphere/spectral_bias_sphere.py --device cuda --steps 10000 --amps equal --seeds 3 --out sphere_equal.json
python probes/spectral_bias_sphere/spectral_bias_pose.py   --device cuda --steps 10000 --amps equal --seeds 3 --out pose_equal.json
python probes/spectral_bias_sphere/spectral_ens.py --device cuda --lrs 1e-4 3e-4 1e-3 --seeds 10 --steps 80000 --budget_s 5400 --out ens.json
python probes/spectral_bias_sphere/plot_spectral.py sphere_equal.json sphere_increasing.json spectral.png
```

Parameter budgets: the models are matched to the small GATr (8 multivector / 32 scalar channels, 4 blocks, 97,762 parameters) within 1%
(integer widths); the paper's 6x256 MLP (330k) is kept as a reference. Results and caveats: `reports/sphere_spectral_bias.md` (sphere task, comparison with the line experiment) and `reports/pose_encoding_symmetry.md` (pose-dependent task: encodings, symmetry, width and depth).
