# FFT-flow: GATr on point clouds learns the polyhedra only with explicit pose harmonics

Branch `FFT-flow`. Background and the hypothesis: `reports/frequency_hypothesis_ru.md` (why a GATr denoiser never
learned SYMSOL's tetrahedron / cube / icosahedron: their velocity fields have no angular frequency below
l = 3 / 4 / 6 in the pose, and at initialisation the GATr field only reaches l <= 2).

## The experiment

SYMSOL I as point clouds (`datasets/symsol_points.py`, `--dataset symsol_points`): the five shapes are generated,
512 points are sampled from the surface, the cloud is turned by the pose (x = R x_canonical), and every symmetric
equivalent counts as a correct label. Two models that differ **only** in 49 input scalars per point:

| Run | Flags (besides the protocol) | Params |
|---|---|---|
| control | `--vector_field gatr --cond_tokens points --pose_tokens frame_pin --condition_head none --no-gatr_zero_readout --gatr_mv_channels 13 --gatr_s_channels 52` | 253,169 |
| **+ pose harmonics** | the same `+ --gatr_pose_harmonics 6` | 256,354 |

One GATr over [pose token ((R e1) e123, R e2, R e3); one token per point (vector x_i, scalar |x_i|)], velocity read
as a bivector (`GATrCloudField` in `models/gatr_denoiser.py`). With `--gatr_pose_harmonics 6` every point token
also gets the real spherical harmonics Y_lm(u_i), l <= 6 (49 numbers), of its position in the **current pose's
frame**, u_i = R_t^T x_i / |R_t^T x_i|. They are invariant under turning cloud and pose together, so the model stays
exactly equivariant (`tests/test_modelnet_points.py::test_gatr_with_pose_harmonics_is_invariant_and_sees_them`).

Protocols (both with `--points_n 512`):
- **10 %** (as the image runs): `--symsol_train_views 4500 --points_fixed_train --symsol_test_views 1000
  --no-ema --optimizer adamw --lr 1e-4 --lr_scaling sqrt --lr_schedule step --warmup_epochs 0 --n_epochs 40
  --batch_size 64` (4,500 fixed clouds per shape; test 1,000 per shape, as the likelihood over 512 tokens per flow
  step would not fit Kaggle's 12 h on 5,000);
- **unlimited data**: `--symsol_train_views 4500 --symsol_test_views 1000 --ema --lr 7.07e-4 --lr_scaling sqrt
  --n_epochs 60 --batch_size 64` (fresh clouds every epoch).

## Results (exact log-likelihood, uniform = -2.29; error = mean over shapes of the median nearest-equivalent error)

| | Avg | tet | cube | icosa | cone | cyl | Error |
|---|---|---|---|---|---|---|---|
| 10 %, control | 1.47 | -2.29 | -2.29 | -2.29 | 8.33 | 5.87 | 25.5° |
| **10 %, + pose harmonics** | **7.58** | **8.87** | **8.24** | **5.91** | 8.35 | 6.51 | **5.7°** |
| unlimited, control | 8.82 | 9.47 | 11.39 | 5.56 | 9.21 | 8.47 | 9.2° |
| **unlimited, + pose harmonics** | **10.96** | **13.33** | **12.62** | **10.30** | 10.03 | 8.55 | **0.64°** |

On 10 % the control sits on the plateau every GATr reached on SYMSOL (cone and cylinder learned, polyhedra at
uniform); with the harmonics all five shapes are learned. Same 10 % protocol, other models: SO(3)-Fourier on the same
points 6.32, GATr 8.95-style -0.05, DiT -0.96.

A cheaper form keeps the points out of the flow loop: a GATr encoder (once per cloud) weights the harmonics at the
points' own directions, followed by the SO(3)-Fourier read-out (`--cond_tokens pointgatrsphere`,
`models/so3fourier_head.py::PointGATrSphereHead`): 8.70 on the 10 % protocol (full test 5,000), 85 s per epoch
instead of 400 s. Harmonics of *learned* directions (`--cond_tokens pointcloud --gatr_pose_harmonics 6`, the
condition tokens' vector parts) did not help (-0.19).

## Running

Notebooks were built from the code at this commit and run on Kaggle (RTX Pro 6000, one GPU); the trainer is
`python -m pose3d --dataset symsol_points --path_to_datasets <any dir> <flags above>`. No data files are needed:
the clouds are generated. Tests: `pytest tests/test_modelnet_points.py`.
