# Toy SO(3) pose benchmark

Generate the exact raw point-cloud splits used in the short pose-flow experiments. The original variant contains weighted D2 axes, a tetrahedron, a cube, a D5 pentagon, and an icosahedron. The pyramid variant adds a regular 7-sided pyramid with the same vertex proportions as SYMSOL-CHILE.

The generator needs PyTorch, NumPy, and SciPy. It runs on CPU and uses local random generators. Run these commands from the repository's `pose3d` directory.

```bash
python -m pose3d.toy_pose --variant original --out data/toy-original
python -m pose3d.toy_pose --variant pyramid --out data/toy-pyramid
```

Each command writes `train.npz`, `val.npz`, `test.npz`, `geometry.npz`, and `metadata.json`. The defaults use 4,096 train and 1,024 validation examples with seeds 301 and 302. The independent pose test uses seed 7701. It contains 64 clouds per class in the original variant, and 64 icosahedra plus 256 pyramids in the pyramid variant.

## Data format

| Array | Meaning |
| --- | --- |
| `points`, `weights`, `mask` | Rotated vertices, scalar weights, and a 0/1 validity mask. Clouds are padded to 12 vertices. |
| `class_id` | Zero-based index into the class names in `metadata.json`. |
| `truth`, `start_rotation` | Ground-truth and random initial rotations as unit quaternions in `(w, x, y, z)` order. |
| `pose`, `frame`, `time` | Flow-only arrays. Intermediate pose quaternion, its three axes as rows, and time in `(0, 1)`. |
| `endpoint` | Flow-only array. The symmetry-equivalent target nearest to the initial pose. |
| `angular_velocity` | Flow-only array. The spatial velocity of the shortest geodesic to the endpoint, in radians per unit of flow time. |
| `target`, `velocity_scale` | Flow-only arrays. The velocity is divided by the training RMS and packed as `(z, -y, x)` in slots 4–6 of a `(N, 1, 8)` multivector target. |

`geometry.npz` contains the canonical clouds and one `symmetry_<class_id>` array of proper symmetry quaternions per class. The benchmark supplies ground-truth pose for supervision and scoring. At inference, the observations consist of points, weights, mask, and class, with `pose`, `frame`, and `time` added for a flow model.

## Python API

```python
from pose3d.toy_pose import make_flow_splits, make_pose_test, pose_error_degrees, rollout

train, val = make_flow_splits("original")
test = make_pose_test("original")

# field(batch) returns an (N, 3) tensor of spatial angular velocities.
# For a normalized multivector model, convert (z, -y, x) back to (x, y, z)
# and multiply by train["velocity_scale"] before returning it.
predicted = rollout(field, test, steps=32)
errors = pose_error_degrees(predicted, test["truth"], test["class_id"])
```

Report the median of the per-cloud errors for each shape. For the 7-pyramid, also report axis tilt and residual twist separately.

```python
from pose3d.toy_pose import pyramid_tilt_twist_degrees

is_pyramid = test["class_id"] == 5
tilt, twist = pyramid_tilt_twist_degrees(predicted[is_pyramid], test["truth"][is_pyramid])
```

Use the pyramid variant for this diagnostic. Twist is scored modulo (360^\circ / 7), with a random-twist median of (90^\circ / 7 \approx 12.86^\circ). Twist is undefined at exactly antiparallel axes and is returned as NaN there.
