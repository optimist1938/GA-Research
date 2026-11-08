"""Five-shape toy pose benchmark and its C7 pyramid extension."""

import math
from functools import lru_cache

import numpy as np
import torch
from scipy.spatial.transform import Rotation


_NAMES = ("weighted D2 axes", "tetrahedron", "cube", "pentagon D5", "icosahedron", "7-pyramid")
_PYRAMID_PERIOD = 2 * math.pi / 7
_MAX_POINTS = 12
_CONDITION_KEYS = ("points", "weights", "mask", "class_id")


def _multiply(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv)), dim=-1)


def _conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _exp(vector):
    angle = vector.norm(dim=-1, keepdim=True)
    return torch.cat(((angle / 2).cos(), vector * ((angle / 2).sin() / angle.clamp_min(1e-8))), dim=-1)


def _log(q):
    q = torch.where(q[..., :1] < 0, -q, q)
    sine = q[..., 1:].norm(dim=-1, keepdim=True)
    return q[..., 1:] * (2 * torch.atan2(sine, q[..., :1]) / sine.clamp_min(1e-8))


def _rotate(q, vectors):
    cross = torch.linalg.cross(q[..., 1:], vectors)
    return vectors + 2 * (q[..., :1] * cross + torch.linalg.cross(q[..., 1:], cross))


def _frame(q):
    axes = torch.eye(3, dtype=q.dtype, device=q.device)
    return _rotate(q[:, None, :], axes[None, :, :])


def _random_quaternions(count, generator):
    return torch.nn.functional.normalize(torch.randn(count, 4, generator=generator, dtype=torch.float32), dim=-1)


def _class_count(variant):
    if variant == "original":
        return 5
    if variant == "pyramid":
        return 6
    raise ValueError(f"Unknown toy-pose variant {variant!r}. Choose 'original' or 'pyramid'.")


@lru_cache(maxsize=1)
def _geometry():
    points = torch.zeros(len(_NAMES), _MAX_POINTS, 3, dtype=torch.float32)
    weights = torch.zeros(len(_NAMES), _MAX_POINTS, dtype=torch.float32)
    mask = torch.zeros_like(weights)
    symmetries = []
    golden_ratio = (1 + math.sqrt(5)) / 2

    for index, name in enumerate(("D2", "T", "O", "D5", "I")):
        group = Rotation.create_group(name)
        symmetries.append(torch.tensor(group.as_quat()[:, [3, 0, 1, 2]], dtype=torch.float32))
        if name == "D2":
            vertices = np.concatenate((np.eye(3), -np.eye(3)))
            vertex_weights = [1, 2, 3, 1, 2, 3]
        else:
            if name in ("T", "O"):
                direction = np.ones(3) / math.sqrt(3)
            elif name == "D5":
                direction = np.array([1, 0, 0])
            else:
                direction = np.array([golden_ratio, 0, 1])
                direction /= np.linalg.norm(direction)
            vertices = np.unique(np.round(group.apply(direction), 7), axis=0)
            vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
            vertex_weights = [1] * len(vertices)
        points[index, :len(vertices)] = torch.as_tensor(vertices, dtype=torch.float32)
        weights[index, :len(vertices)] = torch.tensor(vertex_weights, dtype=torch.float32)
        mask[index, :len(vertices)] = 1

    # SYMSOL-CHILE proportions, with the base in z = -1 / sqrt(2).
    angle = torch.arange(7, dtype=torch.float32) * _PYRAMID_PERIOD
    radius = 1 / math.sqrt(2)
    base = torch.stack((radius * angle.cos(), radius * angle.sin(), torch.full_like(angle, -radius)), dim=-1)
    points[5, :7] = base
    points[5, 7, 2] = radius
    weights[5, :8] = 1
    mask[5, :8] = 1
    zero = torch.zeros_like(angle)
    symmetries.append(torch.stack(((angle / 2).cos(), zero, zero, (angle / 2).sin()), dim=-1))
    return points, weights, mask, tuple(symmetries)


def canonical_geometry(variant="original"):
    """Return names, padded vertices, vertex weights, masks, and proper symmetry groups."""
    count = _class_count(variant)
    points, weights, mask, groups = _geometry()
    return {
        "class_names": _NAMES[:count],
        "points": points[:count].clone(),
        "weights": weights[:count].clone(),
        "mask": mask[:count].clone(),
        "symmetries": tuple(group.clone() for group in groups[:count]),
    }


def _nearest_equivalent(pose, truth, class_id):
    result = torch.empty_like(pose)
    groups = _geometry()[3]
    for index in class_id.unique().tolist():
        if not 0 <= index < len(groups):
            raise ValueError(f"Unknown toy-pose class ID {index}.")
        selected = class_id == index
        symmetry = groups[index].to(pose)
        modes = _multiply(truth[selected, None, :], symmetry[None, :, :])
        scores = (modes * pose[selected, None, :]).sum(dim=-1)
        nearest = scores.abs().argmax(1, keepdim=True)
        chosen = modes.gather(1, nearest[..., None].expand(-1, -1, 4)).squeeze(1)
        result[selected] = torch.where(scores.gather(1, nearest) < 0, -chosen, chosen)
    return result


def _observation(truth, class_id):
    points, weights, mask, _ = _geometry()
    return {
        "points": _rotate(truth[:, None, :], points[class_id]),
        "weights": weights[class_id],
        "mask": mask[class_id],
        "class_id": class_id,
        "truth": truth,
    }


def _flow_split(count, seed, variant, scale=None):
    if count < 1:
        raise ValueError("A flow split must contain at least one example.")
    generator = torch.Generator().manual_seed(seed)
    class_id = torch.randint(_class_count(variant), (count,), generator=generator)
    truth = _random_quaternions(count, generator)
    start = _random_quaternions(count, generator)
    endpoint = _nearest_equivalent(start, truth, class_id)
    velocity = _log(_multiply(endpoint, _conjugate(start)))
    time = 0.01 + 0.98 * torch.rand(count, generator=generator, dtype=torch.float32)
    pose = _multiply(_exp(time[:, None] * velocity), start)
    if scale is None:
        scale = velocity.square().sum(-1).mean().sqrt()
    target = velocity.new_zeros(count, 1, 8)
    # Spatial angular velocity (x, y, z) in the (e12, e13, e23) slots.
    target[:, 0, 4:7] = velocity[:, [2, 1, 0]] * velocity.new_tensor([1, -1, 1]) / scale
    return {
        **_observation(truth, class_id),
        "start_rotation": start,
        "endpoint": endpoint,
        "pose": pose,
        "frame": _frame(pose),
        "time": time,
        "angular_velocity": velocity,
        "velocity_scale": scale,
        "target": target,
    }


def make_flow_splits(variant="original", *, train_size=4096, val_size=1024, train_seed=301, val_seed=302):
    """Return fixed train and validation splits for nearest-coupled flow matching."""
    train = _flow_split(train_size, train_seed, variant)
    validation = _flow_split(val_size, val_seed, variant, train["velocity_scale"])
    return train, validation


def make_pose_test(variant="original", *, seed=7701):
    """Return the original pose test or the dedicated icosahedron/C7 test."""
    if _class_count(variant) == 5:
        class_id = torch.arange(5).repeat_interleave(64)
    else:
        class_id = torch.tensor([4] * 64 + [5] * 256)
    generator = torch.Generator().manual_seed(seed)
    truth = _random_quaternions(len(class_id), generator)
    start = _random_quaternions(len(class_id), generator)
    return {**_observation(truth, class_id), "start_rotation": start}


def pose_error_degrees(prediction, truth, class_id):
    """Return the geodesic angle to the nearest proper symmetry equivalent."""
    equivalent = _nearest_equivalent(prediction, truth, class_id)
    return torch.rad2deg(_log(_multiply(equivalent, _conjugate(prediction))).norm(dim=-1))


def pyramid_tilt_twist_degrees(prediction, truth):
    """Return apex-axis tilt and residual C7 twist after removing that tilt.

    Residual twist is undefined for exactly antiparallel axes. Report NaN in
    that case so callers can handle it explicitly when aggregating.
    """
    relative = _multiply(_conjugate(truth), prediction)
    axis_components = relative[:, [0, 3]].norm(dim=-1)
    tilt = 2 * torch.atan2(relative[:, 1:3].norm(dim=-1), axis_components)
    twist = 2 * torch.atan2(relative[:, 3], relative[:, 0])
    twist = (twist + _PYRAMID_PERIOD / 2).remainder(_PYRAMID_PERIOD) - _PYRAMID_PERIOD / 2
    twist = twist.abs().masked_fill(axis_components <= 1e-8, float("nan"))
    return torch.rad2deg(tilt), torch.rad2deg(twist)


@torch.no_grad()
def rollout(field, test, *, steps=32):
    """Integrate a field returning spatial angular velocity in radians per flow unit."""
    if steps < 1:
        raise ValueError("The rollout must have at least one step.")
    condition = {key: test[key] for key in _CONDITION_KEYS}
    pose = test["start_rotation"]
    for index in range(steps):
        time = pose.new_full((len(pose),), (index + 0.5) / steps)
        velocity = field({**condition, "pose": pose, "frame": _frame(pose), "time": time})
        if velocity.shape != (len(pose), 3):
            raise ValueError("Expected a field returning angular velocities of shape (batch, 3).")
        pose = torch.nn.functional.normalize(_multiply(_exp(velocity / steps), pose), dim=-1)
    return pose
