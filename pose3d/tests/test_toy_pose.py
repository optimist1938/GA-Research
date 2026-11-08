import math

import pytest
import torch

from pose3d.toy_pose import (
    canonical_geometry,
    make_flow_splits,
    make_pose_test,
    pose_error_degrees,
    pyramid_tilt_twist_degrees,
    rollout,
)


@pytest.mark.parametrize("variant,counts", [("original", [64] * 5), ("pyramid", [0, 0, 0, 0, 64, 256])])
def test_fixed_splits_and_rng(variant, counts):
    state = torch.get_rng_state()
    train, val = make_flow_splits(variant)
    test = make_pose_test(variant)
    replay = make_pose_test(variant)
    assert torch.equal(torch.get_rng_state(), state)
    assert train["points"].shape == (4096, 12, 3)
    assert val["points"].shape == (1024, 12, 3)
    assert torch.equal(train["velocity_scale"], val["velocity_scale"])
    assert test["class_id"].bincount().tolist() == counts
    assert all(torch.equal(test[key], replay[key]) for key in test)
    assert torch.count_nonzero(test["points"][test["mask"] == 0]) == 0


def test_symmetry_equivalent_poses():
    geometry = canonical_geometry("pyramid")
    groups = geometry["symmetries"]
    assert [len(group) for group in groups] == [4, 12, 24, 10, 60, 7]
    assert geometry["mask"].sum(1).tolist() == [6, 4, 8, 5, 12, 8]
    classes = torch.arange(len(groups))
    truth = torch.zeros(len(groups), 4)
    truth[:, 0] = 1
    prediction = -torch.stack([group[-1] for group in groups])
    torch.testing.assert_close(pose_error_degrees(prediction, truth, classes),
                               torch.zeros(len(groups)), atol=1e-4, rtol=0)


@pytest.mark.parametrize("variant", ["original", "pyramid"])
def test_oracle_flow_reaches_the_endpoint(variant):
    train, _ = make_flow_splits(variant, train_size=128, val_size=32)

    def field(inputs):
        assert set(inputs) == {"points", "weights", "mask", "class_id", "pose", "frame", "time"}
        return train["angular_velocity"]

    prediction = rollout(field, train)
    error = pose_error_degrees(prediction, train["truth"], train["class_id"])
    torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-4, rtol=0)


def test_pyramid_twist_wraps_and_axis_flip_is_undefined():
    truth = torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)
    angle = 2 * math.pi / 7 + math.radians(10)
    prediction = torch.tensor([[math.cos(angle / 2), 0., 0., math.sin(angle / 2)],
                               [0., 1., 0., 0.]])
    tilt, twist = pyramid_tilt_twist_degrees(prediction, truth)
    torch.testing.assert_close(tilt, torch.tensor([0., 180.]))
    torch.testing.assert_close(twist[0], torch.tensor(10.), atol=1e-4, rtol=0)
    assert twist[1].isnan()
