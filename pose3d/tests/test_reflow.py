"""Reflow: the frozen teacher stays out of the student's weights, gradients and EMA copy."""

import copy

import torch
from clifford.algebra.cliffordalgebra import CliffordAlgebra

from pose3d.config import parse_args
from pose3d.models.clifford_flow import CliffordFlow

ALGEBRA = CliffordAlgebra((1, 1, 1))


def _flow(**kwargs):
    return CliffordFlow(ALGEBRA, hidden_dim=[16], n_cond_mv=8, pretrained_backbone=False,
                        encoder_type="resnet50", conv_adapter=False, n_time_samples=2, **kwargs)


def _student_and_teacher():
    torch.manual_seed(0)
    teacher = _flow()
    # A non-zero velocity field, so the teacher's couplings differ from its source noise.
    for p in teacher.vector_field.parameters():
        torch.nn.init.normal_(p, std=0.05)
    student = _flow()
    student.load_state_dict(teacher.state_dict())
    student.set_reflow_teacher(teacher, steps=3)
    return student, teacher


def test_teacher_is_not_part_of_the_student():
    student, teacher = _student_and_teacher()
    assert set(student.state_dict()) == set(teacher.state_dict())
    assert sum(p.numel() for p in student.parameters()) == sum(p.numel() for p in teacher.parameters())
    student.train()
    assert not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())


def test_ema_copy_shares_the_teacher():
    student, teacher = _student_and_teacher()
    clone = copy.deepcopy(student)
    assert clone._reflow.module is teacher
    assert clone.vector_field is not student.vector_field


def test_reflow_loss_trains_the_student_only():
    student, teacher = _student_and_teacher()
    img = torch.rand(3, 3, 64, 64)
    rot = torch.eye(3).expand(3, 3, 3).clone()
    loss = student.compute_loss(img, rot)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_reflow_targets_come_from_the_teacher():
    # Ground truth is ignored: two different targets give the same loss on the same noise.
    student, _ = _student_and_teacher()
    student.eval()
    img = torch.rand(2, 3, 64, 64)
    a = torch.eye(3).expand(2, 3, 3).clone()
    b = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]]).expand(2, 3, 3).clone()
    torch.manual_seed(1)
    loss_a = student.compute_loss(img, a)
    torch.manual_seed(1)
    loss_b = student.compute_loss(img, b)
    torch.testing.assert_close(loss_a, loss_b)


def test_predict_uses_sample_steps():
    flow = _flow(sample_steps=3).eval()
    calls = []
    original = flow.velocity
    flow.velocity = lambda *a: calls.append(1) or original(*a)
    flow.predict(torch.rand(2, 3, 64, 64))
    assert len(calls) == 3
    calls.clear()
    flow.predict(torch.rand(2, 3, 64, 64), steps=1)
    assert len(calls) == 1


def test_reflow_flags_parse():
    cfg = parse_args(["--path_to_datasets=x", "--reflow_teacher=t.pth", "--reflow_steps=10",
                      "--sample_steps=2"])
    assert (cfg.flow.reflow_teacher, cfg.flow.reflow_steps, cfg.flow.sample_steps) == ("t.pth", 10, 2)
