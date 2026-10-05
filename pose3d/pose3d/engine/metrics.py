"""Pose-error metrics (median rotation error in degrees, Acc@theta) and model-output decoding."""

import inspect

import numpy as np
import torch
from image2sphere.so3_utils import rotation_error
from tqdm import tqdm

from pose3d.engine.distributed import gather_errors, is_main
from pose3d.geometry.quaternion import project_multivector_to_rotor, unit_quaternion_to_matrix


def project_to_orthogonal_manifold(x):
    '''
    Finds nearest rotation matrix for a given one using SVD decomposition

    :param x: (B, 3, 3)
    returns : (B, 3, 3)
    '''
    u, _, v = torch.svd(x)
    determinants = torch.det(u @ v)
    correction = torch.eye(3, device=x.device).repeat(len(x), 1, 1)
    correction[:, -1, -1] = determinants
    return u @ correction @ torch.transpose(v, -1, -2)


def acc_at(err, theta=15):
    '''
    A function to compute acc@15
    
    :param theta: threshold for angle, degrees 
    '''
    if isinstance(err, torch.Tensor):
        return (err < theta).float().mean().item()
    err = np.asarray(err)
    return float((err < theta).mean())


def per_class_median(err, cls):
    """(mean of the per-class median errors, {class index: median}).

    The Pascal3D+ tables of IPDF / Image2Sphere / Rotation Laplace report this mean over
    the 12 classes, not one median over all test images, so hard classes (boat, bicycle)
    weigh as much as easy ones (bus, car).
    """
    err, cls = np.asarray(err), np.asarray(cls).reshape(-1)
    medians = {int(c): float(np.median(err[cls == c])) for c in np.unique(cls)}
    return float(np.mean(list(medians.values()))), medians


def per_class_acc(err, cls, theta=15):
    """{class index: fraction of that class's samples with error below `theta` degrees}."""
    err, cls = np.asarray(err), np.asarray(cls).reshape(-1)
    return {int(c): float((err[cls == c] < theta).mean()) for c in np.unique(cls)}


def macro_metrics(err, cls):
    """Class-averaged (macro) counterparts of the pooled (micro) median and Acc@theta.

    Every class weighs the same, so the small, hard ones (boat, bicycle, diningtable) count as
    much as the large ones (car, chair). `class_mean_median_error` is the published Pascal3D+
    number (see per_class_median).
    """
    mean_median, medians = per_class_median(err, cls)
    acc15, acc30 = per_class_acc(err, cls, 15), per_class_acc(err, cls, 30)
    return {
        "class_mean_median_error": mean_median,
        "class_mean_acc@15": float(np.mean(list(acc15.values()))),
        "class_mean_acc@30": float(np.mean(list(acc30.values()))),
        "class_medians": medians, "class_acc@15": acc15, "class_acc@30": acc30,
    }


def rotation_error_with_projection(input, target):
    input = project_to_orthogonal_manifold(input)
    target = project_to_orthogonal_manifold(target)
    err = rotation_error(input, target) / torch.pi * 180
    return err.cpu().numpy()


def _supports_class_argument(method) -> bool:
    # Only positional parameters can receive the class tensor; keyword-only
    # options such as n_samples must not be mistaken for one.
    positional = [
        p for p in inspect.signature(method).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return any(p.kind == inspect.Parameter.VAR_POSITIONAL
               for p in inspect.signature(method).parameters.values()) or len(positional) >= 2


def _sampling_kwargs(method, n_samples: int) -> dict:
    if n_samples > 1 and "n_samples" in inspect.signature(method).parameters:
        return {"n_samples": n_samples}
    return {}


def decode_output(model, outputs, cfg):
    """Turn the forward output of a model without `predict` into rotation matrices.

    Which output convention a model uses is selected by the training loss.
    """
    loss = cfg.train.loss
    if loss == "prob" and hasattr(model, "so3_rotmats_cache"):
        return model.so3_rotmats_cache[torch.argmax(outputs, dim=-1)]
    if loss == "rotor":
        return unit_quaternion_to_matrix(outputs)
    if loss == "mv_rotor":
        return unit_quaternion_to_matrix(project_multivector_to_rotor(outputs))
    return outputs


@torch.no_grad()
def calculate_evaluation_metrics(model, loader, cfg, n_samples: int = 1, return_classes: bool = False):
    """Rotation error (degrees) of every sample in `loader`.

    With return_classes, returns (errors, class indices) instead, the classes in the same
    order (None when the loader has neither "cls" nor "cls_eval", the labels the cached
    validation set gets from the Pascal3D+ annotations).

    Models exposing `predict` are evaluated through it (with `n_samples` draws when it
    accepts them); the rest go through `forward` plus `decode_output`. Under DDP the loader holds
    this rank's shard and the errors of all ranks are joined, so every rank must call it.
    """
    if getattr(cfg.features, "fixed_val_noise", False):
        # Same noise every evaluation: seed the RNGs (per rank, the loader is sharded) and
        # restore their state afterwards so training draws are not affected.
        devices = [cfg.device] if cfg.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(1234 + cfg.rank)
            return _evaluation_errors(model, loader, cfg, n_samples, return_classes)
    return _evaluation_errors(model, loader, cfg, n_samples, return_classes)


def _evaluation_errors(model, loader, cfg, n_samples, return_classes):
    device = cfg.device
    err, classes = [], []

    model.eval()
    model.to(device)
    for batch in tqdm(loader, desc="Evaluating Model", disable=not is_main() or cfg.run.platform == "kaggle"):
        img = batch["img"].to(device)

        clas = None
        if "cls" in batch:
            clas = batch["cls"].to(device)
            classes.append(batch["cls"].view(-1).cpu().numpy())
        elif "cls_eval" in batch:
            # Labels for the per-class metrics only; the model never sees them.
            classes.append(batch["cls_eval"].view(-1).cpu().numpy())

        if hasattr(model, "predict") and callable(getattr(model, "predict")):
            kwargs = _sampling_kwargs(model.predict, n_samples)
            if clas is not None and _supports_class_argument(model.predict):
                pred_rotmat = model.predict(img, clas, **kwargs)
            else:
                pred_rotmat = model.predict(img, **kwargs)
        else:
            if clas is not None and _supports_class_argument(model.forward):
                outputs = model(img, clas)
            else:
                outputs = model(img)
            pred_rotmat = decode_output(model, outputs, cfg)

        gt_rotmat = batch['rot'].to(device)
        err.append(rotation_error_with_projection(pred_rotmat, gt_rotmat))
    err = gather_errors(np.hstack(err))
    if not return_classes:
        return err
    # gather_errors joins ranks in the same order for both arrays, so they stay paired.
    return err, (gather_errors(np.hstack(classes)) if classes else None)
