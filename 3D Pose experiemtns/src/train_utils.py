import torch
import inspect
from tqdm import tqdm
from pathlib import Path
from src.evaluation_metrics import calculate_evaluation_metrics
import numpy as np

def rotation_matrix_loss(pred, target, alpha=0.1, beta=0.01):
    mse = torch.mean((pred - target) ** 2)

    eye = torch.eye(3, device=pred.device).unsqueeze(0)
    ortho = torch.mean((pred.transpose(1, 2) @ pred - eye) ** 2)

    return mse + alpha * ortho


def project_to_rotation_matrix(x: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(x)
    r = u @ vh

    det = torch.det(r)

    correction = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
    correction[:, -1, -1] = torch.where(
        det < 0,
        torch.tensor(-1.0, device=x.device, dtype=x.dtype),
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
    )

    return u @ correction @ vh


def geodesic_rotation_matrix_loss(pred, target, alpha=0.01, beta=0.001):
    pred_rot = project_to_rotation_matrix(pred)

    relative = pred_rot.transpose(1, 2) @ target

    trace = relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = cos_theta.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    angle_loss = torch.acos(cos_theta).mean()

    eye = torch.eye(3, device=pred.device, dtype=pred.dtype).unsqueeze(0)
    ortho = torch.mean((pred.transpose(1, 2) @ pred - eye) ** 2)

    det_loss = torch.mean((torch.det(pred) - 1.0) ** 2)

    return angle_loss + alpha * ortho + beta * det_loss

def matrix_to_unit_quaternion(rot: torch.Tensor) -> torch.Tensor:
    if rot.ndim != 3 or rot.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with shape [B, 3, 3], got {tuple(rot.shape)}")

    m00 = rot[:, 0, 0]
    m01 = rot[:, 0, 1]
    m02 = rot[:, 0, 2]
    m10 = rot[:, 1, 0]
    m11 = rot[:, 1, 1]
    m12 = rot[:, 1, 2]
    m20 = rot[:, 2, 0]
    m21 = rot[:, 2, 1]
    m22 = rot[:, 2, 2]
    trace = m00 + m11 + m22

    q = torch.zeros((rot.shape[0], 4), device=rot.device, dtype=rot.dtype)
    eps = 1e-8

    mask_trace = trace > 0.0
    if mask_trace.any():
        t = torch.sqrt((trace[mask_trace] + 1.0).clamp_min(eps)) * 2.0
        q[mask_trace, 0] = 0.25 * t
        q[mask_trace, 1] = (m21[mask_trace] - m12[mask_trace]) / t
        q[mask_trace, 2] = (m02[mask_trace] - m20[mask_trace]) / t
        q[mask_trace, 3] = (m10[mask_trace] - m01[mask_trace]) / t

    mask_m00 = (~mask_trace) & (m00 > m11) & (m00 > m22)
    if mask_m00.any():
        t = torch.sqrt((1.0 + m00[mask_m00] - m11[mask_m00] - m22[mask_m00]).clamp_min(eps)) * 2.0
        q[mask_m00, 0] = (m21[mask_m00] - m12[mask_m00]) / t
        q[mask_m00, 1] = 0.25 * t
        q[mask_m00, 2] = (m01[mask_m00] + m10[mask_m00]) / t
        q[mask_m00, 3] = (m02[mask_m00] + m20[mask_m00]) / t

    mask_m11 = (~mask_trace) & (~mask_m00) & (m11 > m22)
    if mask_m11.any():
        t = torch.sqrt((1.0 + m11[mask_m11] - m00[mask_m11] - m22[mask_m11]).clamp_min(eps)) * 2.0
        q[mask_m11, 0] = (m02[mask_m11] - m20[mask_m11]) / t
        q[mask_m11, 1] = (m01[mask_m11] + m10[mask_m11]) / t
        q[mask_m11, 2] = 0.25 * t
        q[mask_m11, 3] = (m12[mask_m11] + m21[mask_m11]) / t

    mask_m22 = (~mask_trace) & (~mask_m00) & (~mask_m11)
    if mask_m22.any():
        t = torch.sqrt((1.0 + m22[mask_m22] - m00[mask_m22] - m11[mask_m22]).clamp_min(eps)) * 2.0
        q[mask_m22, 0] = (m10[mask_m22] - m01[mask_m22]) / t
        q[mask_m22, 1] = (m02[mask_m22] + m20[mask_m22]) / t
        q[mask_m22, 2] = (m12[mask_m22] + m21[mask_m22]) / t
        q[mask_m22, 3] = 0.25 * t

    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return q


def rotor_loss(pred: torch.Tensor, target_rot: torch.Tensor) -> torch.Tensor:
    target = matrix_to_unit_quaternion(target_rot)
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    dot = torch.sum(pred * target, dim=-1).abs()
    loss = 1.0 - dot.pow(2)
    return loss.mean()


def multivector_rotor_loss(
    pred_mv,
    target_rot,
    lambda_non_even=0.0,
    lambda_norm=0.0,
):
    target = matrix_to_unit_quaternion(target_rot)

    pred = pred_mv[:, [0, 4, 5, 6]]
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    dot = torch.sum(pred * target, dim=-1).abs()
    return (1.0 - dot.pow(2)).mean()


def form_checkpoint(model, optimizer, scheduler, config):
    checkpoint = {
        "model": model.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(config),
    }
    path = Path(f"{config.run_name}.pth").resolve()
    torch.save(checkpoint, path.__str__())
    return path


def load_checkpoint(model, optimizer, scheduler, path, device):
    if path is None:
        return model, optimizer, scheduler

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    try:
        scheduler.load_state_dict(checkpoint["scheduler"])
    except:
        scheduler = scheduler
    return model, optimizer, scheduler


def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def get_currently_used_device(model):
    return next(model.parameters()).device


def get_available_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _supports_class_argument(method) -> bool:
    params = list(inspect.signature(method).parameters.values())
    return any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params) or len(params) >= 2


def _compute_loss(model, data, criterion, config):
    img = data["img"].to(config.device)
    targets = data["rot"].to(config.device)

    clas = None
    if "cls" in data:
        clas = data["cls"].to(config.device)

    if config.loss == "prob":
        if hasattr(model, "compute_loss") and callable(getattr(model, "compute_loss")):
            # Important: avoid a duplicate forward pass before model.compute_loss(),
            # which can skew BatchNorm running statistics during training.
            return model.compute_loss(img, targets, criterion)

    if clas is not None and _supports_class_argument(model.forward):
        outputs = model(img, clas)
    else:
        outputs = model(img)

    if config.loss == "prob":
        idx = model.get_nearest_idx(targets).long().view(-1)
        return criterion(outputs, idx)
    return criterion(outputs, targets)


def train_epoch(model, loader, optimizer, criterion, config):
    total_loss = 0.0
    n_objects = 0
    device_type = "cuda" if config.device.type == "cuda" else "cpu"
    
    scaler = torch.amp.GradScaler(device_type) if device_type == "cuda" else None
    
    model.train()
    for data in tqdm(loader):
        optimizer.zero_grad(set_to_none=True)

        
        loss = _compute_loss(model, data, criterion, config)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = data["img"].shape[0]
        total_loss += float(loss.detach().item()) * bs
        n_objects += bs

    return total_loss / max(n_objects, 1)


@torch.no_grad()
def validate_epoch(model, loader, criterion, config):
    total_loss = 0.0
    n_objects = 0

    model.eval()
    for data in tqdm(loader):
        loss = _compute_loss(model, data, criterion, config)

        bs = data["img"].shape[0]
        total_loss += float(loss.detach().item()) * bs
        n_objects += bs

    return total_loss / max(n_objects, 1)


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, run, config):

    for i in range(config.n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config)
        val_loss = validate_epoch(model, val_loader, criterion, config)
        mre =  np.median(calculate_evaluation_metrics(model, val_loader, config)).__float__()

        if run is not None:
            run.log({
                "train_loss" : train_loss,
                "val_loss" : val_loss,
                "val_rot_err_deg" : mre,
                "learning_rate" : scheduler.get_last_lr()[0],
                "gradient_norm" : grad_norm(model)
            })
        scheduler.step()
        print(
            f"Training on {config.device} epoch {i + 1} / {config.n_epochs}. "
            f"Train loss {train_loss}, val loss {val_loss}"
            f"Median rotation error {mre}"
        )
