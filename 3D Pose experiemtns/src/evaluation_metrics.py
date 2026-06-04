import torch
import inspect
from image2sphere.so3_utils import rotation_error
from image2sphere.predictor import I2S
from model import I2S as I2SFake
from tqdm import tqdm
import numpy as np


def unit_quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    if q.ndim != 2 or q.shape[-1] != 4:
        raise ValueError(f"Expected unit quaternion shape [B, 4], got {tuple(q.shape)}")
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            torch.stack((ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)), dim=-1),
            torch.stack((2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)), dim=-1),
            torch.stack((2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz), dim=-1),
        ],
        dim=-2,
    )


def project_multivector_to_rotor(mv: torch.Tensor) -> torch.Tensor:
    if mv.shape[-1] != 8:
        raise ValueError(
            "project_multivector_to_rotor currently supports only Cl(3,0), "
            f"expected mv_dim=8, got {mv.shape[-1]}"
        )
    rotor = mv[:, [0, 4, 5, 6]]
    rotor = rotor / rotor.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return rotor


def create_technical_matrices(config):
    global I
    I = torch.eye(3)
    I = torch.stack([I for _ in range(config.batch_size)])
    I = I.to(config.device)
    return I


def project_to_orthogonal_manifold(x):
    '''
    Finds nearest rotation matrix for a given one using SVD decomposition
    
    :param x: (B, 3, 3)
    returns : (B, 3, 3)
    '''
    N = len(x)
    u, _, v = torch.svd(x)
    determinants = torch.det(u @ v)
    I[:N, -1, -1] = determinants
    return u @ I[:N] @ torch.transpose(v, -1, -2)


def acc_at(err, theta=15):
    '''
    A function to compute acc@15
    
    :param theta: threshold for angle, degrees 
    '''
    if isinstance(err, torch.Tensor):
        return (err < theta).float().mean().item()
    err = np.asarray(err)
    return float((err < theta).mean())


def rotation_error_with_projection(input, target):
    input = project_to_orthogonal_manifold(input)
    target = project_to_orthogonal_manifold(target)
    err = rotation_error(input, target) / torch.pi * 180
    return err.cpu().numpy()


def _supports_class_argument(method) -> bool:
    params = list(inspect.signature(method).parameters.values())
    return any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params) or len(params) >= 2


@torch.no_grad()
def calculate_evaluation_metrics(model, loader, config):
    device = config.device
    err = []

    model.eval()
    model.to(device)
    for batch in tqdm(loader, desc="Evaluating Model"):
        img = batch["img"].to(device)

        clas = None
        if "cls" in batch:
            clas = batch["cls"].to(device)
        
        if clas is not None and _supports_class_argument(unwrapped_model.forward):
            outputs = model(img, clas)
        else:
            outputs = model(img)

        if config.loss == "prob" and hasattr(unwrapped_model, "so3_rotmats_cache"):
            idx = torch.argmax(outputs, dim=-1)
            pred_rotmat = unwrapped_model.so3_rotmats_cache[idx]
        elif config.loss == "rotor":
            pred_rotmat = unit_quaternion_to_matrix(outputs)
        elif config.loss == "mv_rotor":
            rotor = project_multivector_to_rotor(outputs)
            pred_rotmat = unit_quaternion_to_matrix(rotor)
        else:
            pred_rotmat = model(img)

        gt_rotmat = batch['rot'].to(device)
        err.append(rotation_error_with_projection(pred_rotmat, gt_rotmat))
    return np.hstack(err)
