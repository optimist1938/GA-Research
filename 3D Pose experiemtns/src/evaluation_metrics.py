import torch
from image2sphere.so3_utils import rotation_error
from image2sphere.predictor import I2S
from model import I2S as I2SFake
from tqdm import tqdm
import numpy as np


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
    return (err < theta).item() / len(err)


def rotation_error_with_projection(input, target):
    input = project_to_orthogonal_manifold(input)
    target = project_to_orthogonal_manifold(target)
    err = rotation_error(input, target) / torch.pi * 180
    return err.cpu().numpy()


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
        
        if hasattr(model, "predict") and callable(getattr(model, "predict")):
            if clas is not None:
                pred_rotmat = model.predict(img, clas)
            else:
                pred_rotmat = model.predict(img)
        else:
            pred_rotmat = model(img)

        gt_rotmat = batch['rot'].to(device)
        err.append(rotation_error_with_projection(pred_rotmat, gt_rotmat))
    return np.hstack(err)
