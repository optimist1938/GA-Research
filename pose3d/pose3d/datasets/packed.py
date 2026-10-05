"""(Same file as packed_i2s.py in the Kaggle dataset syfry5suvzovvakmuj/pascal3d-synth-pack.)

RAM-resident Image2Sphere PASCAL3D+ training data: real images + RenderForCNN synthetic images, read from the packed files of this dataset.

Reproduces image2sphere's `Pascal3D(train=True, use_warp=..., use_synth=...)`: same camera maths (the functions are imported from
`image2sphere.pascal_dataset`), same augmentation, same epoch length (4 x len(real)), but every image comes from RAM instead of disk.

    from pose3d.datasets.packed import PackedPascal3D
    ds = PackedPascal3D('/kaggle/input/datasets/syfry5suvzovvakmuj/pascal3d-synth-pack', use_warp=True, use_synth=True)
    sample = ds[0]          # dict(img=[3,224,224] float32 in 0..1, cls=[1] long, rot=[3,3] float32), like image2sphere

Synthetic labels: the renders come from ShapeNet v2 models, which face a different way than ShapeNet v1 (the models the PASCAL3D+ azimuth
convention is defined on). The pack therefore stores the file azimuth `a_file` and the corrected azimuth `a` = (a_file - 90) mod 360 for the
11 ShapeNet-v2 classes (bicycle: already correct). With relabel=True (default) the corrected one is used. `weight` is an importance weight
that restores the PASCAL3D+ viewpoint distribution after that shift; use_weights=True draws synthetic images with probability ~ weight.
"""
import glob
import io
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import image2sphere.pascal_dataset as _p
import skimage.transform

CLASS_NAMES = ('aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair',
               'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor')
SYN_CLASS_IDS = ('02691156', '02834778', '02858304', '02876657', '02924116', '02958343',
                 '03001627', '04379243', '03790512', '04256520', '04468005', '03211117')


def _read_bin(paths):
    """All files into one flat uint8 tensor (one big buffer: forked DataLoader workers share it without copying)."""
    sizes = [os.path.getsize(p) for p in paths]
    buf = torch.empty(sum(sizes), dtype=torch.uint8)
    arr = buf.numpy()
    pos = 0
    for p, s in zip(paths, sizes):
        with open(p, 'rb') as f:
            mv = memoryview(arr[pos:pos + s])
            got = 0
            while got < s:
                n = f.readinto(mv[got:])
                if not n:
                    raise IOError(f'short read {p}')
                got += n
        pos += s
    return buf, np.cumsum([0] + sizes)


def _decode(raw):
    """Same pixels as image2sphere (single-band files repeated to 3 channels), returned as uint8 HxWx3."""
    with Image.open(io.BytesIO(raw)) as im:
        if im.mode == '1':
            im = im.convert('L')
        arr = np.asarray(im)
    if arr.ndim == 2:
        arr = arr[:, :, None].repeat(3, 2)
    return arr


def _warp_to_224(img, shape_hw, bbox, principal_point, angle, cam, distance, img_size, flip, use_warp, scale_xy=(1.0, 1.0), backend='skimage'):
    """The part of Pascal3DReal/Pascal3DSynth.__getitem__ after the image is loaded.

    img: the image array actually held in RAM (HxWx3 uint8, possibly smaller than the original); shape_hw: the ORIGINAL (h, w), used for all
    camera maths; scale_xy: (w_stored / w_orig, h_stored / h_orig).
    """
    H, W = shape_hw
    bbox = list(map(float, bbox)); angle = np.array(angle, dtype=float); principal_point = np.array(principal_point, dtype=float)
    if flip:
        angle = angle * np.array([-1.0, 1.0, -1.0])
        img = img[:, ::-1, :]
        bbox[0] = W - bbox[0]
        bbox[2] = W - bbox[2]
        principal_point[0] = W - principal_point[0]
    if use_warp:
        desired_up = np.random.normal(0, 0.4, size=(3)) + np.array([3.0, 0.0, 0.0])
        desired_up[2] = 0
        desired_up /= np.linalg.norm(desired_up)
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox = np.array(bbox)
        bbox[0::2] += np.random.uniform(-bbox_w * 0.1, bbox_w * 0.1, size=(2))
        bbox[1::2] += np.random.uniform(-bbox_h * 0.1, bbox_h * 0.1, size=(2))
    else:
        desired_up = np.array([1.0, 0.0, 0.0])
    intrinsic, extrinsic = _p.get_camera_parameters(cam, principal_point, angle, (H, W), distance)
    back_proj_bbx = _p.get_back_proj_bbx(bbox, intrinsic)
    extrinsic_desired_change, intrinsic_new = _p.get_desired_camera(img_size, back_proj_bbx, desired_up)
    extrinsic_after = np.matmul(extrinsic_desired_change, extrinsic)
    P = np.matmul(np.matmul(intrinsic_new, extrinsic_desired_change[:3, :3]), np.linalg.inv(intrinsic))
    P /= P[2, 2]
    Pinv = np.linalg.inv(P)                       # output pixel -> ORIGINAL image pixel
    sx, sy = scale_xy
    if sx != 1.0 or sy != 1.0:                    # ... -> pixel of the stored (shrunk) image; pixel centres at integer coordinates
        S = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])
        Pinv = S @ Pinv
    im = img.astype(np.float32) / 255
    if backend == 'cv2':
        import cv2
        warped = cv2.warpPerspective(im, Pinv, (img_size, img_size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        warped = skimage.transform.warp(im, skimage.transform.ProjectiveTransform(Pinv), output_shape=(img_size, img_size), mode='constant', cval=0.0)
    return warped, torch.from_numpy(extrinsic_after[:3, :3]).to(torch.float32)


class PackedReal(Dataset):
    def __init__(self, root, img_size=224, use_warp=True, backend='skimage'):
        self.idx = np.load(f'{root}/real/real_train.idx.npy')
        self.buf, _ = _read_bin([f'{root}/real/real_train.bin'])
        self.img_size, self.use_warp, self.backend = img_size, use_warp, backend
        self.class_names = CLASS_NAMES

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.idx[i]
        raw = self.buf[int(r['off']):int(r['off']) + int(r['len'])].numpy().tobytes()
        img = _decode(raw)
        H, W = img.shape[:2]
        flip = np.random.randint(2) if self.use_warp else 0
        warped, rot = _warp_to_224(img, (H, W), r['bbox'], (r['px'], r['px']), (r['az'], r['el'], r['th']), np.array([r['focal'], r['viewport']]),
                                   r['dist'], self.img_size, flip, self.use_warp, backend=self.backend)
        return dict(img=torch.from_numpy(warped).permute(2, 0, 1).to(torch.float32), cls=torch.tensor((int(r['cls']),), dtype=torch.long), rot=rot)


class PackedSynth(Dataset):
    def __init__(self, root, img_size=224, relabel=True, use_weights=False, max_synth=0, seed=0, backend='skimage'):
        files = json.load(open(f'{root}/synth_shards.json'))
        self.idx = np.load(f'{root}/synth_index.npy')
        self.buf, self.shard_start = _read_bin([f'{root}/synth/{f}' for f in files])
        self.img_size, self.relabel, self.backend = img_size, relabel, backend
        self.sel = np.arange(len(self.idx))
        if 0 < max_synth < len(self.idx):
            self.sel = np.sort(np.random.default_rng(seed).choice(len(self.idx), size=max_synth, replace=False))
        self.cdf = None
        if use_weights:
            w = self.idx['weight'][self.sel].astype(np.float64)
            self.cdf = np.cumsum(w) / w.sum()
        self.class_names = CLASS_NAMES

    def __len__(self):
        return len(self.sel)

    def draw(self):
        """Index of a random synthetic image (uniform, or ~ importance weight)."""
        if self.cdf is None:
            return np.random.randint(len(self.sel))
        return min(int(np.searchsorted(self.cdf, np.random.random())), len(self.sel) - 1)

    def __getitem__(self, i):
        r = self.idx[self.sel[i]]
        start = int(self.shard_start[int(r['shard'])]) + int(r['off'])
        img = _decode(self.buf[start:start + int(r['len'])].numpy().tobytes())
        w0, h0 = int(r['w0']), int(r['h0'])
        a = int(r['a']) if self.relabel else int(r['a_file'])
        t = int(r['t'])
        e = int(r['e'])
        flip = np.random.randint(2)
        if flip:
            a, t = -a, -t
        # upstream: flip is applied to the angles here and to the image/bbox/principal point inside _warp_to_224 -> pass flip=0 there
        img_f = img[:, ::-1, :] if flip else img
        bbox = [0, 0, w0, h0]
        pp = np.array([w0 / 2, h0 / 2], dtype=np.float32)
        if flip:
            bbox[0] = w0 - bbox[0]
            bbox[2] = w0 - bbox[2]
            pp[0] = w0 - pp[0]
        warped, rot = _warp_to_224(img_f, (h0, w0), bbox, pp, (a, e, -t), 3000, 4.0, self.img_size, 0, True,
                                   scale_xy=(img.shape[1] / w0, img.shape[0] / h0), backend=self.backend)
        return dict(img=torch.from_numpy(warped).permute(2, 0, 1).to(torch.float32), cls=torch.tensor((int(r['cls']),), dtype=torch.long), rot=rot)


class PackedPascal3D(Dataset):
    """Drop-in for image2sphere.pascal_dataset.Pascal3D(train=True, use_warp, use_synth): len = 4 x real when use_synth."""

    def __init__(self, root, use_warp=True, use_synth=True, img_size=224, relabel=True, use_weights=False, max_synth=0, backend='skimage'):
        self.real_dataset = PackedReal(root, img_size, use_warp, backend)
        self.use_synth = use_synth
        self.synth_dataset = PackedSynth(root, img_size, relabel, use_weights, max_synth, backend=backend) if use_synth else None
        self.img_shape = (3, img_size, img_size)
        self.num_classes = 12
        self.class_names = CLASS_NAMES

    def __getitem__(self, idx):
        if idx < len(self.real_dataset):
            return self.real_dataset[idx]
        return self.synth_dataset[self.synth_dataset.draw()]

    def __len__(self):
        return 4 * len(self.real_dataset) if self.use_synth else len(self.real_dataset)
