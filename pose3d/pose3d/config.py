"""Experiment configuration: the single place that decides what a default run is.

The defaults reproduce the best run, `clifford_flow_gatr_warp_synth_ema_b64` (Kaggle kernel
`syfry5suvzovvakmuj/clifford-gatr-ema-b64-rtx`, one RTX Pro 6000): 8.95 deg mean of per-class
median rotation errors on Pascal3D+ (the published metric), 7.85 deg median over all test images.
`--model=clifford_flow`, pretrained ResNet-101, no ConvAdapter (the pooled backbone vector is
reshaped straight into 256 multivectors), a Clifford MLP condition head 88 wide
(`flow_hidden_dim=88`) feeding a GATr vector field (`vector_field=gatr`), `n_cond_mv=64`,
`n_time_samples=8`, trained on Image2Sphere's data (real images with `use_warp` augmentation +
RenderForCNN renders, `use_synth`), global batch 64 with lr 1e-4 scaled by sqrt(64/32), 100
epochs, EMA weights (`ema`), seeded validation noise (`fixed_val_noise`), 32-sample medoid
evaluation with 20 Euler steps. The training data is read from the Kaggle dataset
`syfry5suvzovvakmuj/pascal3d-synth-pack`, picked up automatically when mounted (SYNTH_PACK_DIRS);
without it, use_warp / use_synth fall back to the image files and need the RenderForCNN download.
DDP over every visible GPU and the pre-built Pascal3D tensors are on by default too; they speed
a run up and leave the recipe's hyperparameters alone.

The previous reference recipe, W&B `mnpsfhmd` (9.63 deg median over all test images), is the same
model with the Clifford MLP vector field on real images only:
`--vector_field clifford --no-use_warp --no-use_synth --no-ema --no-fixed_val_noise
--batch_size 32 --lr_scaling none`. (Before it: `6te3pvqa`, 9.46 deg with the ConvAdapter,
`--conv_adapter --flow_hidden_dim 32`; and `k5sblpo8`, 10.25 deg with ResNet-50.)

Experiment workflow (see the root README and the idea board in pose3d/README.md):

* A change that is proven better on Pascal3D+ is adopted with its flag set to True in
  `Features` below (or, for numeric options, its value becomes the default).
* A change that is not proven yet lands with its flag False, so `main` stays the
  reference recipe and the experiment stays one `--flag` away.
* A change that turned out worse is not added; it is written up in a report.

Every field below is also a command-line flag of the same name (`--use_warp`,
`--no-medoid_eval`, `--n_cond_mv 32`, ...). Sections only group them; flag names are
globally unique.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Union, get_args, get_origin, get_type_hints

ModelName = Literal[
    "clifford_flow",   # the main line: conditional flow matching on Spin(3) rotors
    "mlp_flow",        # non-equivariant flow-matching baseline
    "i2s_real",        # published Image2Sphere, adapted to this harness
    "i2s",             # in-repo I2S with a GA head (average-pooled features)
    "ga_i2s",          # in-repo I2S with the GA pixel encoder
    "tralalero",       # direct regression with a GA head
    "mlp",             # direct regression with a linear head
    "i2s_backbone",    # ResNet/ConvNeXt + conv adapter + GA head (teammates' line)
    "i2s_resnet",      # alias of i2s_backbone, kept for old command lines
    "i2s_conv_resnet",   # conv_adapter branch: adapters + ResNet-50
    "i2s_conv_convnext",  # conv_adapter branch: adapters + ConvNeXt
    "ipdf_resnet",     # Clifford implicit PDF over rotor-rotated features
    "vit_baseline",    # ViT / Depth-Anything hidden states + pooling head
    "image2pcd",       # Depth-Anything features, direct regression (I2P)
    "image2pcd_pointnet",      # Depth-Anything point cloud + PointNet regression (I2PPointNet)
    "image2pcd_late_fusion",   # Depth-Anything hidden states + point stream, late fusion (I2PLateFusion)
    "image2pcd_ipdf",  # Depth-Anything features, implicit PDF (I2P_IPDF)
    "dummynet",        # point-cloud debugging model (ModelNet10 stub)
]
EncoderName = Literal[
    "resnet", "resnet50", "resnet101", "depth_anything", "ga", "ga_canonical",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
]
LossName = Literal["mse", "mse_ortho", "geodesic", "prob", "rotor", "mv_rotor"]


@dataclass
class RunConfig:
    """Where the data is, how the run is named and logged."""

    path_to_datasets: str = field(default="", metadata={"required": True})
    dataset: Literal["pascal", "dummynet", "modelnet10", "symsol"] = "pascal"
    platform: Literal["kaggle", "colab"] = "kaggle"
    run_name: Optional[str] = None
    path_to_checkpoint: Optional[str] = None   # evaluate this checkpoint before training
    sanity_check: bool = False                 # one tiny batch; no checkpoint saved
    save_checkpoint: bool = True
    wandb_project: str = "3D Pose Estimation"
    wandb_entity: str = "clifforders"
    wandb_group: Optional[str] = None
    # Base seed for the run. None: torch's default. Under --ddp every rank adds its rank,
    # so the flow's (t, r0) noise differs between GPUs.
    seed: Optional[int] = None


@dataclass
class TrainConfig:
    n_epochs: int = 100
    warmup_epochs: int = 5
    # GLOBAL batch: under --ddp each GPU sees batch_size // world_size, so the same value is
    # the same recipe on 1, 2 or 4 GPUs. 64 is the 8.95 deg recipe (32 before it).
    batch_size: int = 64
    lr: float = 1e-4
    # Rescale lr by (batch_size / lr_reference_batch): "linear" or "sqrt". "none" keeps lr
    # as given. The 8.95 deg recipe uses sqrt: 1e-4 * sqrt(64 / 32) = 1.41e-4.
    lr_scaling: Literal["none", "linear", "sqrt"] = "sqrt"
    lr_reference_batch: int = 32
    # mse: plain MSE on the matrix; mse_ortho: MSE + orthogonality penalty (teammates' default).
    loss: LossName = "mse"
    label_smoothing: float = 0.0   # only for loss=prob
    # Samples per image for the multi-sample evaluation run after the last epoch
    # (used when Features.medoid_eval is on).
    eval_samples: int = 32
    # Per-optimizer-step decay of Features.ema (0.999: ~1000-step horizon).
    ema_decay: float = 0.999


@dataclass
class Features:
    """Switches for modifications of the reference pipeline.

    True  = adopted: measurably better, part of the reference recipe.
    False = experimental: not proven yet, or an ablation kept for comparison.
    """

    # ---- adopted (True) ---------------------------------------------------------
    # ImageNet-initialised, fully fine-tuned backbone (scratch: 56.8 deg vs 10.25 deg).
    pretrained_backbone: bool = True
    # Preload Pascal3D tensors into RAM (infrastructure; same numbers, faster epochs).
    ram_memory: bool = True
    # Draw FlowConfig.n_time_samples (t, r0) pairs per image per step, sharing one
    # backbone pass (8 samples/image is part of the 10.25 deg recipe).
    time_sample_batching: bool = True
    # Score the final model with TrainConfig.eval_samples draws and take the geodesic
    # medoid instead of a single draw (32 samples is what the reported numbers use).
    medoid_eval: bool = True
    # Data-parallel training over every visible GPU (torch DistributedDataParallel; the run
    # relaunches itself under torchrun). With one GPU or CPU it changes nothing. batch_size stays
    # the global batch; BatchNorm statistics are per GPU unless --sync_bn. On Kaggle T4 x2 an
    # epoch took roughly 1.5-1.6x less than on one T4 (compared across runs). --no-ddp turns it off.
    ddp: bool = True
    # Reuse the pre-built Pascal3D tensors instead of decoding every image each session
    # (~34 min): when the Kaggle dataset `syfry5suvzovvakmuj/pascal3d-ram-cache` is mounted,
    # --ram_cache_dir is picked automatically (PRE_CACHE_DIRS). The tensors come from the same
    # InMemoryDataset build a normal run does (not bit-compared with a fresh build). With
    # nothing mounted, or with use_warp / use_synth / raw_cache / fisher_prior, the run builds
    # them as before. --no-pre_cache turns it off.
    pre_cache: bool = True
    # The next four were adopted together with vector_field=gatr and batch 64 (8.95 deg, see the
    # module docstring); their separate contributions are not measured.
    # Evaluate (every epoch and at the end) and save an exponential moving average of the
    # weights (TrainConfig.ema_decay; BatchNorm statistics averaged too) instead of the last
    # iterate. The final evaluation also scores the last iterate as final_*_raw.
    ema: bool = True
    # Draw the flow's validation noise from a fixed seed, so every evaluation of the same
    # weights gives the same numbers and epoch-to-epoch changes come from the weights only.
    # Training randomness is untouched (the generator state is restored afterwards).
    fixed_val_noise: bool = True
    # Pascal3D's own augmentation (flip / up-direction jitter / bbox jitter), as Image2Sphere.
    use_warp: bool = True
    # RenderForCNN synthetic training images, 3 per real image per epoch, as Image2Sphere
    # (from the synth pack, DataConfig.synth_pack_dir; otherwise a separate download, see max_synth).
    use_synth: bool = True

    # ---- experimental (False) ---------------------------------------------------
    # With ram_memory, cache the file reads instead of augmented crops so use_warp stays
    # random per access.
    raw_cache: bool = False
    # Freeze the backbone and train only the heads.
    freeze_encoder: bool = False
    # Swap CliffordFlow's two GA heads for parameter-matched plain MLPs (43.7 deg: a
    # comparison showing the GA layers matter, not a candidate improvement).
    mlp_heads: bool = False
    # Draw the flow's source rotor r0 from a pretrained matrix Fisher head instead of the
    # uniform distribution (needs FlowConfig.fisher_checkpoint). Unfinished run only.
    fisher_prior: bool = False


@dataclass
class DataConfig:
    multiprocessing: bool = False   # spawn workers while filling the RAM cache
    max_synth: int = 0              # cap on the synthetic pool with use_synth (0 keeps all)
    cache_draws: int = 1            # augmented passes cached with ram_memory + use_warp
    # DataLoader workers per process. None: 2 (4 with raw_cache) split across GPUs, at least 1.
    num_workers: Optional[int] = None
    # Reuse the tensors ram_memory builds instead of decoding every image each session.
    ram_cache_dir: Optional[str] = None        # read pascal_{train,val}.pt from here
    ram_cache_save_dir: Optional[str] = None   # write them here after a normal build
    # Image2Sphere's training data held in RAM (Kaggle dataset `syfry5suvzovvakmuj/pascal3d-synth-pack`:
    # the real train images + 2.38M re-rendered RenderForCNN images, see datasets/packed.py). With
    # use_warp / use_synth the train set is read from here and augmented per access; picked
    # automatically when mounted (SYNTH_PACK_DIRS).
    synth_pack_dir: Optional[str] = None
    # Draw synthetic images with the pack's importance weights (restores the PASCAL3D+ viewpoint
    # distribution after the ShapeNet-v2 azimuth relabel). Off: uniform draws, as Image2Sphere.
    synth_pack_weights: bool = False
    # --dataset symsol: which shape subset (image2sphere.dataset.SymsolDataset class_names).
    # 1: the standard 5-shape benchmark (tet, cube, icosa, cone, cyl). 2/3/4: the single-shape
    # near-symmetric variants (sphereX/cylO/tetX).
    symsol_set: int = 1


@dataclass
class DistConfig:
    """Options for Features.ddp."""

    num_gpus: Optional[int] = None   # None: every visible GPU
    sync_bn: bool = False            # SyncBatchNorm instead of per-GPU BatchNorm statistics
    ddp_find_unused: bool = False    # only for models with parameters that get no gradient
    # NCCL peer-to-peer copies; off by default because Kaggle's T4 x2 can hang with it on.
    nccl_p2p: bool = False


@dataclass
class ModelConfig:
    name: ModelName = field(default="clifford_flow", metadata={"flag": "model"})
    encoder: EncoderName = "resnet101"  # "resnet" is an alias of resnet50; 6te3pvqa (9.46 deg) used resnet101
    # GA head widths of the other models; CliffordFlow uses FlowConfig.flow_hidden_dim.
    hidden_dim: List[int] = field(default_factory=lambda: [32])
    algebra_dim: int = 3                # Cl(algebra_dim); most GA paths need 3
    depth_anything_model: str = "depth-anything/Depth-Anything-V2-Base-hf"


@dataclass
class FlowConfig:
    """CliffordFlow options. Defaults are the reference recipe; the rest are ablations."""

    n_cond_mv: int = 64        # conditioning multivectors passed to the vector field
    n_time_samples: int = 8    # (t, r0) pairs per image (see Features.time_sample_batching)
    # Hidden widths of both GA heads (condition head and vector field). 88 refills the
    # parameter budget freed by dropping the ConvAdapter (1,292,217 non-backbone params).
    flow_hidden_dim: List[int] = field(default_factory=lambda: [88])
    # ConvAdapter pools to (adapter_grid x adapter_grid) multivectors. 9 -> 10.90 deg,
    # 7 -> 11.95 deg, 11 -> 11.31 deg (n=1 each, before the recipe fix; unconfirmed).
    adapter_grid: int = 16
    # Width of the adapter's first 1x1 conv. 96 -> 10.92 deg (unconfirmed).
    adapter_channels: int = 256
    # Hidden widths of the vector field alone (None: same as flow_hidden_dim). Paired with a
    # smaller adapter_grid this reallocates parameters; 10.49 deg (unconfirmed).
    vector_field_hidden_dim: Optional[List[int]] = None
    # Off: no ConvAdapter; the globally pooled backbone vector is reshaped into
    # backbone_channels / 8 multivectors (256 for ResNet) for the condition head, and
    # adapter_grid / adapter_channels are unused. 9.63 deg (mnpsfhmd) vs 9.46 with it
    # (6te3pvqa), n=1 each. --conv_adapter brings the adapter back.
    conv_adapter: bool = False
    # Denoiser network of the flow. "gatr" (reference recipe, 8.95 deg): the Geometric Algebra
    # Transformer (Brehmer et al. 2023) over the rotor, time and condition multivectors as tokens;
    # the condition head stays a Clifford MLP. Needs the GATr package (see models/gatr_denoiser.py).
    # "clifford": the CGENN-style Clifford MLP (the 9.63 deg recipe); the gatr_* options are
    # unused then.
    vector_field: Literal["clifford", "gatr"] = "gatr"
    # Same choice for the condition head (backbone multivectors -> n_cond_mv condition
    # multivectors): "gatr" runs GATr over the backbone tokens plus n_cond_mv learned queries.
    # Shares the gatr_* sizes with the vector field. Not with mlp_heads or fisher_prior.
    condition_head: Literal["clifford", "gatr"] = "clifford"
    # How the backbone map becomes the condition head's 256 input multivectors. "pooled": global
    # average pool, then 8 consecutive channels are declared one multivector (the 8.95 deg
    # recipe), so the "vectors" do not rotate with the image. "so2": keep the 7x7 map, feature
    # channels stay scalars and the vector / bivector parts are feature-weighted sums of each
    # cell's in-plane direction from the image centre, plus a constant optical-axis (e3) token:
    # rotating the image by 90 deg rotates every token about e3 exactly (models/so2_head.py).
    # Without conv_adapter, fisher_prior. so2_channels = width of its first 1x1 conv.
    cond_tokens: Literal["pooled", "so2"] = "pooled"
    so2_channels: int = 128
    # How the GATr vector field sees the current pose. "rotor": one rotor token, which GATr's
    # sandwich action conjugates (R -> G R G^T), unlike a camera rotation (R -> G R).
    # "frame": three vector tokens R e1, R e2, R e3, with the velocity read out in the camera
    # frame, so GATr's symmetry is the physical one. Pairs with cond_tokens=so2; GATr only.
    pose_tokens: Literal["rotor", "frame"] = "rotor"
    gatr_blocks: int = 4
    gatr_mv_channels: int = 8    # hidden multivector channels per token
    gatr_s_channels: int = 32    # hidden scalar channels per token
    gatr_heads: int = 4
    # Path to Liu et al.'s Pascal3D+ matrix Fisher checkpoint (state_dict_119.pkl);
    # used only with Features.fisher_prior.
    fisher_checkpoint: Optional[str] = None


@dataclass
class I2SConfig:
    """Image2Sphere-family options (i2s, i2s_real, ga_i2s and the I2S-style heads)."""

    lmax: int = 6
    rec_level: int = 3
    n_mv: int = 8
    ga_pool_hw: Tuple[int, int] = (28, 28)
    temperature: float = 1.0
    # i2s_real: eval grid level (upstream's 5 needs a 4.3 GB Wigner matrix).
    i2s_eval_rec_level: int = 3
    # i2s_real: apply ImageNet normalisation to the [0, 1] inputs.
    i2s_normalize: bool = True


@dataclass
class IPDFConfig:
    """Implicit-PDF options (image2pcd_ipdf, ipdf_resnet)."""

    pe_freqs: int = 4
    n_train_queries: int = 4096
    grad_ascent_steps: int = 100
    grad_ascent_lr: float = 1e-4
    # image2pcd / image2pcd_ipdf: freeze the Depth-Anything backbone (distinct from
    # Features.freeze_encoder, which applies to the flow/I2S encoders).
    freeze_backbone: bool = True
    ipdf_n_queries: int = 511   # ipdf_resnet: negative rotors per image


@dataclass
class PointCloudConfig:
    """Depth-Anything point-cloud regressors (image2pcd_pointnet, image2pcd_late_fusion)."""

    i2p_n_points: int = 2048
    # image2pcd_late_fusion: False zeroes the depth channel (the depth ablation, 10.12 deg,
    # beat the run with depth, 10.74 deg).
    i2p_use_depth: bool = True


@dataclass
class ViTConfig:
    """`--model vit_baseline`."""

    vit_backbone_type: Literal["vit", "depth_anything_v2"] = "vit"
    vit_model_name: str = "google/vit-base-patch16-224-in21k"
    vit_layers: List[int] = field(default_factory=lambda: [-1, -3, -6, -9])
    freeze_vit: bool = True
    vit_pooling_type: Literal["mean", "attention", "transformer_attention", "convolution", "ga"] = "mean"
    vit_num_transformer_layers: int = 1
    vit_transformer_nhead: int = 8
    vit_transformer_ff_dim: int = 1024
    vit_transformer_dropout: float = 0.1
    vit_ga_input_features: int = 196
    vit_ga_hidden_dim: List[int] = field(default_factory=lambda: [32])
    vit_ga_readout_type: Literal["scalar", "mean", "linear", "grade", "rotor"] = "linear"


@dataclass
class I2SBackboneConfig:
    """`--model i2s_backbone` (alias `i2s_resnet`); `ipdf_resnet` reuses the backbone flags."""

    i2s_resnet_output_mode: Literal[
        "auto", "rotation_matrix", "fourier", "rotor", "multivector_rotor"] = "auto"
    i2s_resnet_backbone_name: Literal["resnet50", "convnext_tiny"] = "resnet50"
    i2s_resnet_pretrained_backbone: bool = True
    i2s_resnet_freeze_backbone: bool = True
    i2s_resnet_use_positional_encoding: bool = True
    i2s_resnet_mv_per_position: int = 1
    i2s_resnet_adapter_mid_channels: int = 0
    i2s_resnet_adapter_high_channels: int = 0
    i2s_resnet_adapter_output_size: int = 16
    i2s_resnet_ga_head_type: Literal[
        "tralalero", "transformer_like", "reduced", "residual_gp"] = "tralalero"
    i2s_resnet_ga_head_mixing_layer: Literal["gp", "mvlinear", "linear"] = "gp"
    i2s_resnet_ga_num_blocks: int = 2
    i2s_resnet_ga_head_dropout: float = 0.0
    i2s_resnet_ga_head_use_layer_norm: bool = False


@dataclass
class I2SConvConfig:
    """`--model i2s_conv_resnet | i2s_conv_convnext` (the conv_adapter branch)."""

    i2s_conv_variant: Literal["tiny", "small", "base", "large"] = "tiny"   # ConvNeXt only
    i2s_conv_output_mode: Literal["auto", "rotation_matrix", "fourier", "rotor", "vector_proj"] = "auto"
    i2s_conv_pretrained_backbone: bool = True
    i2s_conv_freeze_backbone: bool = True
    i2s_conv_use_positional_encoding: bool = True
    i2s_conv_adapter_type: Literal["conv", "mlp_block", "linear", "geometric", "inc"] = "conv"
    i2s_conv_head_type: Literal["ga", "mlp"] = "ga"


SECTIONS = (
    "run", "train", "features", "data", "distributed", "model", "flow",
    "i2s", "ipdf", "pointcloud", "vit", "i2s_backbone", "i2s_conv",
)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    features: Features = field(default_factory=Features)
    data: DataConfig = field(default_factory=DataConfig)
    distributed: DistConfig = field(default_factory=DistConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    i2s: I2SConfig = field(default_factory=I2SConfig)
    ipdf: IPDFConfig = field(default_factory=IPDFConfig)
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    vit: ViTConfig = field(default_factory=ViTConfig)
    i2s_backbone: I2SBackboneConfig = field(default_factory=I2SBackboneConfig)
    i2s_conv: I2SConvConfig = field(default_factory=I2SConvConfig)

    # Set at runtime, not a flag.
    device: Any = None
    rank: int = 0
    world_size: int = 1

    @property
    def eval_samples(self) -> int:
        return self.train.eval_samples if self.features.medoid_eval else 1

    @property
    def per_gpu_batch_size(self) -> int:
        return self.train.batch_size // self.world_size

    @property
    def effective_lr(self) -> float:
        """train.lr rescaled by the batch size when lr_scaling asks for it."""
        t = self.train
        ratio = t.batch_size / t.lr_reference_batch
        if t.lr_scaling == "linear":
            return t.lr * ratio
        if t.lr_scaling == "sqrt":
            return t.lr * ratio ** 0.5
        return t.lr

    @property
    def n_time_samples(self) -> int:
        return self.flow.n_time_samples if self.features.time_sample_batching else 1

    def to_dict(self) -> dict:
        """Nested plain-dict view (for checkpoints and W&B); excludes the runtime device."""
        return {name: dataclasses.asdict(getattr(self, name)) for name in SECTIONS}


def _flag(section_field: dataclasses.Field) -> str:
    return section_field.metadata.get("flag", section_field.name)


def _unwrap_optional(tp):
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _add_flag(parser: argparse.ArgumentParser, fld: dataclasses.Field, hints: dict):
    tp, _ = _unwrap_optional(hints[fld.name])
    name = "--" + _flag(fld)
    default = (fld.default_factory() if fld.default_factory is not dataclasses.MISSING
               else fld.default)
    kwargs: dict = {"default": default, "dest": _flag(fld)}
    if fld.metadata.get("required"):
        kwargs["required"] = True

    origin = get_origin(tp)
    if tp is bool:
        parser.add_argument(name, action=argparse.BooleanOptionalAction, **kwargs)
        return
    if origin is Literal:
        choices = get_args(tp)
        parser.add_argument(name, type=type(choices[0]), choices=choices, **kwargs)
        return
    if origin in (list, List, tuple, Tuple):
        inner = get_args(tp)[0]
        nargs = len(get_args(tp)) if origin in (tuple, Tuple) else "+"
        parser.add_argument(name, type=inner, nargs=nargs, **kwargs)
        return
    parser.add_argument(name, type=tp, **kwargs)


def _summary(cls) -> Optional[str]:
    """First docstring line, ignoring the signature string dataclass generates when there is none."""
    doc = cls.__doc__
    if not doc or doc.startswith(cls.__name__ + "("):
        return None
    return doc.strip().splitlines()[0]


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train / evaluate a 3D pose (SO(3)) estimator. Defaults are the reference recipe.")
    seen: set = set()
    for name, cls in _section_classes().items():
        hints = get_type_hints(cls)
        group = parser.add_argument_group(name, _summary(cls))
        for fld in dataclasses.fields(cls):
            flag = _flag(fld)
            if flag in seen:
                raise ValueError(f"duplicate flag --{flag}")
            seen.add(flag)
            _add_flag(group, fld, hints)
    return parser


def _section_classes() -> dict:
    hints = get_type_hints(Config)
    return {name: hints[name] for name in SECTIONS}


# Where Kaggle mounts the pre-built Pascal3D tensors (pascal_train.pt / pascal_val.pt) of the dataset
# `syfry5suvzovvakmuj/pascal3d-ram-cache`; the first directory holding both files is used.
PRE_CACHE_DIRS = (
    "/kaggle/input/pascal3d-ram-cache",
    "/kaggle/input/datasets/syfry5suvzovvakmuj/pascal3d-ram-cache",
)


# Where Kaggle mounts `syfry5suvzovvakmuj/pascal3d-synth-pack` (DataConfig.synth_pack_dir).
SYNTH_PACK_DIRS = (
    "/kaggle/input/pascal3d-synth-pack",
    "/kaggle/input/datasets/syfry5suvzovvakmuj/pascal3d-synth-pack",
)


def _auto_synth_pack(cfg: Config) -> None:
    """Point --synth_pack_dir at the mounted pack when the run asks for warp / synthetic data."""
    f, d = cfg.features, cfg.data
    if d.synth_pack_dir or cfg.run.dataset != "pascal" or not (f.use_warp or f.use_synth):
        return
    for directory in SYNTH_PACK_DIRS:
        if (pathlib.Path(directory) / "synth_index.npy").exists():
            d.synth_pack_dir = directory
            print(f"synth pack: training data from {directory}")
            return


def _auto_pre_cache(cfg: Config) -> None:
    """Point --ram_cache_dir at the mounted pre-built tensors (Features.pre_cache)."""
    f, d = cfg.features, cfg.data
    if d.ram_cache_dir or not f.pre_cache:
        return
    # The tensors are one un-augmented pass over the images with no class labels, so they only
    # stand in for a normal build when nothing per-access is asked of the data.
    # With the synth pack the train set comes from the pack, so the cache only serves validation,
    # which use_warp / use_synth do not touch.
    per_access = (f.use_warp or f.use_synth) and not d.synth_pack_dir
    if (cfg.run.dataset != "pascal" or cfg.run.sanity_check or not f.ram_memory or per_access
            or f.raw_cache or f.fisher_prior or d.cache_draws != 1):
        return
    for directory in PRE_CACHE_DIRS:
        path = pathlib.Path(directory)
        if (path / "pascal_train.pt").exists() and (path / "pascal_val.pt").exists():
            d.ram_cache_dir = directory
            print(f"pre_cache: using the pre-built Pascal3D tensors in {directory}")
            return


def parse_args(argv: Optional[List[str]] = None) -> Config:
    """Parse command-line flags into a Config."""
    parser = create_argparser()
    ns = vars(parser.parse_args(argv))
    cfg = Config()
    for name, cls in _section_classes().items():
        values = {}
        for fld in dataclasses.fields(cls):
            value = ns[_flag(fld)]
            if get_origin(_unwrap_optional(get_type_hints(cls)[fld.name])[0]) in (tuple, Tuple) and value is not None:
                value = tuple(value)
            values[fld.name] = value
        setattr(cfg, name, cls(**values))
    _auto_synth_pack(cfg)
    _auto_pre_cache(cfg)
    return cfg
