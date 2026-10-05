"""Model registry: `build_model(cfg, algebra)` maps `--model` to a constructed nn.Module.

Each entry imports its module lazily, so a run only pays for the dependencies of the model
it uses (transformers for the ViT / Depth-Anything models, open3d for the point-cloud ones).
"""

from typing import Callable, Dict

import torch.nn as nn

from pose3d.config import Config


def _clifford_flow(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.clifford_flow import CliffordFlow

    f, m, fl = cfg.features, cfg.model, cfg.flow
    if f.fisher_prior and not fl.fisher_checkpoint:
        raise ValueError("fisher_prior needs --fisher_checkpoint (Liu et al.'s state_dict_119.pkl)")
    return CliffordFlow(
        algebra,
        hidden_dim=fl.flow_hidden_dim,
        n_cond_mv=fl.n_cond_mv,
        pretrained_backbone=f.pretrained_backbone,
        n_time_samples=cfg.n_time_samples,
        adapter_grid=fl.adapter_grid,
        adapter_channels=fl.adapter_channels,
        encoder_type=m.encoder,
        depth_anything_model=m.depth_anything_model,
        freeze_backbone=f.freeze_encoder,
        vector_field_hidden_dim=fl.vector_field_hidden_dim,
        conv_adapter=fl.conv_adapter,
        vector_field=fl.vector_field,
        condition_head=fl.condition_head,
        gatr=dict(num_blocks=fl.gatr_blocks, mv_channels=fl.gatr_mv_channels,
                  s_channels=fl.gatr_s_channels, num_heads=fl.gatr_heads),
        mlp_heads=f.mlp_heads,
        fisher_checkpoint=fl.fisher_checkpoint if f.fisher_prior else None,
    )


def _mlp_flow(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.mlp_flow import MLPFlow

    return MLPFlow(
        algebra,
        pretrained_backbone=cfg.features.pretrained_backbone,
        adapter_grid=cfg.flow.adapter_grid,
        encoder_type=cfg.model.encoder,
    )


def _i2s_real(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.i2s_real import I2SReal

    return I2SReal(
        encoder_type=cfg.model.encoder,
        pretrained_backbone=cfg.features.pretrained_backbone,
        lmax=cfg.i2s.lmax,
        rec_level=cfg.i2s.rec_level,
        eval_rec_level=cfg.i2s.i2s_eval_rec_level,
        normalize_input=cfg.i2s.i2s_normalize,
        depth_anything_model=cfg.model.depth_anything_model,
        freeze_backbone=cfg.features.freeze_encoder,
    )


def _i2s(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.i2s import I2S

    return I2S(
        algebra=algebra,
        lmax=cfg.i2s.lmax,
        rec_level=cfg.i2s.rec_level,
        n_mv=cfg.i2s.n_mv,
        hidden_dim=cfg.model.hidden_dim,
        temperature=cfg.i2s.temperature,
        encoder_type=cfg.model.encoder,
        pretrained_backbone=cfg.features.pretrained_backbone,
    )


def _ga_i2s(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.i2s import GA_I2S

    return GA_I2S(
        algebra=algebra,
        lmax=cfg.i2s.lmax,
        rec_level=cfg.i2s.rec_level,
        n_mv=cfg.i2s.n_mv,
        hidden_dim=cfg.model.hidden_dim,
        temperature=cfg.i2s.temperature,
        encoder_type=cfg.model.encoder,
        ga_pool_hw=tuple(cfg.i2s.ga_pool_hw),
    )


def _tralalero(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.direct import TralaleroCompetitor

    return TralaleroCompetitor(
        algebra,
        encoder_type=cfg.model.encoder,
        ga_pool_hw=tuple(cfg.i2s.ga_pool_hw),
        pretrained_backbone=cfg.features.pretrained_backbone,
        hidden_dim=cfg.model.hidden_dim,
    )


def _mlp(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.direct import MLPBaseline

    return MLPBaseline(encoder_type=cfg.model.encoder,
                       pretrained_backbone=cfg.features.pretrained_backbone)


def _resolve_i2s_backbone_mode(cfg: Config) -> str:
    mode = cfg.i2s_backbone.i2s_resnet_output_mode
    if mode != "auto":
        return mode
    return {"prob": "fourier", "rotor": "rotor", "mv_rotor": "multivector_rotor"}.get(
        cfg.train.loss, "rotation_matrix")


def _i2s_backbone(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.i2s_backbone import I2S_Backbone

    b = cfg.i2s_backbone
    return I2S_Backbone(
        algebra=algebra,
        lmax=cfg.i2s.lmax,
        rec_level=cfg.i2s.rec_level,
        hidden_dim=cfg.model.hidden_dim,
        temperature=cfg.i2s.temperature,
        backbone_name=b.i2s_resnet_backbone_name,
        pretrained_backbone=b.i2s_resnet_pretrained_backbone,
        freeze_backbone=b.i2s_resnet_freeze_backbone,
        use_positional_encoding=b.i2s_resnet_use_positional_encoding,
        output_mode=_resolve_i2s_backbone_mode(cfg),
        mv_per_position=b.i2s_resnet_mv_per_position,
        adapter_mid_channels=b.i2s_resnet_adapter_mid_channels,
        adapter_high_channels=b.i2s_resnet_adapter_high_channels,
        adapter_output_size=b.i2s_resnet_adapter_output_size,
        ga_head_type=b.i2s_resnet_ga_head_type,
        ga_head_mixing_layer=b.i2s_resnet_ga_head_mixing_layer,
        ga_head_num_blocks=b.i2s_resnet_ga_num_blocks,
        ga_head_dropout=b.i2s_resnet_ga_head_dropout,
        ga_head_use_layer_norm=b.i2s_resnet_ga_head_use_layer_norm,
    )


def _resolve_i2s_conv_mode(cfg: Config) -> str:
    mode = cfg.i2s_conv.i2s_conv_output_mode
    if mode != "auto":
        return mode
    return {"prob": "fourier", "rotor": "rotor"}.get(cfg.train.loss, "rotation_matrix")


def _i2s_conv_kwargs(cfg: Config) -> dict:
    c = cfg.i2s_conv
    return dict(
        lmax=cfg.i2s.lmax,
        rec_level=cfg.i2s.rec_level,
        hidden_dim=cfg.model.hidden_dim,
        temperature=cfg.i2s.temperature,
        pretrained_backbone=c.i2s_conv_pretrained_backbone,
        freeze_backbone=c.i2s_conv_freeze_backbone,
        use_positional_encoding=c.i2s_conv_use_positional_encoding,
        output_mode=_resolve_i2s_conv_mode(cfg),
        adapter_type=c.i2s_conv_adapter_type,
        head_type=c.i2s_conv_head_type,
    )


def _i2s_conv_resnet(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.i2s_conv import I2SConvResNet

    return I2SConvResNet(algebra=algebra, **_i2s_conv_kwargs(cfg))


def _i2s_conv_convnext(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.i2s_conv import I2SConvNext

    return I2SConvNext(algebra=algebra, variant=cfg.i2s_conv.i2s_conv_variant,
                       **_i2s_conv_kwargs(cfg))


def _ipdf_resnet(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.ipdf_clifford import IPDF_ResNet

    b = cfg.i2s_backbone
    return IPDF_ResNet(
        algebra=algebra,
        hidden_dim=cfg.model.hidden_dim,
        n_queries=cfg.ipdf.ipdf_n_queries,
        pretrained_backbone=b.i2s_resnet_pretrained_backbone,
        freeze_backbone=b.i2s_resnet_freeze_backbone,
        rec_level=cfg.i2s.rec_level,
    )


def _vit_baseline(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.vit_baseline import ViTMultiLayerPoseBaseline

    v = cfg.vit
    return ViTMultiLayerPoseBaseline(
        model_name=v.vit_model_name,
        backbone_type=v.vit_backbone_type,
        layers=tuple(v.vit_layers),
        freeze_vit=v.freeze_vit,
        pooling_type=v.vit_pooling_type,
        num_transformer_layers=v.vit_num_transformer_layers,
        transformer_nhead=v.vit_transformer_nhead,
        transformer_ff_dim=v.vit_transformer_ff_dim,
        transformer_dropout=v.vit_transformer_dropout,
        algebra=algebra,
        ga_input_features=v.vit_ga_input_features,
        ga_hidden_dim=v.vit_ga_hidden_dim,
        ga_readout_type=v.vit_ga_readout_type,
    )


def _image2pcd(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.ipdf_depth import I2P

    return I2P(freeze_backbone=cfg.ipdf.freeze_backbone)


def _image2pcd_pointnet(cfg: Config, algebra) -> nn.Module:
    from pose3d.engine.checkpoint import get_available_device
    from pose3d.models.i2p_pointcloud import I2PPointNet

    return I2PPointNet(n_points=cfg.pointcloud.i2p_n_points,
                       freeze_backbone=cfg.ipdf.freeze_backbone,
                       device=get_available_device())


def _image2pcd_late_fusion(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.i2p_pointcloud import I2PLateFusion

    return I2PLateFusion(n_points=cfg.pointcloud.i2p_n_points,
                         freeze_backbone=cfg.ipdf.freeze_backbone,
                         use_depth=cfg.pointcloud.i2p_use_depth)


def _image2pcd_ipdf(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.ipdf_depth import I2P_IPDF

    d = cfg.ipdf
    return I2P_IPDF(
        rec_level=cfg.i2s.rec_level,
        n_train_queries=d.n_train_queries,
        pe_freqs=d.pe_freqs,
        grad_ascent_steps=d.grad_ascent_steps,
        grad_ascent_lr=d.grad_ascent_lr,
        freeze_backbone=d.freeze_backbone,
        pretrained_backbone=cfg.features.pretrained_backbone,
    )


def _dummynet(cfg: Config, algebra) -> nn.Module:
    from pose3d.models.dummynet import DummyNet

    return DummyNet()


MODEL_BUILDERS: Dict[str, Callable[[Config, object], nn.Module]] = {
    "clifford_flow": _clifford_flow,
    "mlp_flow": _mlp_flow,
    "i2s_real": _i2s_real,
    "i2s": _i2s,
    "ga_i2s": _ga_i2s,
    "tralalero": _tralalero,
    "mlp": _mlp,
    "i2s_backbone": _i2s_backbone,
    "i2s_resnet": _i2s_backbone,
    "i2s_conv_resnet": _i2s_conv_resnet,
    "i2s_conv_convnext": _i2s_conv_convnext,
    "ipdf_resnet": _ipdf_resnet,
    "vit_baseline": _vit_baseline,
    "image2pcd": _image2pcd,
    "image2pcd_pointnet": _image2pcd_pointnet,
    "image2pcd_late_fusion": _image2pcd_late_fusion,
    "image2pcd_ipdf": _image2pcd_ipdf,
    "dummynet": _dummynet,
}

# Models whose GA layers work in a Clifford algebra other than Cl(3,0).
_VARIABLE_ALGEBRA_MODELS = {"i2s_backbone", "i2s_resnet"}


def validate(cfg: Config) -> None:
    """Reject option combinations that are not supported, before anything is built."""
    dim = int(cfg.model.algebra_dim)
    if dim <= 0:
        raise ValueError("algebra_dim must be positive")

    vit_ga = cfg.model.name == "vit_baseline" and cfg.vit.vit_pooling_type == "ga"
    if vit_ga and cfg.vit.vit_ga_readout_type == "rotor" and dim != 3:
        raise ValueError("vit_ga_readout_type='rotor' requires algebra_dim=3 / Cl(3,0)")

    if cfg.model.name not in _VARIABLE_ALGEBRA_MODELS and not vit_ga and dim != 3:
        raise ValueError(
            "Variable algebra_dim is currently supported only for model='i2s_backbone' "
            "or model='vit_baseline' with --vit_pooling_type ga. "
            "Use a supported model/pooling combination or set --algebra_dim 3."
        )

    rotor_losses = {"rotor", "mv_rotor"}
    rotor_modes = {"rotor", "multivector_rotor"}
    if (cfg.train.loss in rotor_losses
            or cfg.i2s_backbone.i2s_resnet_output_mode in rotor_modes) and dim != 3:
        raise ValueError(
            "rotor and mv_rotor modes currently require algebra_dim=3, "
            "because rotor extraction is implemented specifically for Cl(3,0). "
            "Use --algebra_dim 3 or implement generalized rotor extraction."
        )


def build_model(cfg: Config, algebra) -> nn.Module:
    validate(cfg)
    try:
        builder = MODEL_BUILDERS[cfg.model.name]
    except KeyError:
        raise ValueError(f"Unknown model: {cfg.model.name}") from None
    return builder(cfg, algebra)
