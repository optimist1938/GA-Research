import torch
import torch.nn as nn
from pathlib import Path

from clifford.algebra.cliffordalgebra import CliffordAlgebra

from src.config import create_argparser
from src.dataset import create_dataloaders
from src.model import TralaleroCompetitor, MLPBaseline, ViTMultiLayerPoseBaseline, I2S, GA_I2S, I2S_Backbone, I2S_ResNet
from src.train_utils import (
    train,
    form_checkpoint,
    get_available_device,
    load_checkpoint,
    maybe_wrap_model_for_multi_gpu,
    rotation_matrix_loss,
    geodesic_rotation_matrix_loss,
    rotor_loss,
    multivector_rotor_loss,
)
from src.evaluation_metrics import calculate_evaluation_metrics,create_technical_matrices
from src.wandb_utils import (
    wandb_create_run,
    wandb_log_code,
    wandb_log_artifact,
    wandb_finish_run,
)



def _make_algebra(algebra_dim: int):
    algebra_dim = int(algebra_dim)
    if algebra_dim <= 0:
        raise ValueError("algebra_dim must be positive")

    mv_dim = 2 ** algebra_dim
    if algebra_dim > 6:
        print(
            f"Warning: algebra_dim={algebra_dim} gives mv_dim={mv_dim}. "
            "This can significantly increase memory usage and runtime."
        )

    return CliffordAlgebra(tuple([1] * algebra_dim))


def instantiate(config):
    config.algebra_dim = int(config.algebra_dim)
    if config.algebra_dim <= 0:
        raise ValueError("algebra_dim must be positive")

    vit_pooling_type = str(getattr(config, "vit_pooling_type", "mean")).lower()
    vit_ga_readout_type = str(getattr(config, "vit_ga_readout_type", "linear")).lower()
    vit_ga_requested = (
        config.model == "vit_baseline"
        and vit_pooling_type == "ga"
    )
    if vit_ga_requested and vit_ga_readout_type == "rotor" and config.algebra_dim != 3:
        raise ValueError("vit_ga_readout_type='rotor' requires algebra_dim=3 / Cl(3,0)")

    if config.model != "i2s_resnet" and not vit_ga_requested and config.algebra_dim != 3:
        raise ValueError(
            "Variable algebra_dim is currently supported only for model='i2s_resnet' "
            "or model='vit_baseline' with --vit_pooling_type ga. "
            "Use a supported model/pooling combination or set --algebra_dim 3."
        )

    rotor_losses = {"rotor", "mv_rotor"}
    rotor_modes = {"rotor", "multivector_rotor"}
    requested_rotor_mode = (
        config.loss in rotor_losses
        or getattr(config, "i2s_resnet_output_mode", None) in rotor_modes
    )
    if requested_rotor_mode and config.algebra_dim != 3:
        raise ValueError(
            "rotor and mv_rotor modes currently require algebra_dim=3, "
            "because rotor extraction is implemented specifically for Cl(3,0). "
            "Use --algebra_dim 3 or implement generalized rotor extraction."
        )

    algebra = _make_algebra(config.algebra_dim)

    train_loader, val_loader = create_dataloaders(config)
    print("Created Tralaloaders")

    if config.model == "tralalero":
        model = TralaleroCompetitor(
            algebra,
            encoder_type=config.encoder,
            ga_pool_hw=tuple(config.ga_pool_hw),
            hidden_dim=config.hidden_dim
        )
    elif config.model == "mlp":
        model = MLPBaseline(encoder_type=config.encoder)
    elif config.model == "vit_baseline":
        model = ViTMultiLayerPoseBaseline(
            model_name=config.vit_model_name,
            backbone_type=config.vit_backbone_type,
            layers=tuple(config.vit_layers),
            freeze_vit=config.freeze_vit,
            pooling_type=getattr(config, "vit_pooling_type", "mean"),
            num_transformer_layers=getattr(config, "vit_num_transformer_layers", 1),
            transformer_nhead=getattr(config, "vit_transformer_nhead", 8),
            transformer_ff_dim=getattr(config, "vit_transformer_ff_dim", 1024),
            transformer_dropout=getattr(config, "vit_transformer_dropout", 0.1),
            algebra=algebra,
            ga_input_features=getattr(config, "vit_ga_input_features", 196),
            ga_hidden_dim=getattr(config, "vit_ga_hidden_dim", [32]),
            ga_readout_type=getattr(config, "vit_ga_readout_type", "linear"),
        )
    elif config.model == "i2s":
        model = I2S(
            algebra=algebra,
            lmax=config.lmax,
            rec_level=config.rec_level,
            n_mv=config.n_mv,
            hidden_dim=config.hidden_dim,
            temperature=config.temperature,
            encoder_type=config.encoder,
        )
    elif config.model == "ga_i2s":
        model = GA_I2S(
            algebra=algebra,
            lmax=config.lmax,
            rec_level=config.rec_level,
            n_mv=config.n_mv,
            hidden_dim=config.hidden_dim,
            temperature=config.temperature,
            encoder_type=config.encoder,
            ga_pool_hw=tuple(config.ga_pool_hw),
        )
    elif config.model == "i2s_resnet":
        output_mode = config.i2s_resnet_output_mode
        if output_mode == "auto":
            if config.loss == "prob":
                output_mode = "fourier"
            elif config.loss == "rotor":
                output_mode = "rotor"
            elif config.loss == "mv_rotor":
                output_mode = "multivector_rotor"
            else:
                output_mode = "rotation_matrix"
        model = I2S_Backbone(
            algebra=algebra,
            lmax=config.lmax,
            rec_level=config.rec_level,
            hidden_dim=config.hidden_dim,
            temperature=config.temperature,
            backbone_name=config.i2s_resnet_backbone_name,
            pretrained_backbone=config.i2s_resnet_pretrained_backbone,
            freeze_backbone=config.i2s_resnet_freeze_backbone,
            use_positional_encoding=config.i2s_resnet_use_positional_encoding,
            output_mode=output_mode,
            mv_per_position=config.i2s_resnet_mv_per_position,
            adapter_mid_channels=config.i2s_resnet_adapter_mid_channels,
            adapter_high_channels=config.i2s_resnet_adapter_high_channels,
            adapter_output_size=config.i2s_resnet_adapter_output_size,
            ga_head_type=config.i2s_resnet_ga_head_type,
            ga_head_mixing_layer=config.i2s_resnet_ga_head_mixing_layer,
            ga_head_num_blocks=config.i2s_resnet_ga_num_blocks,
            ga_head_dropout=config.i2s_resnet_ga_head_dropout,
            ga_head_use_layer_norm=config.i2s_resnet_ga_head_use_layer_norm,
        )
    else:
        raise ValueError(f"Unknown model: {config.model}")
    
    config.device = get_available_device()
    
    model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    warmup_epochs = 5
    cosine_epochs = config.n_epochs - warmup_epochs

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1, 
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=config.lr * 0.05 
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )

    if config.loss == "mse":
        criterion = rotation_matrix_loss
    elif config.loss == "geodesic":
        criterion = geodesic_rotation_matrix_loss
    elif config.loss == "prob":
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    elif config.loss == "rotor":
        criterion = rotor_loss
    elif config.loss == "mv_rotor":
        criterion = multivector_rotor_loss
    else:
        raise ValueError(f"Unknown loss: {config.loss}")

    run = wandb_create_run(
        config.run_name,
        project=config.wandb_project,
        entity=config.wandb_entity,
        group=config.wandb_group,
    )
    print("W&B logging set up completed")

    return train_loader, val_loader, model, optimizer, scheduler, criterion, run


def main():
    parser = create_argparser()
    config = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, model, optimizer, scheduler, criterion, run = instantiate(config)
    create_technical_matrices(config)

    path = config.path_to_checkpoint
    if path is not None:
        
        try:
            load_checkpoint(model, optimizer, scheduler, path , config.device)
            print("Checkpoint successfully loaded. Starting evaluation")
            torch.save(torch.tensor(calculate_evaluation_metrics(model,val_loader,config)),"res.pth")
        except Exception as e:
            print(f"Failed to load checkpoint. Starting from scratch. Error: {e}")

    wandb_log_code(run, Path("."))
    torch.cuda.empty_cache()
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, run, config)

    checkpoint_path = form_checkpoint(model, optimizer, scheduler, config)
    wandb_log_artifact(run, checkpoint_path, artifact_type="checkpoint")
    wandb_finish_run(run)


if __name__ == "__main__":
    main()
