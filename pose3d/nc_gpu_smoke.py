import torch
from pose3d.config import parse_args
from pose3d.train import make_algebra
from pose3d.models import build_model
from pose3d.engine import distributed
from pose3d.engine.trainer import build_step

cfg = parse_args(["--path_to_datasets", "/tmp", "--platform", "kaggle", "--model", "dit_flow",
    "--encoder", "resnet50", "--no-pretrained_backbone", "--dit_sphere_tokens", "--so3f_s2_channels", "32",
    "--dit_dim", "40", "--dit_mlp_ratio", "4.0", "--dit_pose_lmax", "6", "--symsol_nearest_coupling",
    "--ddp", "--sync_bn", "--num_gpus", "2", "--seed", "0"])
distributed.setup(cfg)
distributed.seed_everything(cfg)
model = build_model(cfg, make_algebra(3)).to(cfg.device)
assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 23756871
assert model.vector_field.pose_lmax == 6
model = distributed.convert_sync_bn(model, cfg)
step = build_step(model, torch.nn.MSELoss(), cfg).train()
eq = torch.stack([torch.eye(3), torch.diag(torch.tensor([-1., -1., 1.]))]).to(cfg.device)
data = dict(img=torch.rand(2, 3, 224, 224, device=cfg.device), rot=eq[0].expand(2, 3, 3),
            rots_eq=eq[None].expand(2, 2, 3, 3), n_eq=torch.full((2,), 2, device=cfg.device))
loss = step(data)
loss.backward()
assert torch.isfinite(loss)
assert model.vector_field.out.weight.grad.abs().max() > 0
if distributed.is_main():
    print(f"TWO_T4_NC_SMOKE_PASSED world_size={cfg.world_size} params=23756871 loss={loss.item():.6f}", flush=True)
distributed.cleanup()
