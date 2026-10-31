"""Kaggle real-allocation full-batch DDP smoke and timing, before training."""
import json,os,time
import torch
from pose3d.config import parse_args
from pose3d.train import make_algebra,build_optimizer
from pose3d.models import build_model
from pose3d.engine import distributed
from pose3d.engine.trainer import build_step

args=json.load(open('native_launch_args.json'))
cfg=parse_args(['--path_to_datasets','/tmp','--platform','kaggle',*args])
distributed.setup(cfg);distributed.seed_everything(cfg)
assert cfg.world_size==2 and torch.cuda.device_count()==2
assert not cfg.data.symsol_nearest_coupling and cfg.n_time_samples==8
model=build_model(cfg,make_algebra(3)).to(cfg.device)
model=distributed.convert_sync_bn(model,cfg)
assert type(model.vector_field).__name__=='NativeSTFCloudField'
assert len(model.vector_field.stf_blocks)==4
step=build_step(model,torch.nn.MSELoss(),cfg).train()
optimizer=build_optimizer(model,cfg)
bs=cfg.per_gpu_batch_size
assert bs==32,bs
data=dict(img=torch.rand(bs,3,224,224,device=cfg.device),
          rot=torch.eye(3,device=cfg.device).expand(bs,3,3))
times=[]
for i in range(4):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize();start=time.monotonic()
    loss=step(data);loss.backward()
    assert torch.isfinite(loss)
    if i==0:
        for name,p in model.named_parameters():
            if p.grad is not None:assert torch.isfinite(p.grad).all(),name
        for j,block in enumerate(model.vector_field.stf_blocks):
            for rank in range(2,7):
                g=block.mix[str(rank)].weight.grad
                assert g is not None and g.abs().max()>0,(j,rank)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
    torch.cuda.synchronize();times.append(time.monotonic()-start)
model.eval()
with torch.no_grad():
    torch.cuda.synchronize();start=time.monotonic()
    result=model.predict(data['img'],n_samples=1,steps=20)
    torch.cuda.synchronize();single_seconds=time.monotonic()-start
    assert result.shape==(bs,3,3) and torch.isfinite(result).all()
    torch.cuda.synchronize();start=time.monotonic()
    result=model.predict(data['img'],n_samples=32,steps=20)
    torch.cuda.synchronize();medoid_seconds=time.monotonic()-start
    assert result.shape==(bs,3,3) and torch.isfinite(result).all()
if distributed.is_main():
    result=dict(status='PASSED',world_size=cfg.world_size,batch_per_gpu=bs,
       field_batch_per_gpu=bs*cfg.n_time_samples,seconds_per_update=times,
       peak_GB=torch.cuda.max_memory_allocated()/2**30,
       trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
       nc=False,native_ranks=list(range(2,7)),stf_channels=8,
       seconds_single_batch32=single_seconds,seconds_medoid32_batch32=medoid_seconds)
    print('NATIVE_STF_TWO_T4_FULL_BATCH_SMOKE '+json.dumps(result),flush=True)
    open('/kaggle/working/native_gpu_smoke.json','w').write(json.dumps(result,indent=2))
distributed.cleanup()
