"""Run on Kaggle only: exact tensor algebra, physical covariance and gradient paths."""
import pytest
import torch
from pose3d.models.native_stf import CartesianMaps,NativeSTFCloudField
from pose3d.models.stf_track import stf_next,contract_down,apply_rotation
from pose3d.config import parse_args

torch.set_num_threads(2)

@pytest.fixture(scope='module')
def maps():return CartesianMaps(6)

@pytest.mark.parametrize('l',range(2,7))
def test_exact_cartesian_up_down(maps,l):
    torch.manual_seed(l)
    previous=getattr(maps,f'basis{l-1}')
    current=getattr(maps,f'basis{l}')
    torch.testing.assert_close(current.T@current,torch.eye(2*l+1),atol=4e-6,rtol=4e-6)
    s=torch.randn(2,3,2*l-1);v=torch.randn(2,3,3)
    full=(s@previous.T).reshape(2,3,*([3]*(l-1)))
    expected=stf_next(full,v).flatten(2)@current
    torch.testing.assert_close(maps.up(l,s,v,False),expected,atol=2e-6,rtol=2e-5)
    high=torch.randn(2,3,2*l+1)
    full=(high@current.T).reshape(2,3,*([3]*l))
    expected=contract_down(full,v).flatten(2)@previous
    torch.testing.assert_close(maps.down(l,high,v,False),expected,atol=2e-6,rtol=2e-5)

def transform(z,Q,l,maps):
    basis=getattr(maps,f'basis{l}')
    full=(z@basis.T).reshape(*z.shape[:-1],*([3]*l))
    return apply_rotation(full,Q,l).flatten(z.ndim-1)@basis

@pytest.mark.parametrize('reflection',[False,True])
def test_all_ranks_covariant(maps,reflection):
    torch.manual_seed(81)
    q,_=torch.linalg.qr(torch.randn(3,3))
    if reflection:q=q@torch.diag(torch.tensor([-1.,1.,1.]))
    v=torch.randn(2,3,3);v=v/v.norm(dim=-1,keepdim=True)
    a=maps.harmonic_seeds(v);b=maps.harmonic_seeds(v@q.T)
    for l in a:
        torch.testing.assert_close(b[l],transform(a[l],q,l,maps),atol=2e-6,rtol=3e-5)
        torch.testing.assert_close(a[l].norm(dim=-1),torch.ones(2,3),atol=1e-6,rtol=1e-6)

def inputs():
    torch.manual_seed(18)
    r,_=torch.linalg.qr(torch.randn(2,3,3))
    pose=torch.zeros(2,3,8)
    e1=r[:,:,0]
    pose[:,0,4:7]=torch.stack([e1[:,2],-e1[:,1],e1[:,0]],-1)
    pose[:,1,1:4]=r[:,:,1];pose[:,2,1:4]=r[:,:,2]
    cond=torch.randn(2,7,8+9);cond[...,:8]=0
    cond[...,1:4]=torch.nn.functional.normalize(torch.randn(2,7,3),dim=-1)
    return r,pose,torch.tensor([.25,.75]),cond

@pytest.mark.parametrize('reflection',[False,True])
def test_full_field_physical_covariance(reflection):
    model=NativeSTFCloudField(9,mv_channels=4,s_channels=16,num_blocks=2,num_heads=2,stf_channels=3).eval()
    r,pose,t,cond=inputs()
    q,_=torch.linalg.qr(torch.randn(3,3))
    if reflection:q=q@torch.diag(torch.tensor([-1.,1.,1.]))
    rp=q@r
    if reflection:rp=rp@torch.diag(torch.tensor([-1.,1.,1.]))
    other=pose.clone();e1=rp[:,:,0]
    other[:,0,4:7]=torch.stack([e1[:,2],-e1[:,1],e1[:,0]],-1)
    other[:,1,1:4]=rp[:,:,1];other[:,2,1:4]=rp[:,:,2]
    moved=cond.clone();moved[...,1:4]=cond[...,1:4]@q.T
    with torch.no_grad():a=model(pose,t,cond);b=model(other,t,moved)
    axial=torch.stack([a[:,2],-a[:,1],a[:,0]],-1)@q.T*torch.linalg.det(q)
    expected=torch.stack([axial[:,2],-axial[:,1],axial[:,0]],-1)
    assert a.abs().max()>1e-4
    torch.testing.assert_close(b,expected,atol=3e-5,rtol=5e-4)

def test_persistent_states_image_dependence_and_gradients():
    model=NativeSTFCloudField(9,mv_channels=4,s_channels=16,num_blocks=4,num_heads=2,stf_channels=3)
    _,pose,t,cond=inputs();cond.requires_grad_()
    _,_,seed=model.native_seed(pose,t,cond)
    changed=cond.detach().clone();changed[...,8:]+=1
    _,_,second=model.native_seed(pose,t,changed)
    for l in range(2,7):assert not torch.allclose(seed[l],second[l])
    seen=[]
    hooks=[]
    def record(module,args,out):
        seen.append(out[2]);[z.retain_grad() for z in out[2].values()]
    if model.fast:   # the fused-rank path never calls block.forward, so record its packed states instead
        import pose3d.models.native_stf_fast as nsf
        reference_fn=nsf.fast_block_forward
        def recorded(block,mv,s,T,fm):
            out=reference_fn(block,mv,s,T,fm);out[2].retain_grad();seen.append(out[2]);return out
        nsf.fast_block_forward=recorded
    else:
        for block in model.stf_blocks:hooks.append(block.register_forward_hook(record))
    try:
        loss=(model(pose,t,cond)-torch.tensor([.4,-.7,.6])).square().mean();loss.backward()
    finally:
        if model.fast:nsf.fast_block_forward=reference_fn
    assert torch.isfinite(loss) and len(seen)==4
    for j,state in enumerate(seen):
        if model.fast:
            assert state.shape[-2:]==(5,13)
            for i,l in enumerate(range(2,7)):
                assert (state[...,i,2*l+1:]==0).all(),(j,l)   # padding stays exactly zero
            grads={l:state.grad[...,i,:2*l+1] for i,l in enumerate(range(2,7))}
            for l,g in grads.items():
                assert g.shape[-1]==2*l+1 and torch.isfinite(g).all() and g.abs().max()>0,(j,l)
            continue
        for l,z in state.items():
            assert z.shape[-1]==2*l+1
            assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().max()>0,(j,l)
    for name,p in model.named_parameters():
        if name.startswith(('stf_seed','stf_blocks')):
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert cond.grad[...,8:].abs().max()>0
    [h.remove() for h in hooks]

def test_no_nc_launch_config():
    cfg=parse_args(['--path_to_datasets','/tmp','--vector_field','gatr','--condition_head','none',
        '--cond_tokens','spherecloud','--pose_tokens','frame_pin','--gatr_native_stf',
        '--no-gatr_zero_readout','--no-symsol_nearest_coupling','--no-symsol_likelihood'])
    assert cfg.flow.gatr_native_stf and cfg.flow.gatr_native_stf_lmax==6
    assert not cfg.data.symsol_nearest_coupling and not cfg.flow.gatr_zero_readout


def _rotate_cl3(x,q):
    """A rotation q acting on Cl(3,0) multivectors (blades 1, e1, e2, e3, e12, e13, e23, e123)."""
    y=x.clone();y[...,1:4]=x[...,1:4]@q.T
    w=torch.stack([x[...,6],-x[...,5],x[...,4]],-1)@q.T      # bivector -> dual vector, rotate, back
    y[...,4],y[...,5],y[...,6]=w[...,2],-w[...,1],w[...,0]
    return y

def test_stf_vector_field_so2_frame_ch_covariant_and_trainable():
    from pose3d.models.native_stf import STFVectorField
    torch.manual_seed(5)
    f=STFVectorField(6,mv_channels=4,s_channels=16,num_blocks=2,num_heads=2,stf_channels=3)
    x=torch.randn(2,2+6,3,8)
    x[:,1,:,1:]=0;x[:,1,1:]=0                                 # time token: a scalar in channel 0
    x[:,2:,1:]=0                                              # so2 tokens live in channel 0
    q,_=torch.linalg.qr(torch.randn(3,3));q=q*torch.sign(torch.linalg.det(q))
    a=f(x);b=f(_rotate_cl3(x,q))
    assert a.shape==(2,1,8) and a.abs().max()>1e-5
    torch.testing.assert_close(b,_rotate_cl3(a,q),atol=3e-5,rtol=5e-4)
    a.square().sum().backward()
    for name,p in f.named_parameters():
        if name.startswith(('stf_seed','stf_blocks')):
            assert p.grad is not None and torch.isfinite(p.grad).all(),name

def test_stf_vector_field_launch_config():
    cfg=parse_args(['--path_to_datasets','/tmp','--vector_field','gatr','--cond_tokens','so2','--pose_tokens','frame_ch',
        '--so2_split_norm','--condition_head','none','--gatr_native_stf','--gatr_native_stf_channels','8',
        '--no-pretrained_backbone'])
    from pose3d.models import build_model
    from pose3d.train import make_algebra
    from pose3d.models.native_stf import STFVectorField
    model=build_model(cfg,make_algebra(cfg.model.algebra_dim))
    assert isinstance(model.vector_field,STFVectorField)

def test_stf_vector_field_full_model_bf16_autocast():
    # The launch model end to end under bf16 autocast (POSE3D_AMP=bf16), on CPU: catches dtype mismatches before a GPU run.
    cfg=parse_args(['--path_to_datasets','/tmp','--vector_field','gatr','--cond_tokens','so2','--pose_tokens','frame_ch',
        '--so2_split_norm','--condition_head','none','--gatr_native_stf','--gatr_native_stf_channels','8',
        '--no-pretrained_backbone','--n_time_samples','2'])
    from pose3d.models import build_model
    from pose3d.train import make_algebra
    torch.manual_seed(3)
    model=build_model(cfg,make_algebra(cfg.model.algebra_dim))
    img=torch.randn(2,3,224,224)
    q,_=torch.linalg.qr(torch.randn(2,3,3));rot=q*torch.sign(torch.linalg.det(q))[:,None,None]
    with torch.autocast('cpu',dtype=torch.bfloat16):
        loss=model.compute_loss(img,rot)
    assert loss.dtype==torch.float32 and torch.isfinite(loss)
    loss.backward()
    grads=[p.grad for n,p in model.named_parameters() if n.startswith('vector_field.stf_') and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
