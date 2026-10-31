"""Native Cartesian-STF + GATr cloud field, SO(3)/O(3) covariant head.

STF_2..STF_6 are persistent per-token states, not scalar features discarded at
the first block. Fixed compact up/down maps are obtained from the Cartesian
STF recursion in stf_track.py, in orthonormal 2*l+1 coordinates. No CG tables
or Wigner-D matrices are used. The image seed is learned scalar weight times
STF_l(unit sphere direction), with the same physical pose frame as frame_pin.
"""
import math

import torch
from torch import nn

from pose3d.models.gatr_denoiser import GATrCloudField, _VECTOR, _ROTATION_BIVECTOR
from pose3d.models.stf_track import stf_basis, stf_next, contract_down


class CartesianMaps(nn.Module):
    """Fixed exact STF maps plus invariant rank-wise variance calibration."""
    def __init__(self, lmax=6):
        super().__init__()
        self.lmax=int(lmax)
        if not 2<=self.lmax<=6:raise ValueError('Native STF supports ranks 2..6')
        self.register_buffer('basis1',torch.eye(3),persistent=False)
        for l in range(2,self.lmax+1):
            self.register_buffer(f'basis{l}',stf_basis(l).clone(),persistent=False)
            previous=getattr(self,f'basis{l-1}').double()
            current=getattr(self,f'basis{l}').double()
            dp,dl=previous.shape[1],current.shape[1]
            old=previous.T.reshape(dp,*([3]*(l-1)))
            old=old[:,None].expand(dp,3,*([3]*(l-1)))
            v=torch.eye(3,dtype=torch.float64)[None].expand(dp,3,3)
            up=stf_next(old,v).reshape(dp,3,-1)@current
            full=current.T.reshape(dl,*([3]*l))
            full=full[:,None].expand(dl,3,*([3]*l))
            vv=torch.eye(3,dtype=torch.float64)[None].expand(dl,3,3)
            down=contract_down(full,vv).reshape(dl,3,-1)@previous
            self.register_buffer(f'up{l}',up.float(),persistent=False)
            self.register_buffer(f'down{l}',down.float(),persistent=False)
            # Unit-norm isotropic inputs have expected output norm one after
            # calibration. This changes no transformation law and avoids a
            # systematic multiplicative suppression of large ranks.
            self.register_buffer(f'up_scale{l}',torch.tensor(math.sqrt(dp*3/float(up.square().sum()))),persistent=False)
            self.register_buffer(f'down_scale{l}',torch.tensor(math.sqrt(dl*3/float(down.square().sum()))),persistent=False)

    def up(self,l,S,v,calibrated=True):
        out=torch.einsum('...ca,...ci,aio->...co',S,v,getattr(self,f'up{l}').to(S))
        return out*getattr(self,f'up_scale{l}').to(S) if calibrated else out

    def down(self,l,S,v,calibrated=True):
        out=torch.einsum('...ca,...ci,aio->...co',S,v,getattr(self,f'down{l}').to(S))
        return out*getattr(self,f'down_scale{l}').to(S) if calibrated else out

    def harmonic_seeds(self,v):
        # v: (..., C, 3), unit polar vectors. Normalization is an invariant scalar.
        seeds={1:v}
        for l in range(2,self.lmax+1):
            z=self.up(l,seeds[l-1],v,calibrated=False)
            seeds[l]=z/z.square().sum(-1,keepdim=True).clamp(min=1e-12).sqrt()
        return {l:seeds[l] for l in range(2,self.lmax+1)}


def channel_mix(x,layer):
    return layer(x.transpose(-1,-2)).transpose(-1,-2)


def track_norm(x):
    # One invariant radius per token; retains relative channel amplitudes.
    return x/(x.square().sum(-1).mean(-1,keepdim=True)[...,None]+1e-6).sqrt()


class ImageGeometrySeed(nn.Module):
    def __init__(self,scalar_dim,lmax,channels):
        super().__init__()
        self.lmax,self.channels=lmax,channels
        self.vectors=nn.Linear(3,channels,bias=False)
        self.amplitude=nn.ModuleDict({str(l):nn.Linear(scalar_dim,channels) for l in range(2,lmax+1)})
        for layer in self.amplitude.values():
            nn.init.normal_(layer.weight,std=.03)
            nn.init.constant_(layer.bias,.2)

    def forward(self,mv,scalars,maps):
        # Raw input has 3 PGA channels. Only honest polar grade-1 vectors are
        # mixed: pose axes e2/e3 and cloud directions. Axial e1 stays a bivector.
        polar=mv[...,list(_VECTOR)]
        v=channel_mix(polar,self.vectors)
        v=v/(v.square().sum(-1,keepdim=True)+1e-8).sqrt()
        geometry=maps.harmonic_seeds(v)
        return {l:geometry[l]*self.amplitude[str(l)](scalars)[...,None] for l in geometry}


class NativeSTFBlock(nn.Module):
    def __init__(self,mv_channels,s_channels,lmax,channels):
        super().__init__()
        self.lmax,self.channels=lmax,channels
        self.scalar_norm=nn.LayerNorm(s_channels)
        self.qs=nn.Linear(s_channels,16,bias=False)
        self.ks=nn.Linear(s_channels,16,bias=False)
        self.q=nn.ModuleDict({str(l):nn.Linear(channels,channels,bias=False) for l in range(2,lmax+1)})
        self.k=nn.ModuleDict({str(l):nn.Linear(channels,channels,bias=False) for l in range(2,lmax+1)})
        self.val=nn.ModuleDict({str(l):nn.Linear(channels,channels,bias=False) for l in range(2,lmax+1)})
        self.mix=nn.ModuleDict({str(l):nn.Linear(channels,channels,bias=False) for l in range(2,lmax+1)})
        self.v=nn.Linear(mv_channels,channels,bias=False)
        self.w=nn.Linear(mv_channels,channels,bias=False)
        self.gate=nn.ModuleDict({str(l):nn.Linear(s_channels+channels,channels) for l in range(2,lmax+1)})
        self.polar_out=nn.ModuleDict({str(l):nn.Linear(channels,mv_channels,bias=False) for l in range(2,lmax+1)})
        self.axial_out=nn.ModuleDict({str(l):nn.Linear(channels,mv_channels,bias=False) for l in range(2,lmax+1)})
        self.scalar_out=nn.Linear(2*(lmax-1)*channels,s_channels)
        self.feedback_gate=nn.Linear(s_channels,channels)
        self.track_step=nn.Parameter(torch.full((lmax-1,),.1))
        self.mv_step=nn.Parameter(torch.tensor(.1))
        self.s_step=nn.Parameter(torch.tensor(.1))

    def forward(self,mv,scalars,tracks,maps):
        s=self.scalar_norm(scalars)
        normalized={l:track_norm(z) for l,z in tracks.items()}
        score=self.qs(s)@self.ks(s).transpose(-1,-2)/4
        for l,z in normalized.items():
            q=channel_mix(z,self.q[str(l)]).flatten(-2)
            k=channel_mix(z,self.k[str(l)]).flatten(-2)
            score=score+(q@k.transpose(-1,-2))/math.sqrt(q.shape[-1]*(self.lmax-1))
        attention=score.softmax(-1)
        vectors=mv[...,list(_VECTOR)]
        v=channel_mix(vectors,self.v)
        w=channel_mix(vectors,self.w)
        v=v/(v.square().sum(-1,keepdim=True)+1e-6).sqrt()
        w=w/(w.square().sum(-1,keepdim=True)+1e-6).sqrt()
        updated={};invariants=[]
        for l in range(2,self.lmax+1):
            value=channel_mix(normalized[l],self.val[str(l)])
            message=torch.einsum('bpn,bncd->bpcd',attention,value)
            lower=v if l==2 else normalized[l-1]
            upward=maps.up(l,lower,v)
            downward=maps.down(l+1,normalized[l+1],w) if l<self.lmax else torch.zeros_like(message)
            delta=channel_mix((message+upward+downward)/math.sqrt(3),self.mix[str(l)])
            norms=(tracks[l].square().sum(-1)+1e-8).sqrt()
            gate=torch.sigmoid(self.gate[str(l)](torch.cat([s,torch.log1p(norms)],-1)))[...,None]
            updated[l]=tracks[l]+self.track_step[l-2]*gate*delta
            invariants.extend([torch.log1p(norms),(normalized[l]*message).sum(-1)])
        # Scalar update is used immediately by the feedback gate, including in
        # the final block; no trainable "dead tail" under DDP.
        scalars=scalars+self.s_step*self.scalar_out(torch.cat(invariants,-1))
        gate=(1+torch.tanh(self.feedback_gate(self.scalar_norm(scalars))))[...,None]
        polar=torch.zeros_like(vectors);axial=torch.zeros_like(vectors)
        for l,z in updated.items():
            vector=track_norm(z)
            for rank in range(l,1,-1):vector=maps.down(rank,vector,v)
            vector=vector*gate
            polar=polar+channel_mix(vector,self.polar_out[str(l)])
            omega=torch.linalg.cross(vector,w.to(vector.dtype),dim=-1)   # autocast: the operands can differ
            bivector=torch.stack([omega[...,2],-omega[...,1],omega[...,0]],-1)
            axial=axial+channel_mix(bivector,self.axial_out[str(l)])
        # Scatter into their honest PGA types: polar vector and axial bivector.
        delta=torch.zeros_like(mv)
        # .to(delta.dtype): under autocast polar/axial and mv can differ in dtype, and index_put needs one.
        delta[...,list(_VECTOR)]=(polar/math.sqrt(self.lmax-1)).to(delta.dtype)
        delta[...,list(_ROTATION_BIVECTOR)]=(axial/math.sqrt(self.lmax-1)).to(delta.dtype)
        return mv+self.mv_step*delta,scalars,updated


class NativeSTFCloudField(GATrCloudField):
    def __init__(self,cond_scalars,mv_channels=8,s_channels=32,num_blocks=4,
                 num_heads=4,final_norm=False,cond_mv=8,pose_lmax=0,
                 stf_lmax=6,stf_channels=8):
        if pose_lmax:raise ValueError('Native STF launch does not use external pose harmonics')
        super().__init__(cond_scalars,mv_channels,s_channels,num_blocks,num_heads,
                         final_norm,cond_mv)
        self.stf_lmax,self.stf_channels=stf_lmax,stf_channels
        self.cartesian=CartesianMaps(stf_lmax)
        self.stf_seed=ImageGeometrySeed(cond_scalars+2,stf_lmax,stf_channels)
        self.stf_blocks=nn.ModuleList([NativeSTFBlock(mv_channels,s_channels,stf_lmax,stf_channels)
                                      for _ in range(num_blocks)])
        # NATIVE_STF_FAST=1: the fused-rank forward of native_stf_fast.py (same function, same parameters).
        import os
        self.fast=os.environ.get('NATIVE_STF_FAST')=='1'
        self._stacked=None

    def native_seed(self,pose,t,cond):
        b,n=cond.shape[:2]
        mv=pose.new_zeros(b,1+n,3,16)
        mv[:,0,:,self._pga_index]=pose
        if self.cond_mv==16:mv[:,1:,0]=cond[...,:16]
        else:mv[:,1:,0,self._pga_index]=cond[...,:8]
        scalars=pose.new_zeros(b,1+n,self.cond_scalars+2)
        scalars[:,1:,:self.cond_scalars]=cond[...,self.cond_mv:]
        scalars[...,-2]=t[:,None];scalars[:,0,-1]=1
        return mv,scalars,self.stf_seed(mv,scalars,self.cartesian)

    def forward(self,pose,t,cond,R=None):
        from gatr.utils.tensors import construct_reference_multivector
        mv,s,tracks=self.native_seed(pose,t,cond)
        if self.fast:return self._fast_forward(mv,s,tracks)
        reference=construct_reference_multivector('data',mv)
        add_mv,add_s=self.net._construct_reinserted_channels(mv,s)
        mv,s=self.net.linear_in(mv,scalars=s)
        for gatr_block,stf_block in zip(self.net.blocks,self.stf_blocks):
            mv,s=gatr_block(mv,scalars=s,reference_mv=reference,
                            additional_qk_features_mv=add_mv,
                            additional_qk_features_s=add_s,attention_mask=None)
            mv,s,tracks=stf_block(mv,s,tracks,self.cartesian)
        out,_=self.net.linear_out(mv,scalars=s)
        return out[:,0,0][:,self._rot_index]

    def _fast_forward(self,mv,s,tracks):
        from gatr.utils.tensors import construct_reference_multivector
        from pose3d.models.native_stf_fast import StackedMaps,block_fn,pack
        if self._stacked is None or self._stacked.up.device!=mv.device:
            self._stacked=StackedMaps(self.cartesian,self.stf_lmax,mv.device)
        T=pack(tracks,self.stf_lmax)
        reference=construct_reference_multivector('data',mv)
        add_mv,add_s=self.net._construct_reinserted_channels(mv,s)
        mv,s=self.net.linear_in(mv,scalars=s)
        block=block_fn()
        for gatr_block,stf_block in zip(self.net.blocks,self.stf_blocks):
            mv,s=gatr_block(mv,scalars=s,reference_mv=reference,
                            additional_qk_features_mv=add_mv,
                            additional_qk_features_s=add_s,attention_mask=None)
            mv,s,T=block(stf_block,mv,s,T,self._stacked)
        out,_=self.net.linear_out(mv,scalars=s)
        return out[:,0,0][:,self._rot_index]


class STFVectorField(nn.Module):
    """GATrVectorField (so2 condition tokens, pose_tokens frame_ch) with the native STF track after every GATr block.

    Input as GATrVectorField with pose_channels=3: (B, 2 + n_cond, 3, 8) Cl(3,0) multivectors; token 0 holds the frame
    R e1, R e2, R e3 as 3 channels, token 1 the time, tokens 2.. the so2 tokens (channel 0, the e3 axis token last).
    The STF seed reads every token's 3 channel vectors (polar: R e_i, the so2 tokens' feature-weighted in-plane
    directions, e3). Its amplitudes see O(3)-invariant per-token scalars only: the GATr scalars (t, query flag), each
    channel's grade-0 part and each channel's vector norm, so the track stays exactly covariant. The read-out is
    GATrVectorField's (token 0's vector part, dualised to the camera-frame velocity bivector).
    """

    def __init__(self,n_cond_mv,mv_channels=8,s_channels=32,num_blocks=4,num_heads=4,n_pose_tokens=1,
                 pose_channels=3,stf_lmax=6,stf_channels=8):
        super().__init__()
        from pose3d.models.gatr_denoiser import GATrVectorField,_PGA_INDEX
        if n_pose_tokens!=1 or pose_channels!=3:raise ValueError('STFVectorField needs pose_tokens=frame_ch')
        base=GATrVectorField(n_cond_mv,mv_channels,s_channels,num_blocks,num_heads,n_pose_tokens,pose_channels)
        self.net,self.out=base.net,base.net.linear_out
        self.n_pose_tokens,self.pose_channels=n_pose_tokens,pose_channels
        self.register_buffer('_pga_index',torch.tensor(_PGA_INDEX),persistent=False)
        self.register_buffer('_vec_index',torch.tensor(_VECTOR),persistent=False)
        self.stf_lmax,self.stf_channels=stf_lmax,stf_channels
        self.cartesian=CartesianMaps(stf_lmax)
        self.stf_seed=ImageGeometrySeed(1+n_pose_tokens+2*pose_channels,stf_lmax,stf_channels)
        self.stf_blocks=nn.ModuleList([NativeSTFBlock(mv_channels,s_channels,stf_lmax,stf_channels)
                                      for _ in range(num_blocks)])

    def forward(self,x):
        from gatr.utils.tensors import construct_reference_multivector
        if x.dim()==3:x=x.unsqueeze(2)
        b,n,c,_=x.shape
        pga=x.new_zeros(b,n,c,16)
        pga[...,self._pga_index]=x
        scalars=x.new_zeros(b,n,1+self.n_pose_tokens)
        scalars[...,0]=x[:,self.n_pose_tokens,0,0].unsqueeze(1)   # t: scalar part of the time token
        scalars[:,0,1]=1.0                                        # query flag on the pose token
        vec=pga[...,list(_VECTOR)]
        seed_scalars=torch.cat([scalars,pga[...,0],(vec.square().sum(-1)+1e-12).sqrt()],-1)
        tracks=self.stf_seed(pga,seed_scalars,self.cartesian)
        reference=construct_reference_multivector('data',pga)
        add_mv,add_s=self.net._construct_reinserted_channels(pga,scalars)
        mv,s=self.net.linear_in(pga,scalars=scalars)
        for gatr_block,stf_block in zip(self.net.blocks,self.stf_blocks):
            mv,s=gatr_block(mv,scalars=s,reference_mv=reference,additional_qk_features_mv=add_mv,
                            additional_qk_features_s=add_s,attention_mask=None)
            mv,s,tracks=stf_block(mv,s,tracks,self.cartesian)
        out_mv,_=self.net.linear_out(mv,scalars=s)
        w=out_mv[:,0,0][:,self._vec_index]
        result=x.new_zeros(b,1,x.shape[-1])
        result[:,0,4:7]=torch.stack([w[:,2],-w[:,1],w[:,0]],dim=-1).to(result.dtype)
        return result
