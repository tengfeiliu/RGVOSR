"""Experimental round modulation and bounded residual routing; base stays frozen."""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from models.lora_moe import iter_moe_lora_layers, route_logits


def gradient_consistency(prediction,target):
    """Match paired-HR derivatives; do not merely penalize image gradients."""
    horizontal=(prediction[:,:,:,1:]-prediction[:,:,:,:-1])-(target[:,:,:,1:]-target[:,:,:,:-1])
    vertical=(prediction[:,:,1:,:]-prediction[:,:,:-1,:])-(target[:,:,1:,:]-target[:,:,:-1,:])
    return horizontal.abs().mean()+vertical.abs().mean()


class RoundStateAdapter(nn.Module):
    def __init__(self, width, max_round=16, max_state_delta=.15):
        super().__init__()
        self.width, self.max_round = int(width), int(max_round)
        self.max_state_delta = float(max_state_delta)
        self.round_embed = nn.Embedding(self.max_round+1,self.width)
        nn.init.normal_(self.round_embed.weight,std=.02)
        self.round_gate = nn.Parameter(torch.zeros(()))
        self.state_round = nn.Embedding(self.max_round+1,32)
        self.state_encoder = nn.Sequential(nn.Conv2d(384,32,3,padding=1),nn.SiLU(),
            nn.Conv2d(32,32,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.state_head = nn.Sequential(nn.Linear(64,64),nn.SiLU(),nn.Linear(64,4))
        self.state_gate = nn.Parameter(torch.zeros(()))

    def rounds(self, rounds, batch, device):
        r=torch.as_tensor(rounds,device=device,dtype=torch.long).reshape(-1)
        if r.numel()==1: r=r.expand(batch)
        if r.numel()!=batch or (r<1).any() or (r>self.max_round).any():
            raise ValueError('Round must match batch and be in [1,max_round]')
        return r

    def inject(self, baseline, rounds):
        r=self.rounds(rounds,len(baseline),baseline.device)
        residual=self.round_embed(r).to(baseline.dtype)*self.round_gate.to(baseline.dtype)
        if baseline.ndim==3: residual=residual.unsqueeze(1)
        return baseline+residual

    def state_delta(self, current, anchor, rounds):
        r=self.rounds(rounds,len(current),current.device)
        a=F.adaptive_avg_pool2d(anchor.float(),(8,8))
        c=F.adaptive_avg_pool2d(current.float(),(8,8))
        features=self.state_encoder(torch.cat([a,c,c-a],dim=1))
        logits=self.state_head(torch.cat([features,self.state_round(r)],dim=1))
        return self.max_state_delta*self.state_gate.tanh()*logits.tanh()


@contextmanager
def hold_moe_routing(transformer, alpha):
    """Checkpoint backward must observe the forward dispatch weights."""
    layers=list(iter_moe_lora_layers(transformer))
    prior=[layer._routing_alpha for layer in layers]
    for layer in layers: layer.set_routing(alpha)
    try: yield
    finally:
        for layer,value in zip(layers,prior): layer.set_routing(value)


class AdapterHooks:
    """Scoped hooks; no installed Diffusers or base checkpoint modifications."""
    def __init__(self,artist,adapter,mode='temb',state=False):
        self.artist,self.adapter,self.mode,self.state=artist,adapter,mode,bool(state)
        self.context=None; self.last_route=None
        self.handles=[artist.register_forward_pre_hook(self.pre,with_kwargs=True)]
        point=artist.transformer.time_guidance_embed if mode=='temb' else artist.refinement_condition_adapter
        self.handles.append(point.register_forward_hook(self.inject))
        self.handles.append(artist.moe_router.register_forward_hook(self.route,with_kwargs=True))

    def pre(self,module,args,kwargs):
        self.context=kwargs

    def inject(self,module,args,output):
        k=(self.context or {}).get('refinement_round')
        return output if k is None else self.adapter.inject(output,k)

    def route(self,module,args,kwargs,output):
        self.last_route=None
        k=(self.context or {}).get('refinement_round')
        if not self.state or k is None: return output
        current=self.context['z_lr']; anchor=self.context['z_anchor_lr']
        delta=self.adapter.state_delta(current,anchor,k)
        old_logits=output['logits'].float(); logits=old_logits+delta
        temperature=float(kwargs.get('temperature',.7))
        dense=route_logits(logits,mode='soft',temperature=temperature)
        base=route_logits(old_logits,mode='soft',temperature=temperature).detach()
        alpha=route_logits(logits,mode=kwargs.get('routing_mode','topk'),
                           top_k=int(kwargs.get('top_k',2)),temperature=temperature)
        kl=(base*(base.clamp_min(1e-8).log()-dense.clamp_min(1e-8).log())).sum(-1).mean()
        self.last_route=dict(kl=kl,delta=delta,base=base,dense=dense,alpha=alpha)
        return {**output,'logits':logits,'routed_logits':logits,'alpha':alpha,
                'dense_alpha':dense,'clean_dense_alpha':dense,'dispatch_dense_alpha':dense}

    def close(self):
        for handle in self.handles: handle.remove()


def choose_candidate(f0,candidates,psnr_tol=.05,ssim_tol=.002):
    rows=[]
    for name,m in candidates:
        ok=(m['psnr']>=f0['psnr']-psnr_tol and m['ssim']>=f0['ssim']-ssim_tol
            and m['lpips']<=f0['lpips'] and m['dists']<=f0['dists'])
        rows.append(dict(name=name,metrics=m,eligible=bool(ok)))
    eligible=[r for r in rows if r['eligible']]
    winner=min(eligible,key=lambda r:r['metrics']['lpips']+r['metrics']['dists']) if eligible else None
    return dict(selected=winner['name'] if winner else None,candidates=rows)
