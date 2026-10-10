"""Spatial observation injection and a shared visual residual Transformer."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def positions(height,width,device,dtype):
    yy,xx=torch.meshgrid(torch.linspace(-1,1,height,device=device,dtype=dtype),
                         torch.linspace(-1,1,width,device=device,dtype=dtype),indexing='ij')
    return torch.stack([yy,xx],dim=-1).reshape(1,height*width,2)


class SpatialLRResidual(nn.Module):
    def __init__(self,hidden_width,latent_channels=128,attention_width=256,grid=32):
        super().__init__();self.grid=grid;self.attention_width=attention_width
        self.norm=nn.LayerNorm(hidden_width);self.query=nn.Linear(hidden_width,attention_width)
        self.key=nn.Linear(latent_channels,attention_width);self.value=nn.Linear(latent_channels,attention_width)
        self.position=nn.Linear(2,attention_width,bias=False)
        self.output=nn.Linear(attention_width,hidden_width)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,hidden,original_lr,geometry):
        height,width=geometry
        if hidden.shape[1]!=height*width or len(hidden)!=len(original_lr):
            raise ValueError('Spatial query/observation geometry or batch mismatch')
        reference=F.adaptive_avg_pool2d(original_lr.float(),(self.grid,self.grid)).flatten(2).transpose(1,2)
        query=self.query(self.norm(hidden.float()))+self.position(positions(height,width,hidden.device,torch.float32))
        key=self.key(reference)+self.position(positions(self.grid,self.grid,hidden.device,torch.float32))
        value=self.value(reference)
        attended=F.scaled_dot_product_attention(query.unsqueeze(1),key.unsqueeze(1),value.unsqueeze(1)).squeeze(1)
        return self.output(attended).to(hidden.dtype)


class FluxSpatialHooks:
    """A scoped image-space branch; no base model/package files change."""
    def __init__(self,actor,branch):
        self.context=None;self.branch=branch;self.last_delta=None;self.enabled=True
        self.handles=[actor.register_forward_pre_hook(self.pre,with_kwargs=True),
                      actor.transformer.x_embedder.register_forward_hook(self.inject)]

    def pre(self,module,args,kwargs):
        self.context=dict(kwargs)
        if args:self.context.setdefault('z_t',args[0])

    def inject(self,module,args,output):
        context=self.context or {};self.last_delta=None
        if not self.enabled or context.get('refinement_round') is None:return output
        original=context.get('z_anchor_lr');target=context['z_t'];h,w=target.shape[-2:]
        if original is None:raise ValueError('Spatial refinement requires original LR')
        delta=self.branch(output[:,:h*w],original,(h,w));self.last_delta=delta
        return torch.cat([output[:,:h*w]+delta,output[:,h*w:]],dim=1)

    def close(self):
        for handle in self.handles:handle.remove()


class LoopBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.heads=heads;self.width=width
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width);self.norm3=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,width*3);self.self_output=nn.Linear(width,width)
        self.query=nn.Linear(width,width);self.kv=nn.Linear(width,width*2);self.cross_output=nn.Linear(width,width)
        self.mlp=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
        self.affine=nn.Linear(width,width*2)

    def split(self,tokens):return tokens.reshape(len(tokens),-1,self.heads,self.width//self.heads).transpose(1,2)

    def attention(self,q,k,v):
        return F.scaled_dot_product_attention(self.split(q),self.split(k),self.split(v)).transpose(1,2).flatten(2)

    def forward(self,hidden,reference,condition):
        scale,shift=self.affine(condition).chunk(2,-1)
        value=self.norm1(hidden)*(1+.1*scale.tanh().unsqueeze(1))+.1*shift.tanh().unsqueeze(1)
        q,k,v=self.qkv(value).chunk(3,-1)
        hidden=hidden+.25*self.self_output(self.attention(q,k,v))
        k,v=self.kv(reference).chunk(2,-1)
        hidden=hidden+.25*self.cross_output(self.attention(self.query(self.norm2(hidden)),k,v))
        return hidden+.25*self.mlp(self.norm3(hidden))


class ObservationLoop(nn.Module):
    def __init__(self,width=96,depth=2,heads=4,max_round=16):
        super().__init__();self.width=width;self.max_round=max_round
        if width%heads:raise ValueError('Width must divide heads')
        def encoder():return nn.Sequential(nn.Conv2d(3,width//2,4,stride=4),nn.GELU(),
            nn.Conv2d(width//2,width,4,stride=4),nn.GELU())
        self.current_encoder=encoder();self.lr_encoder=encoder()
        self.position=nn.Linear(2,width,bias=False);self.round_embedding=nn.Embedding(max_round+1,width)
        self.condition=nn.Sequential(nn.Linear(width*3,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList([LoopBlock(width,heads) for _ in range(depth)])
        self.residual_head=nn.Conv2d(width,3*16*16,3,padding=1)
        self.gate_head=nn.Conv2d(width,16*16,1)
        nn.init.zeros_(self.residual_head.weight);nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight);nn.init.zeros_(self.gate_head.bias)

    def forward(self,current,original_lr,rounds,state=None,lr_mode='spatial',use_round=True,use_state=True):
        if current.shape!=original_lr.shape or current.ndim!=4 or current.shape[1]!=3:
            raise ValueError('Paired current/LR must be matching BCHW RGB')
        if any(s%16 for s in current.shape[-2:]):raise ValueError('Image H/W must be divisible by 16')
        r=torch.as_tensor(rounds,device=current.device,dtype=torch.long).reshape(-1)
        if r.numel()==1:r=r.expand(len(current))
        if r.numel()!=len(current) or (r<2).any() or (r>self.max_round).any():raise ValueError('Invalid refinement rounds')
        observed=self.lr_encoder(original_lr);feature=self.current_encoder(current)
        b,c,h,w=feature.shape
        if state is not None and state.shape!=feature.shape:raise ValueError('Hidden state geometry mismatch')
        hidden=feature if state is None or not use_state else feature+.25*state
        reference=observed
        if lr_mode=='weak':reference=F.adaptive_avg_pool2d(observed,(1,1))
        elif lr_mode=='off':reference=torch.zeros_like(observed)
        elif lr_mode!='spatial':raise ValueError('Unknown LR mode')
        hidden=hidden.flatten(2).transpose(1,2)+self.position(positions(h,w,current.device,feature.dtype))
        reference=reference.flatten(2).transpose(1,2)
        if lr_mode!='weak':reference=reference+self.position(positions(h,w,current.device,feature.dtype))
        lr_summary=observed.mean((2,3)) if lr_mode!='off' else torch.zeros_like(observed.mean((2,3)))
        e=self.round_embedding(r) if use_round else torch.zeros_like(lr_summary)
        condition=self.condition(torch.cat([e,feature.mean((2,3)),lr_summary],dim=1))
        for block in self.blocks:hidden=block(hidden,reference,condition)
        next_state=hidden.transpose(1,2).reshape(b,c,h,w)
        delta=.25*F.pixel_shuffle(self.residual_head(next_state),16).float().tanh()
        gate=F.pixel_shuffle(self.gate_head(next_state),16).float().sigmoid()
        result=current.float()+gate*delta
        return dict(image=result,state=next_state,delta=delta,gate=gate)
