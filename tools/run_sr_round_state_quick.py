"""Deadline-bounded, paired 128/32/32 round/state conditioning pilot.

Each stage runs in its own process. No original adapter or installed package is
modified. Later-round generic states are generated anew, never relabeled.
"""
from __future__ import annotations
import argparse, copy, csv, datetime, gc, importlib.util, json, os, random, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from rl_sr.artifact_store import read_jsonl, write_json_atomic, write_jsonl_atomic
from rl_sr.schema import adapter_content_sha256, content_sha256, identity_sha256
from models.sr_round_state_adapter import RoundStateAdapter, AdapterHooks, hold_moe_routing, choose_candidate, gradient_consistency
from tools.run_sr_self_refine_experiment import case_seed

ROOT=Path(__file__).resolve().parents[1]
PROMPT=('Refine the current super-resolved image using the original low-resolution observation '
        'as a fidelity reference. Preserve scene content, geometry, colors, and correct details. '
        'Correct remaining blur, artifacts, and unnatural textures only when supported by the inputs. '
        'Preserve regions that are already restored.')
METRICS=('psnr','ssim','lpips','dists')


def load(path): return json.loads(Path(path).read_text())


def atomic_torch(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); torch.save(value,temp); temp.replace(path)


def image(path,size=None):
    with Image.open(path) as im:
        im=im.convert('RGB')
        if size and im.size!=size: im=im.resize(size,Image.Resampling.BICUBIC)
        a=np.array(im,dtype=np.float32)/127.5-1
    return torch.from_numpy(a).permute(2,0,1)


def save_image(tensor,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    a=np.rint((tensor.detach().float().cpu().permute(1,2,0).clamp(-1,1).numpy()+1)*127.5).astype(np.uint8)
    temp=path.with_suffix('.tmp.png'); Image.fromarray(a).save(temp); temp.replace(path)


@torch.no_grad()
def sample_batch(actor,current,anchor,embeds,condition,mask,confidence,k,seeds,steps=25,sigma_start=1.):
    from rg_flux_fm import build_sigma_schedule
    noises=[torch.randn((1,*current.shape[1:]),device=current.device,dtype=current.dtype,
                        generator=torch.Generator(device=current.device).manual_seed(s)) for s in seeds]
    noise=torch.cat(noises)
    z=noise if sigma_start==1. else (1-sigma_start)*current+sigma_start*noise
    schedule=build_sigma_schedule(z.shape,num_steps=steps,schedule='linear',
        sigma_start=sigma_start,device=z.device,dtype=z.dtype)
    for i in range(steps):
        v=actor(z_t=z,timestep=schedule[i].expand(len(z)),prompt_embeds=embeds['prompt_embeds'].expand(len(z),-1,-1),
            pooled_prompt_embeds=embeds['pooled_prompt_embeds'].expand(len(z),-1),
            text_ids=embeds['text_ids'].expand(len(z),-1,-1),z_lr=current,z_anchor_lr=anchor,
            refinement_round=k,lr_cond_mode='flux2_image_concat',router_condition=condition,
            router_condition_mask=mask,router_condition_confidence=confidence)
        z=z-(schedule[i]-schedule[i+1])*v
    return actor.decode_latents(z).float().clamp(-1,1)


class Pilot:
    def __init__(self,config,run):
        self.cfg=config; self.run=Path(run).resolve(); self.run.mkdir(parents=True,exist_ok=True)
        self.prior=Path(config['reuse_run']); self.metric_models=None
        torch.set_num_threads(4)
        self.deadline=datetime.datetime.fromisoformat(config['deadline']).timestamp()

    def check_time(self):
        if time.time()>self.deadline-180: raise TimeoutError('Pilot deadline reached; report partial results')

    def log(self,message):
        print(f'[{datetime.datetime.now().astimezone().isoformat()}] {message}',flush=True)
        with (self.run/'journal.txt').open('a') as f: f.write(message+'\n')

    def rows(self,name=None):
        split=load(self.run/'split.json')
        return split[name] if name else [r for values in split.values() for r in values]

    def prepare(self):
        if adapter_content_sha256(self.cfg['f0_adapter'])!=self.cfg['f0_sha256']:
            raise ValueError('F0 checkpoint content mismatch')
        split=load(self.prior/'split.json'); audits={r['sample_key']:r for r in load(self.prior/'reuse_audit.json')['f0']}
        groups={}; rows=[]
        for name,n in [('train',128),('validation',32),('test',32)]:
            if len(split[name])!=n: raise ValueError('Expected existing 128/32/32 split')
            groups[name]=set()
            from rl_sr.self_refine import source_group
            for row in split[name]:
                groups[name].add(source_group(row))
                audit=audits[row['_sample_key']]
                if audit['status']!='accepted' or content_sha256(row['_f0_path'])!=audit['sha256']:
                    raise ValueError('F0 reuse SHA mismatch')
                for field in ('lq_path','hq_path','_f0_path'):
                    if not Path(row[field]).is_file(): raise FileNotFoundError(row[field])
                with Image.open(row['_f0_path']) as im:
                    if im.size!=(512,512): raise ValueError('Invalid F0 geometry')
                with Image.open(row['hq_path']) as im:
                    if im.size!=(512,512): raise ValueError('HR must already be aligned 512 crop')
                rows.append(dict(key=row['_sample_key'],split=name,lr_sha256=content_sha256(row['lq_path']),
                    hr_sha256=content_sha256(row['hq_path']),f0_sha256=audit['sha256']))
        if any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
            raise ValueError('Source group overlap')
        write_json_atomic(self.run/'split.json',split)
        # Reuse paired metrics only after output and target content verification.
        old_outputs=read_jsonl(self.prior/'trajectories/A/outputs.jsonl')
        output_index={(r['sample_key'],r['round']):r for r in old_outputs}
        keyed={r['_sample_key']:r for r in self.rows()}
        reused=[]
        with (self.prior/'metrics/A.csv').open() as f:
            for m in csv.DictReader(f):
                row=keyed[m['sample_key']]; out=output_index[(m['sample_key'],int(m['round']))]
                if content_sha256(out['path'])!=m['output_sha256'] or content_sha256(row['hq_path'])!=m['hr_sha256']:
                    raise ValueError('Reused metric content mismatch')
                reused.append({**m,'arm':'personalized','path':out['path']})
        self.write_metrics('personalized',reused)
        write_json_atomic(self.run/'provenance.json',dict(config=self.cfg,rows=rows,
            reused_metric_csv=str(self.prior/'metrics/A.csv'),reused_metric_sha256=content_sha256(self.prior/'metrics/A.csv'),
            generic_prompt=PROMPT,generic_prompt_sha256=identity_sha256(PROMPT),
            protocol='RGB [0,1], full512, official paired PyIQA PSNR/SSIM/LPIPS/DISTS; no border crop',
            limitation='Original-HR-path grouping retained; content duplication/pretraining independence not established'))
        self.log('Prepared immutable 128/32/32 and verified cached first-round/raw personalized metrics')

    def cache(self):
        from diffusers import Flux2KleinPipeline
        self.check_time()
        self.log('Loading text-only Qwen3 pipeline')
        pipeline=Flux2KleinPipeline.from_pretrained(self.cfg['model_path'],transformer=None,vae=None,
            torch_dtype=torch.float32,local_files_only=True)
        device=self.cfg.get('text_device','cpu')
        self.log(f'Text pipeline loaded; encoding generic prompt in FP32 on {device}')
        if device!='cpu': pipeline.to(device)
        with torch.no_grad():
            values=pipeline.encode_prompt(prompt=[PROMPT],device=device,max_sequence_length=512)
        embeds=dict(prompt_embeds=values[0].detach().cpu().to(torch.bfloat16),
            pooled_prompt_embeds=torch.zeros(1,0,dtype=torch.bfloat16),
            text_ids=values[1].detach().cpu().to(torch.bfloat16))
        if embeds['text_ids'].ndim==2: embeds['text_ids']=embeds['text_ids'].unsqueeze(0)
        assert embeds['prompt_embeds'].shape==(1,512,7680)
        assert all(torch.isfinite(v).all() for v in embeds.values())
        atomic_torch(self.run/'generic_text.pt',embeds)
        write_json_atomic(self.run/'generic_text.json',dict(prompt=PROMPT,sha256=content_sha256(self.run/'generic_text.pt'),
            model_path=self.cfg['model_path'],dtype='bf16',online_encoder_dtype='fp32',online_encoder_device=device,max_sequence_length=512))
        self.log('Encoded one generic prompt with original FP32 Qwen3; cached BF16 [1,512,7680]')

    def runtime(self):
        config=yaml.safe_load(Path(self.cfg['base_config']).read_text())
        config['text_encoding']['mode']='cached'
        config['model']['gradient_checkpointing']=True
        config['rl_sr']['image_loss']['lpips_weight']=0.
        return config

    def spatial_losses(self,model,z0,hq):
        from train_rg_flux_sr import crop_image_loss_inputs
        with torch.autocast('cuda',enabled=False):
            latent,target,_,_=crop_image_loss_inputs(z0.float(),hq.float(),hq.float(),
                {'loss':{'image_loss_crop_size':256}})
            prediction=model.decode_latents_for_loss(latent.float()).float()
            charb=torch.sqrt((prediction-target.float()).square()+1e-6).mean()
            gradient=gradient_consistency(prediction,target.float())
        return dict(loss_image=.1*charb+.05*gradient,loss_charb=charb,loss_gradient=gradient,
                    loss_lpips=charb.new_zeros(()))

    def artist(self,mode=None,checkpoint=None,state=False):
        from models.rg_flux_artist_factory import build_rg_flux_artist
        torch.manual_seed(42)
        model=build_rg_flux_artist(self.runtime()).to(device='cuda')
        model.load_trainable(self.cfg['f0_adapter'],is_trainable=False)
        model.align_inference_dtype(dtype=torch.bfloat16)
        model.requires_grad_(False); model.eval(); model.set_moe_inference_schedule()
        adapter=hooks=None
        if mode:
            width=3072 if mode=='temb' else 7680
            adapter=RoundStateAdapter(width,max_state_delta=self.cfg['state_max_delta']).to('cuda')
            adapter.requires_grad_(False)
            if checkpoint:
                payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
                if payload['f0_sha256']!=self.cfg['f0_sha256'] or payload['mode']!=mode:
                    raise ValueError('Adapter checkpoint provenance mismatch')
                adapter.load_state_dict(payload['adapter'])
                model.refinement_condition_adapter.load_state_dict(payload['anchor'])
            hooks=AdapterHooks(model,adapter,mode=mode,state=state)
            # Newly initialized LR anchor must accumulate small updates in FP32.
            model.refinement_condition_adapter.to(dtype=torch.float32)
            # Existing zero-gated legacy round addition otherwise promotes the
            # entire text stream to FP32 outside training autocast.
            model.refinement_condition_adapter.round_embed.to(dtype=torch.bfloat16)
        return model,adapter,hooks

    def embeds(self):
        meta=load(self.run/'generic_text.json')
        if meta['sha256']!=content_sha256(self.run/'generic_text.pt'): raise ValueError('Generic text cache changed')
        return {k:v.to('cuda') for k,v in torch.load(self.run/'generic_text.pt',weights_only=True).items()}

    def latent(self,model,path,size=None):
        key=identity_sha256(dict(sha256=content_sha256(path),size=size,vae='F0-packed128-FP32-deterministic'))
        dest=self.run/'latent_cache'/f'{key}.pt'
        if dest.exists():
            result=torch.load(dest,map_location='cpu',weights_only=True)
            if result.shape!=(128,32,32) or result.dtype!=torch.bfloat16 or not torch.isfinite(result).all():
                dest.unlink()
            else: return result
        with torch.no_grad(): result=model.encode_images(image(path,size).unsqueeze(0).to('cuda'),sample=False)[0].cpu().to(torch.bfloat16)
        if result.shape!=(128,32,32) or not torch.isfinite(result).all(): raise ValueError('Invalid latent')
        atomic_torch(dest,result)
        return result

    def generate(self,model,arm,rows,embeds,max_round=4):
        from rl_sr.conditioning import state_router_condition_tensors
        folder=self.run/'trajectories'/arm; folder.mkdir(parents=True,exist_ok=True)
        outputs=[dict(arm=arm,sample_key=r['_sample_key'],split=r['_split'],round=1,path=r['_f0_path']) for r in rows]
        previous={r['_sample_key']:r['_f0_path'] for r in rows}
        actor_file=getattr(self,'active_checkpoint',None)
        policy=content_sha256(actor_file) if actor_file else self.cfg['f0_sha256']
        for k in range(2,max_round+1):
            for start in range(0,len(rows),self.cfg['inference_batch']):
                self.check_time(); batch=rows[start:start+self.cfg['inference_batch']]; pending=[]
                for row in batch:
                    key=row['_sample_key']; path=folder/f'{key}_r{k}.png'; meta=path.with_suffix('.json')
                    expected=dict(policy_sha256=policy,mode=getattr(self,'active_mode',None),
                        prompt_sha256=identity_sha256(PROMPT),parent_sha256=content_sha256(previous[key]),
                        lr_sha256=content_sha256(row['lq_path']),round=k,nfe=25,
                        seed=case_seed(key,k,42),sigma_start=getattr(self,'active_sigma_start',1.))
                    if path.exists() and meta.exists():
                        old=load(meta)
                        if all(old.get(a)==b for a,b in expected.items()) and old['output_sha256']==content_sha256(path):
                            continue
                        path.unlink(); meta.unlink()
                    pending.append((row,path,expected))
                if pending:
                    current=torch.stack([self.latent(model,previous[r['_sample_key']]) for r,_,_ in pending]).to('cuda')
                    anchor=torch.stack([self.latent(model,r['lq_path'],size=(512,512)) for r,_,_ in pending]).to('cuda')
                    cond,mask,conf=state_router_condition_tensors([r['_router_condition'] for r,_,_ in pending],self.runtime(),'cuda',torch.bfloat16)
                    with torch.no_grad(): result=sample_batch(model,current,anchor,embeds,cond,mask,conf,k,
                        [m['seed'] for _,_,m in pending],sigma_start=getattr(self,'active_sigma_start',1.))
                    for (row,path,meta),value in zip(pending,result):
                        save_image(value,path); write_json_atomic(path.with_suffix('.json'),{**meta,'output_sha256':content_sha256(path)})
                for row in batch:
                    key=row['_sample_key']; path=folder/f'{key}_r{k}.png'
                    outputs.append(dict(arm=arm,sample_key=key,split=row['_split'],round=k,path=str(path)))
                    previous[key]=str(path)
                if start%32==0: self.log(f'{arm}: round{k}, {min(start+len(batch),len(rows))}/{len(rows)} sources')
        write_jsonl_atomic(folder/'outputs.jsonl',outputs)
        return outputs

    def collect_generic(self):
        model,_,_=self.artist(); embeds=self.embeds()
        self.active_checkpoint=None; self.active_mode=None
        # Train needs y1,y2,y3 states; heldout also needs raw y4 curves.
        self.generate(model,'generic_train',self.rows('train'),embeds,max_round=3)
        heldout=self.rows('validation')+self.rows('test')
        outputs=self.generate(model,'generic',heldout,embeds)
        del model; gc.collect(); torch.cuda.empty_cache()
        self.score('generic',outputs)

    def metrics(self):
        if self.metric_models is None:
            import pyiqa
            paths={}
            for name,module,relative in [('lpips','lpips','weights/v0.1/alex.pth'),('dists','DISTS_pytorch','weights.pt')]:
                spec=importlib.util.find_spec(module)
                p=Path(spec.origin).parent/relative
                if p.is_file(): paths[name]=p
            self.metric_models={n:pyiqa.create_metric(n,device='cpu',
                **({'pretrained_model_path':str(paths[n])} if n in paths else {}),
                **({'test_y_channel':False} if n in ('psnr','ssim') else {})) for n in METRICS}
            write_json_atomic(self.run/'metric_weights.json',{k:dict(path=str(v),sha256=content_sha256(v)) for k,v in paths.items()})
        return self.metric_models

    def write_metrics(self,arm,rows):
        folder=self.run/'metrics'; folder.mkdir(exist_ok=True)
        fields=['arm','sample_key','split','round','path','output_sha256','hr_sha256',*METRICS]
        temp=folder/f'{arm}.csv.tmp'
        with temp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
        temp.replace(folder/f'{arm}.csv')

    def score(self,arm,outputs):
        cache=self.run/'metrics'/f'{arm}.csv'; prior={}
        if cache.exists():
            with cache.open() as f: prior={(r['sample_key'],int(r['round'])):r for r in csv.DictReader(f)}
        sources={r['_sample_key']:r for r in self.rows()}; result=[]
        for i,out in enumerate(outputs):
            self.check_time(); row=sources[out['sample_key']]
            if row['_split']=='train': continue
            sha=content_sha256(out['path']); gtsha=content_sha256(row['hq_path'])
            old=prior.get((out['sample_key'],out['round']))
            if old and old['output_sha256']==sha and old['hr_sha256']==gtsha:
                result.append(old); continue
            a=(image(out['path']).unsqueeze(0)+1)/2; b=(image(row['hq_path']).unsqueeze(0)+1)/2
            with torch.no_grad(): values={n:float(m(a,b).item()) for n,m in self.metrics().items()}
            if not all(np.isfinite(v) for v in values.values()): raise FloatingPointError('Nonfinite paired metric')
            result.append({**out,'output_sha256':sha,'hr_sha256':gtsha,**values})
            self.write_metrics(arm,result)
        self.write_metrics(arm,result)
        return result

    def average(self,rows,split='validation',k=2):
        rows=[r for r in rows if r['split']==split and int(r['round'])==k]
        if not rows: raise ValueError('Missing evaluation rows')
        return {n:float(np.mean([float(r[n]) for r in rows])) for n in METRICS}

    def states(self,arm):
        rows=read_jsonl(self.run/'trajectories'/arm/'outputs.jsonl')
        index={(r['sample_key'],r['round']):r['path'] for r in rows}
        return [dict(source=r,round=k,current=index[(r['_sample_key'],k-1)]) for r in self.rows('train') for k in (2,3,4)]

    def save(self,path,model,adapter,mode,update,optimizer=None):
        value=dict(f0_sha256=self.cfg['f0_sha256'],mode=mode,update=update,
            adapter={k:v.detach().cpu() for k,v in adapter.state_dict().items()},
            anchor={k:v.detach().cpu() for k,v in model.refinement_condition_adapter.state_dict().items()})
        if optimizer is not None:
            value.update(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        atomic_torch(path,value)

    def train_cycle(self,model,adapter,hooks,mode,state_rows,cycle,arm,state=False,updates_per_cycle=50):
        from rl_sr.conditioning import state_router_condition_tensors
        from rg_flux_fm import build_flow_matching_inputs
        model.eval(); model.requires_grad_(False); adapter.requires_grad_(False)
        if state:
            trainable=[(n,p) for n,p in adapter.named_parameters() if n.startswith('state_')]
            groups=[dict(params=[p for _,p in trainable],lr=1e-3)]
            model.moe_routing_mode='soft' # top-2 remains the evaluation policy
        else:
            trainable=[(n,p) for n,p in adapter.named_parameters() if n.startswith('round_')]
            anchor=[(n,p) for n,p in model.refinement_condition_adapter.named_parameters() if n.startswith('anchor_')]
            trainable+=anchor
            groups=[dict(params=[p for n,p in trainable if n.startswith('round_')],lr=1e-3),
                    dict(params=[p for _,p in anchor],lr=1e-4)]
        for _,p in trainable: p.requires_grad_(True)
        optimizer=torch.optim.AdamW(groups,weight_decay=.01)
        embeds=self.embeds()
        folder=self.run/'training'/arm; folder.mkdir(parents=True,exist_ok=True)
        write_json_atomic(folder/'parameter_groups.json',dict(trainable=[dict(name=n,count=p.numel()) for n,p in trainable],
            total=sum(p.numel() for _,p in trainable),lora_frozen=True,base_router_frozen=True))
        buckets={k:[s for s in state_rows if s['round']==k] for k in (2,3,4)}
        start=cycle*updates_per_cycle; records=[]
        resume=folder/f'resume_cycle{cycle}.pt'
        if resume.exists():
            payload=torch.load(resume,map_location='cpu',weights_only=True)
            if payload['f0_sha256']!=self.cfg['f0_sha256'] or payload['mode']!=mode:
                raise ValueError('Resume mode/F0 differs')
            adapter.load_state_dict(payload['adapter']); model.refinement_condition_adapter.load_state_dict(payload['anchor'])
            optimizer.load_state_dict(payload['optimizer']); start=payload['update']
            # CPU load requires explicitly migrating optimizer state back to GPU.
            for state_value in optimizer.state.values():
                for key,value in state_value.items():
                    if torch.is_tensor(value) and key!='step': state_value[key]=value.to('cuda')
        torch.manual_seed(42+start); torch.cuda.manual_seed_all(42+start)
        histories=load(folder/'loss_history.json') if (folder/'loss_history.json').exists() else []
        for step in range(start+1,(cycle+1)*updates_per_cycle+1):
            self.check_time(); bsz=int(self.cfg['train_batch'])
            local_rng=random.Random(42+step)
            batch=[local_rng.choice(buckets[2+i%3]) for i in range(bsz)]; local_rng.shuffle(batch)
            hr=torch.stack([self.latent(model,s['source']['hq_path']) for s in batch]).to('cuda')
            current=torch.stack([self.latent(model,s['current']) for s in batch]).to('cuda')
            anchor_z=torch.stack([self.latent(model,s['source']['lq_path'],(512,512)) for s in batch]).to('cuda')
            hq=torch.stack([image(s['source']['hq_path']) for s in batch]).to('cuda')
            # Seed per update makes retries reproduce noise and identity sampling.
            torch.manual_seed(42+step); torch.cuda.manual_seed_all(42+step)
            identity=torch.rand(bsz,device='cuda')<.1; current[identity]=hr[identity]
            sigma=torch.rand(bsz,device='cuda',dtype=torch.float32).to(torch.bfloat16)
            zt,target=build_flow_matching_inputs(hr,sigma=sigma)
            rounds=torch.tensor([s['round'] for s in batch],device='cuda')
            cond,mask,conf=state_router_condition_tensors([s['source']['_router_condition'] for s in batch],self.runtime(),'cuda',torch.bfloat16)
            kwargs=dict(z_t=zt,timestep=sigma,prompt_embeds=embeds['prompt_embeds'].expand(bsz,-1,-1),
                pooled_prompt_embeds=embeds['pooled_prompt_embeds'].expand(bsz,-1),text_ids=embeds['text_ids'].expand(bsz,-1,-1),
                z_lr=current,z_anchor_lr=anchor_z,refinement_round=rounds,lr_cond_mode='flux2_image_concat',
                router_condition=cond,router_condition_mask=mask,router_condition_confidence=conf)
            optimizer.zero_grad(set_to_none=True)
            totals={n:0. for n in ('loss','fm','latent','charb','gradient','lpips','route_kl')}
            micro=int(self.cfg.get('micro_batch',bsz))
            for offset in range(0,bsz,micro):
                stop=min(offset+micro,bsz); fraction=(stop-offset)/bsz
                local={n:(v[offset:stop] if torch.is_tensor(v) and v.ndim>0 and v.shape[0]==bsz else v)
                       for n,v in kwargs.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    predicted=model(**local)
                    dispatch=model._last_moe_router_output['alpha_used']; route=hooks.last_route
                    z0=zt[offset:stop].float()-sigma[offset:stop].float().view(-1,1,1,1)*predicted.float()
                    fm=F.mse_loss(predicted.float(),target[offset:stop].float())
                    latent_loss=F.smooth_l1_loss(z0,hr[offset:stop].float())
                    losses=self.spatial_losses(model,z0,hq[offset:stop])
                    route_kl=route['kl'] if route else fm.new_zeros(())
                    loss=fm+.1*latent_loss+losses['loss_image']+.01*route_kl
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                with hold_moe_routing(model.transformer,dispatch): (loss*fraction).backward()
                measured=dict(loss=loss,fm=fm,latent=latent_loss,charb=losses['loss_charb'],
                              gradient=losses['loss_gradient'],lpips=losses['loss_lpips'],route_kl=route_kl)
                for n,value in measured.items(): totals[n]+=float(value.detach())*fraction
            norm=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.)
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient norm')
            optimizer.step()
            if any(not torch.isfinite(p).all() for _,p in trainable): raise FloatingPointError('Nonfinite adapter')
            history=dict(update=step,cycle=cycle,batch=bsz,micro_batch=micro,**totals,
                grad_norm=float(norm),round_gate=float(adapter.round_gate.detach()),
                state_gate=float(adapter.state_gate.detach()),anchor_gate=float(model.refinement_condition_adapter.anchor_gate.detach()),
                identity_count=int(identity.sum()),memory_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            histories=[r for r in histories if r['update']!=step]+[history]
            write_json_atomic(folder/'loss_history.json',histories)
            if step%5==0: self.save(resume,model,adapter,mode,step,optimizer)
            if step%25==0:
                path=folder/f'checkpoint-{step:04d}.pt'; self.save(path,model,adapter,mode,step); records.append(path)
            if step%5==0: self.log(f'{arm}: update {step}/100, loss {history["loss"]:.5f}, reserved {history["memory_reserved_gib"]:.1f} GiB')
        model.requires_grad_(False); adapter.requires_grad_(False); model.set_moe_inference_schedule()
        return sorted(folder.glob('checkpoint-*.pt'))

    def train_arm(self,mode,state=False):
        arm='state' if state else mode
        base=load(self.run/'round_choice.json')['checkpoint'] if state else None
        model,adapter,hooks=self.artist(mode,base,state=state)
        all_scores=[]; checkpoint_files=[]; embeds=self.embeds()
        cycle_limit=1 if state else 2
        state_arm='generic_train'
        for cycle in range(cycle_limit):
            state_rows=self.states(state_arm)
            files=self.train_cycle(model,adapter,hooks,mode,state_rows,cycle,arm,state=state)
            checkpoint_files=files
            for path in files:
                if any(r['path']==str(path) for r in all_scores): continue
                payload=torch.load(path,map_location='cpu',weights_only=True)
                adapter.load_state_dict(payload['adapter']); model.refinement_condition_adapter.load_state_dict(payload['anchor'])
                self.active_checkpoint=path; self.active_mode=mode
                name=f'{arm}_check{payload["update"]}'
                outputs=self.generate(model,name,self.rows('validation')[:8],embeds,max_round=2)
                metrics=self.score(name,outputs)
                all_scores.append(dict(path=str(path),update=payload['update'],metrics=self.average(metrics)))
                write_json_atomic(self.run/f'{arm}_candidate_scores.json',all_scores)
            best=min(all_scores,key=lambda r:r['metrics']['lpips']+r['metrics']['dists'])
            payload=torch.load(best['path'],map_location='cpu',weights_only=True)
            adapter.load_state_dict(payload['adapter']); model.refinement_condition_adapter.load_state_dict(payload['anchor'])
            if cycle+1<cycle_limit:
                self.active_checkpoint=Path(best['path']); self.active_mode=mode
                state_arm=f'{arm}_refresh{cycle}'
                self.generate(model,state_arm,self.rows('train'),embeds,max_round=3)
        best=min(all_scores,key=lambda r:r['metrics']['lpips']+r['metrics']['dists'])
        payload=torch.load(best['path'],map_location='cpu',weights_only=True)
        adapter.load_state_dict(payload['adapter']); model.refinement_condition_adapter.load_state_dict(payload['anchor'])
        self.active_checkpoint=Path(best['path']); self.active_mode=mode
        # Raw candidate is evaluated even if it will be rejected. No controller.
        outputs=self.generate(model,arm,self.rows('validation')+self.rows('test'),embeds,max_round=4)
        hooks.close(); del model,adapter,hooks; gc.collect(); torch.cuda.empty_cache()
        metrics=self.score(arm,outputs)
        write_json_atomic(self.run/f'{arm}_result.json',dict(mode=mode,state_routing=state,checkpoint=best['path'],
            calibration=best['metrics'],validation_round2=self.average(metrics),
            checkpoint_selection='8 validation cases only; final 32 validation and 32 test reported separately'))

    def conditional_state(self):
        choices=[]
        with (self.run/'metrics/personalized.csv').open() as f: f0=self.average(list(csv.DictReader(f)),k=1)
        for mode in ('text','temb'):
            info=load(self.run/f'{mode}_result.json')
            with (self.run/f'metrics/{mode}.csv').open() as f: rows=list(csv.DictReader(f))
            choices.append((mode,info,rows))
        picked=min(choices,key=lambda r:r[1]['validation_round2']['lpips']+r[1]['validation_round2']['dists'])
        decision=choose_candidate(f0,[(picked[0],picked[1]['validation_round2'])])
        gain=self.average(picked[2],k=2); last=self.average(picked[2],k=4)
        recurse=last['lpips']<=gain['lpips']-.001 and last['dists']<=gain['dists']-.001
        enough=time.time()<self.deadline-4*3600
        write_json_atomic(self.run/'round_choice.json',dict(mode=picked[0],checkpoint=picked[1]['checkpoint'],
            decision=decision,recursive_gain=bool(recurse),time_for_state=bool(enough)))
        if decision['selected'] and recurse and enough:
            self.train_arm(picked[0],state=True)
        else:
            write_json_atomic(self.run/'state_result.json',dict(status='skipped',reason=
                'Raw candidate must pass F0 fidelity gate AND show recursive LPIPS/DISTS gain; also reserve 4h before deadline',
                f0_gate=bool(decision['selected']),recursive_gain=bool(recurse),time_for_state=bool(enough)))
            self.log('State routing deferred: candidate quality/time gate not satisfied; report failure rather than scale')

    def warmstart_if_needed(self):
        decision=load(self.run/'round_choice.json')
        if decision['decision']['selected'] or time.time()>self.deadline-3*3600:
            write_json_atomic(self.run/'warmstart_result.json',dict(status='skipped',reason='Round candidate passed, or less than 3h remains'))
            return
        self.log('Round training failed fidelity gate; screen previous-SR noise initialization without further training')
        model,_,_=self.artist(); embeds=self.embeds(); self.active_checkpoint=None; self.active_mode=None
        scores=[]
        for s in (.1,.2,.4):
            self.active_sigma_start=s
            name=f'warmstart_screen_s{s}'
            outputs=self.generate(model,name,self.rows('validation')[:8],embeds,max_round=2)
            metrics=self.score(name,outputs)
            scores.append(dict(sigma_start=s,validation_round2=self.average(metrics)))
        best=min(scores,key=lambda r:r['validation_round2']['lpips']+r['validation_round2']['dists'])
        self.active_sigma_start=best['sigma_start']
        outputs=self.generate(model,'warmstart',self.rows('validation')+self.rows('test'),embeds,max_round=4)
        del model; gc.collect(); torch.cuda.empty_cache()
        metrics=self.score('warmstart',outputs)
        write_json_atomic(self.run/'warmstart_result.json',dict(status='completed',selected=best,screens=scores,
            validation_round2=self.average(metrics),kind='Inference initialization ablation; F0 weights unchanged, not evidence of successful round training',
            nfe=25,schedule='linear sigma_start to zero',initialization='(1-s)*E(previous_SR)+s*noise'))

    def report(self):
        summaries=[]; full=[]
        for path in sorted((self.run/'metrics').glob('*.csv')):
            if '_check' in path.stem or '_screen' in path.stem: continue
            with path.open() as f: rows=list(csv.DictReader(f))
            full+=rows
            for split in ('validation','test'):
                for k in (1,2,3,4):
                    selected=[r for r in rows if r['split']==split and int(r['round'])==k]
                    if selected: summaries.append(dict(arm=path.stem,split=split,round=k,count=len(selected),
                        **{n:float(np.mean([float(r[n]) for r in selected])) for n in METRICS}))
        path=self.run/'round_metrics_summary.csv'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['arm','split','round','count',*METRICS]); w.writeheader(); w.writerows(summaries)
        results={p.stem:load(p) for p in self.run.glob('*_result.json')}
        paired=[]
        original={(r['sample_key'],int(r['round'])):r for r in full if r['arm']=='personalized'}
        for arm in sorted({r['arm'] for r in full}):
            if arm=='personalized': continue
            for split in ('validation','test'):
                for baseline_round in (1,4):
                    rows=[r for r in full if r['arm']==arm and r['split']==split and int(r['round'])==4]
                    if not rows: continue
                    item=dict(arm=arm,split=split,baseline=f'personalized_round{baseline_round}',count=len(rows))
                    for n in METRICS:
                        sign=1 if n in ('psnr','ssim') else -1
                        differences=np.array([sign*(float(r[n])-float(original[(r['sample_key'],baseline_round)][n])) for r in rows])
                        rng=np.random.default_rng(42)
                        draws=differences[rng.integers(0,len(rows),(2000,len(rows)))].mean(axis=1)
                        item[n]=dict(mean_gain=float(differences.mean()),ci95=[float(v) for v in np.quantile(draws,[.025,.975])],
                                     regression_fraction=float((differences<0).mean()))
                    paired.append(item)
        write_json_atomic(self.run/'paired_bootstrap.json',paired)
        write_json_atomic(self.run/'results.json',dict(summary=summaries,results=results,
            limitation='Small feasibility pilot. Original LoRA frozen. V3 test split was previously evaluated; exploratory, not a fresh paper test set.',
            raw_candidates=True,deadline=self.cfg['deadline']))
        self.plots(summaries)
        self.comparison_images(full)
        lines=['# V4 quick validation','',f'Deadline: {self.cfg["deadline"]}',
            '','128 training sources; 32 validation; 32 test. F0 first round fixed. Generic prompt from round 2.',
            'Previous SR remains native image condition; original LR remains spatial anchor.',
            'Original LoRA/router/condition modules frozen except newly initialized spatial anchor branch; new round/state parameters trained.',
            '100 updates in two own-trajectory cycles per round arm; 25-NFE raw output evaluation; no controller/RL.',
            'Loss: FM + 0.1 latent SmoothL1 + 0.1 HR Charbonnier + 0.05 HR gradient consistency; optional state routing KL 0.01. No LPIPS/PSNR/SSIM/DISTS training loss.',
            'Text/temb embedding parameter counts differ; this is not a parameter-count-matched architecture claim.','',
            '| Arm | Split | Round | N | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in summaries:
            lines.append(f'| {r["arm"]} | {r["split"]} | {r["round"]} | {r["count"]} | '+
                ' | '.join(f'{r[n]:.5f}' for n in METRICS)+' |')
        lines+=['','## Decisions','',json.dumps(results,ensure_ascii=False,indent=2),
            '','## Interpretation','',
            'Raw candidates are reported even when rejected. Similarity to F0 caused by fallback is not SR improvement.',
            'Compare generic vs personalized; text vs temb plus the shared new LR anchor; state router only after candidate feasibility gate.',
            'No expanded training is started automatically. A failed short pilot can reflect insufficient optimization or architecture limitations.']
        (self.run/'report.md').write_text('\n'.join(lines)+'\n')
        self.log(f'Report updated with {len(summaries)} round/split summaries')

    def plots(self,summaries):
        if not summaries:return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for ax,n in zip(axes.flat,METRICS):
            for arm in sorted({r['arm'] for r in summaries}):
                rows=sorted([r for r in summaries if r['arm']==arm and r['split']=='validation'],key=lambda r:r['round'])
                ax.plot([r['round'] for r in rows],[r[n] for r in rows],marker='o',label=arm)
            ax.set_title(n); ax.set_xticks([1,2,3,4]); ax.grid(alpha=.2)
        axes.flat[0].legend(); fig.tight_layout(); fig.savefig(self.run/'metric_curves.png',dpi=160); plt.close(fig)

    def comparison_images(self,full):
        arms=[a for a in ('generic','text','temb','state','warmstart') if any(r['arm']==a for r in full)]
        rows=self.rows('validation') if (self.run/'split.json').exists() else []
        if not rows:return
        lookup={(r['arm'],r['sample_key'],int(r['round'])):r['path'] for r in full}
        cases=[rows[i] for i in (0,7,15)]
        width=256*(3+len(arms)); canvas=Image.new('RGB',(width,3*282),'white')
        from PIL import ImageDraw
        draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(cases):
            key=row['_sample_key']
            columns=[('HR',row['hq_path']),('F0 round1',row['_f0_path']),
                     ('personalized round4',lookup.get(('personalized',key,4)))]
            columns += [(f'{a} round4',lookup.get((a,key,4))) for a in arms]
            for j,(label,path) in enumerate(columns):
                draw.text((j*256+5,i*282+4),label,fill='black')
                if path and Path(path).exists():
                    with Image.open(path) as im: canvas.paste(im.convert('RGB').resize((256,256)),(j*256,i*282+24))
        canvas.save(self.run/'image_comparison.png')


def main():
    import faulthandler
    faulthandler.dump_traceback_later(120,repeat=True)
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--run',required=True)
    p.add_argument('--stage',required=True,choices=['prepare','cache','collect_generic','train_text','train_temb','conditional_state','warmstart_if_needed','report','smoke'])
    args=p.parse_args(); pilot=Pilot(yaml.safe_load(Path(args.config).read_text()),args.run)
    if args.stage.startswith('train_'): pilot.train_arm(args.stage.split('_')[1])
    elif args.stage=='smoke': smoke(pilot)
    else: getattr(pilot,args.stage)()


def smoke(pilot):
    model,adapter,hooks=pilot.artist('temb'); row=pilot.rows('train')[0]
    current=pilot.latent(model,row['_f0_path']).unsqueeze(0).to('cuda')
    anchor=pilot.latent(model,row['lq_path'],(512,512)).unsqueeze(0).to('cuda')
    from rl_sr.conditioning import state_router_condition_tensors
    cond,mask,conf=state_router_condition_tensors([row['_router_condition']],pilot.runtime(),'cuda',torch.bfloat16)
    embeds=pilot.embeds(); sigma=torch.tensor([.5],device='cuda',dtype=torch.bfloat16)
    kwargs=dict(z_t=current,timestep=sigma,**embeds,z_lr=current,z_anchor_lr=anchor,refinement_round=2,
        lr_cond_mode='flux2_image_concat',router_condition=cond,router_condition_mask=mask,router_condition_confidence=conf)
    for n,p in adapter.named_parameters(): p.requires_grad_(n.startswith('round_'))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        baseline=model(**{**kwargs,'refinement_round':None,'z_anchor_lr':None})
        initial=model(**kwargs)
    torch.testing.assert_close(initial,baseline,rtol=0,atol=0)
    with torch.autocast('cuda',dtype=torch.bfloat16): result=model(**kwargs)
    alpha=model._last_moe_router_output['alpha_used']
    with hold_moe_routing(model.transformer,alpha): result.float().square().mean().backward()
    assert adapter.round_gate.grad is not None and torch.isfinite(adapter.round_gate.grad)
    assert model.moe_router.condition_proj[1].weight.grad is None
    del result,baseline,initial
    rows=pilot.rows('train')[:6]
    state_rows=[dict(source=r,round=k,current=r['_f0_path']) for r in rows for k in (2,3,4)]
    # Two actual full-loss updates at the configured effective batch size.
    pilot.train_cycle(model,adapter,hooks,'temb',state_rows,0,'smoke',updates_per_cycle=2)
    assert float(adapter.round_gate.detach())!=0
    pilot.save(pilot.run/'smoke_checkpoint.pt',model,adapter,'temb',2)
    restored=RoundStateAdapter(3072).load_state_dict(torch.load(pilot.run/'smoke_checkpoint.pt',weights_only=True)['adapter'])
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    batch=int(pilot.cfg['inference_batch'])
    inference_start=time.monotonic()
    with torch.no_grad():
        output=sample_batch(model,current.expand(batch,-1,-1,-1),anchor.expand(batch,-1,-1,-1),embeds,
            cond.expand(batch,-1),mask.expand(batch,-1),conf.expand(batch),2,list(range(batch)))
    torch.cuda.synchronize()
    assert output.shape==(batch,3,512,512) and torch.isfinite(output).all()
    inference_seconds=time.monotonic()-inference_start
    write_json_atomic(pilot.run/'smoke.json',dict(status='passed',output_shape=[1,128,32,32],
        round_gate_gradient=float(adapter.round_gate.grad),max_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        frozen_base_router=True,zero_gate_exact_forward=True,full_loss_updates=2,effective_batch=pilot.cfg['train_batch'],
        micro_batch=pilot.cfg['micro_batch'],checkpoint_roundtrip=True,checkpoint=str(pilot.run/'smoke_checkpoint.pt'),
        inference_batch=batch,inference_nfe=25,inference_seconds=inference_seconds,
        inference_seconds_per_image=inference_seconds/batch,
        training_peak_reserved_gib=max(r['memory_reserved_gib'] for r in load(pilot.run/'training/smoke/loss_history.json'))))
    pilot.log('Real A100 forward/backward/checkpoint smoke passed')


if __name__=='__main__': main()
