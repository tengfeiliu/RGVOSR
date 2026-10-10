"""One resumable stage of a larger, observation-constrained SR experiment."""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import gc
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import yaml

from rl_sr.artifact_store import read_jsonl,write_json_atomic,write_jsonl_atomic
from rl_sr.observation_loop_data import prepare_run
from rl_sr.schema import content_sha256,identity_sha256
from tools.run_sr_round_state_quick import Pilot,PROMPT,atomic_torch,image,save_image,sample_batch
from tools.run_sr_self_refine_experiment import case_seed

ROOT=Path(__file__).resolve().parents[1]


class Experiment(Pilot):
    def __init__(self,cfg,run):
        self.cfg=cfg;self.run=Path(run).resolve();self.run.mkdir(parents=True,exist_ok=True)
        self.metric_models=None;torch.set_num_threads(4)

    def check_time(self):
        if (self.run/'user_stop.json').exists():raise RuntimeError('User requested this experiment stop')

    def split_sha(self):return content_sha256(self.run/'split_core.json')

    def rows(self,name=None):
        split=json.loads((self.run/'split.json').read_text())
        core={k:[{a:b for a,b in r.items() if a!='_f0_path'} for r in v] for k,v in split.items()}
        if core!=json.loads((self.run/'split_core.json').read_text()):raise ValueError('Split identity changed')
        return split[name] if name else [r for values in split.values() for r in values]

    def prepare(self):
        result=prepare_run(self.cfg,self.run)
        self.log(f'Prepared {result["counts"]}, verified cached training F0={result["reused_training_f0"]}')

    def runtime(self):
        config=yaml.safe_load(Path(self.cfg['base_config']).read_text())
        config['text_encoding'].update(mode='cached',cache_dir=str(self.run/'text_cache'),dtype='bf16',strict=True)
        config['model']['gradient_checkpointing']=True
        config['rl_sr']['image_loss']['lpips_weight']=0.
        return config

    def artist(self):
        from models.rg_flux_artist_factory import build_rg_flux_artist
        torch.manual_seed(42)
        model=build_rg_flux_artist(self.runtime()).to(device='cuda')
        model.load_trainable(self.cfg['f0_adapter'],is_trainable=False)
        model.align_inference_dtype(dtype=torch.bfloat16);model.requires_grad_(False)
        model.eval();model.set_moe_inference_schedule()
        return model

    def cache_text(self):
        from models.text_embedding_cache import TextEmbeddingCache,validate_online_prompt_lengths
        from models.flux2_klein_sr_artist import Flux2KleinSRArtist
        from diffusers import Flux2KleinPipeline
        from types import SimpleNamespace
        prior=Path(self.cfg['generic_cache_run']);meta=json.loads((prior/'generic_text.json').read_text())
        if (meta['sha256']!=content_sha256(prior/'generic_text.pt') or meta['prompt']!=PROMPT
            or Path(meta['model_path']).resolve()!=Path(self.cfg['model_path']).resolve()):
            raise ValueError('Generic prompt cache integrity differs')
        shutil.copyfile(prior/'generic_text.pt',self.run/'generic_text.pt')
        write_json_atomic(self.run/'generic_text.json',{**meta,'reused_from':str(prior)})
        config=self.runtime();cache=TextEmbeddingCache(self.run/'text_cache',config,dtype='bf16')
        rows=self.rows('validation')+self.rows('test')
        pending=[r for r in rows if not cache.find_record(r['_prompt'],r['_sample_key'],validate_file=True)]
        if not pending:return
        pipe=Flux2KleinPipeline.from_pretrained(self.cfg['model_path'],transformer=None,vae=None,
            torch_dtype=torch.float32,local_files_only=True).to('cuda');pipe.text_encoder.requires_grad_(False);pipe.text_encoder.eval()
        proxy=SimpleNamespace(text_pipeline=pipe,text_encoder_device='cuda',max_prompt_sequence_length=512,weight_dtype=torch.bfloat16)
        proxy.prompt_token_lengths=lambda prompts:Flux2KleinSRArtist.prompt_token_lengths(proxy,prompts)
        audit=[]
        for i,row in enumerate(pending):
            self.check_time();validate_online_prompt_lengths(proxy,[row['_prompt']],config)
            values=Flux2KleinSRArtist.encode_prompts(proxy,[row['_prompt']],device='cpu',dtype=torch.bfloat16)
            if values[0].shape!=(1,512,7680) or not all(torch.isfinite(v).all() for v in values):
                raise ValueError('Invalid personalized prompt cache')
            record=cache.save_embedding(row['_prompt'],row['_sample_key'],row['lq_path'],row['hq_path'],
                dict(zip(('prompt_embeds','pooled_prompt_embeds','text_ids'),values)))
            audit.append(dict(key=row['_sample_key'],tokens=proxy.prompt_token_lengths([row['_prompt']])[0]))
            if i%16==0:self.log(f'New heldout prompt cache {i+1}/{len(pending)} (Qwen3 FP32, length512)')
        write_json_atomic(self.run/'prompt_audit.json',dict(encoded=audit,shape=[1,512,7680],dtype='bf16',encoder_dtype='fp32'))

    def f0(self):
        from models.text_embedding_cache import TextEmbeddingCache
        from rl_sr.conditioning import state_router_condition_tensors
        cache=TextEmbeddingCache(self.run/'text_cache',self.runtime(),dtype='bf16')
        split=json.loads((self.run/'split.json').read_text());pending=[]
        for row in split['validation']+split['test']:
            path=self.run/'F0'/f'{row["_sample_key"]}.png';metadata=path.with_suffix('.json')
            expected=dict(actor_sha256=self.cfg['f0_sha256'],lr_sha256=row['_lr_sha256'],hr_sha256=row['_hr_sha256'],
                prompt_sha256=identity_sha256(row['_prompt']),nfe=25,seed=case_seed(row['_sample_key'],1,self.cfg['seed']))
            old=json.loads(metadata.read_text()) if metadata.exists() else {}
            if path.exists() and all(old.get(k)==v for k,v in expected.items()) and old.get('output_sha256')==content_sha256(path):
                row['_f0_path']=str(path)
            else:pending.append((row,path,expected))
        model=self.artist() if pending else None
        for offset in range(0,len(pending),self.cfg['inference_batch']):
            self.check_time();batch=pending[offset:offset+self.cfg['inference_batch']]
            rows=[r for r,_,_ in batch]
            latents=torch.stack([self.latent(model,r['lq_path']) for r in rows]).cuda()
            values=[]
            for row in rows:
                record=cache.find_record(row['_prompt'],row['_sample_key'],validate_file=True)
                if record is None:raise ValueError('Missing personalized embedding')
                value=cache.load_embedding_file(record)
                if value['text_ids'].ndim==2:value['text_ids']=value['text_ids'].unsqueeze(0)
                values.append(value)
            embeds={k:torch.cat([v[k] for v in values]).cuda() for k in ('prompt_embeds','pooled_prompt_embeds','text_ids')}
            cond,mask,conf=state_router_condition_tensors([r['_router_condition'] for r in rows],self.runtime(),'cuda',torch.bfloat16)
            with torch.no_grad():output=sample_batch(model,latents,None,embeds,cond,mask,conf,None,
                [m['seed'] for _,_,m in batch])
            for (row,path,meta),value in zip(batch,output):
                save_image(value,path);write_json_atomic(path.with_suffix('.json'),{**meta,'output_sha256':content_sha256(path)})
                row['_f0_path']=str(path)
            write_json_atomic(self.run/'split.json',split)
            self.log(f'New independent F0 {min(offset+len(batch),len(pending))}/{len(pending)}')
        write_json_atomic(self.run/'split.json',split)
        del model;gc.collect();torch.cuda.empty_cache()
        self.score_f0(self.rows('validation')+self.rows('test'))

    def score_f0(self,rows):
        all_rows=self.rows('validation')+self.rows('test')
        values=self.score('F0',[dict(arm='F0',sample_key=r['_sample_key'],split=r['_split'],round=1,path=r['_f0_path']) for r in all_rows])
        keys={r['_sample_key'] for r in rows}
        return [r for r in values if r['sample_key'] in keys]

    def metrics(self):
        models=super().metrics()
        if torch.cuda.is_available():
            for model in models.values():model.to('cuda')
        return models

    def score(self,arm,outputs):
        path=self.run/'metrics'/f'{arm}.csv';prior={}
        if path.exists():
            with path.open() as f:prior={(r['sample_key'],int(r['round'])):r for r in csv.DictReader(f)}
        sources={r['_sample_key']:r for r in self.rows()};result=[];pending=[]
        for out in outputs:
            row=sources[out['sample_key']]
            if row['_split']=='train':continue
            sha=content_sha256(out['path']);hrsha=content_sha256(row['hq_path'])
            old=prior.get((out['sample_key'],int(out['round'])))
            if old and old['output_sha256']==sha and old['hr_sha256']==hrsha:result.append(old)
            else:pending.append(({**out,'output_sha256':sha,'hr_sha256':hrsha},row))
        device='cuda' if torch.cuda.is_available() else 'cpu'
        for offset in range(0,len(pending),8):
            batch=pending[offset:offset+8]
            a=torch.stack([(image(o['path'])+1)/2 for o,r in batch]).to(device)
            b=torch.stack([(image(r['hq_path'])+1)/2 for o,r in batch]).to(device)
            with torch.no_grad():values={n:m(a,b).detach().float().cpu().reshape(-1).tolist() for n,m in self.metrics().items()}
            if any(len(v)!=len(batch) or not np.isfinite(v).all() for v in values.values()):raise ValueError('Metric batch shape/nonfinite result')
            for i,(out,_) in enumerate(batch):result.append({**out,**{n:float(v[i]) for n,v in values.items()}})
            if offset%128==0:self.log(f'{arm}: paired metrics {min(offset+len(batch),len(pending))}/{len(pending)}')
            self.write_metrics(arm,result)
        self.write_metrics(arm,result)
        return result

    def smoke(self):
        from models.sr_observation_loop import ObservationLoop
        from tools.sr_observation_loop_training import unroll_loss
        torch.manual_seed(self.cfg['seed']);model=ObservationLoop().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0002)
        values=[]
        for row in self.rows('train')[:4]:values.append([image(row[k]) for k in ('_f0_path','lq_path','hq_path')])
        current,lr,hr=[torch.stack([v[i] for v in values]).cuda() for i in range(3)]
        optimizer.zero_grad()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,_=unroll_loss(model,current,lr,hr,3,dict(lr_mode='spatial',use_round=True,use_state=True),.05,.1)
        loss.backward();gradient=float(model.residual_head.weight.grad.norm());optimizer.step()
        if not torch.isfinite(loss) or gradient<=0:raise FloatingPointError('Real-data correction smoke failed')
        write_json_atomic(self.run/'real_smoke.json',dict(loss=float(loss),residual_head_grad_norm=gradient,
            sources=4,actual_loops=3,peak_gib=torch.cuda.max_memory_allocated()/2**30))
        self.log(f'Real aligned-data smoke passed: loss={float(loss):.5f}, gradient={gradient:.6f}')

    def correctors(self):
        from tools.sr_observation_loop_training import train_correctors
        train_correctors(self)

    def flux_generic(self):
        destination=self.run/'flux/generic';destination.mkdir(parents=True,exist_ok=True)
        if (destination/'result.json').exists():return
        model=self.artist();self.active_checkpoint=None;self.active_mode='generic'
        outputs=self.generate(model,'G_generic',self.rows('validation')+self.rows('test'),self.embeds(),max_round=4)
        del model;gc.collect();torch.cuda.empty_cache()
        self.score('G_generic',outputs)
        write_json_atomic(destination/'result.json',dict(arm='G_generic',updates=0,f0_sha256=self.cfg['f0_sha256'],
            raw_full_generation=True,scope='Untrained repeated FLUX control on the identical new heldout sources'))

    def flux_weak(self):
        from tools.sr_observation_loop_flux import train_flux
        train_flux(self,'weak')

    def flux_spatial(self):
        from tools.sr_observation_loop_flux import train_flux
        train_flux(self,'spatial')

    def inspection(self,summary):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        final_arms=('C0_weak','C1_spatial','C2_round_state','C3_recursive','C4_refresh','G_generic','G_weak','G_spatial')
        folder=self.run/'inspection';folder.mkdir(exist_ok=True)
        figure,axes=plt.subplots(1,2,figsize=(11,4))
        for axis,split in zip(axes,('validation','test')):
            for arm in final_arms:
                points=sorted([r for r in summary if r['arm']==arm and r['split']==split],key=lambda r:r['round'])
                if points:axis.plot([r['round'] for r in points],[r['psnr'] for r in points],marker='o',label=arm)
            baseline=[r for r in summary if r['arm']=='F0' and r['split']==split]
            if baseline:axis.axhline(baseline[0]['psnr'],linestyle='--',color='black',label='F0 first round')
            axis.set(title=split,xlabel='Image round',ylabel='Mean RGB PSNR (dB)',xticks=[1,2,3,4]);axis.grid(alpha=.2)
            if axis.has_data():axis.legend(fontsize=7)
        figure.tight_layout();figure.savefig(folder/'psnr_rounds.png',dpi=160);plt.close(figure)
        # Source order fixed before learning/selection. These are not chosen by their gains.
        sources=self.rows('test')[:4]
        for arm in final_arms:
            available=[]
            for row in sources:
                paths=[row['lq_path'],row['hq_path'],row['_f0_path']]+[
                    str(self.run/'trajectories'/arm/f'{row["_sample_key"]}_r{k}.png') for k in (2,3,4)]
                if all(p is not None and Path(p).exists() for p in paths):available.append((row,paths))
            if len(available)!=len(sources) or not available:continue
            figure,axes=plt.subplots(len(available),6,figsize=(16,3*len(available)),squeeze=False)
            for j,(row,paths) in enumerate(available):
                for i,path in enumerate(paths):
                    with Image.open(path) as im:axes[j,i].imshow(np.array(im.convert('RGB')))
                    axes[j,i].set_xticks([]);axes[j,i].set_yticks([])
                    if j==0:axes[j,i].set_title(('LR','HR (evaluation only)','F0 / round 1','round 2','round 3','round 4')[i])
                axes[j,0].set_ylabel(row['_sample_key'][-10:],fontsize=8)
            figure.suptitle(f'{arm}: first 4 fixed test sources; raw outputs',fontsize=12)
            figure.tight_layout();figure.savefig(folder/f'{arm}_fixed_test.png',dpi=130);plt.close(figure)

    def report(self):
        rows=[];files=sorted((self.run/'metrics').glob('*.csv')) if (self.run/'metrics').exists() else []
        for path in files:
            with path.open() as f:values=list(csv.DictReader(f))
            for split in ('validation','test'):
                for k in (1,2,3,4):
                    group=[r for r in values if r['split']==split and int(r['round'])==k]
                    if group:rows.append(dict(arm=path.stem,split=split,round=k,count=len(group),
                        **{n:float(np.mean([float(r[n]) for r in group])) for n in ('psnr','ssim','lpips','dists')}))
        destination=self.run/'round_metrics_summary.csv'
        with destination.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['arm','split','round','count','psnr','ssim','lpips','dists']);w.writeheader();w.writerows(rows)
        intervals=[];rng=np.random.default_rng(self.cfg['seed'])
        for path in files:
            with path.open() as f:values=list(csv.DictReader(f))
            for split in ('validation','test'):
                first={r['sample_key']:r for r in values if r['split']==split and int(r['round'])==1}
                for k in (2,3,4):
                    later={r['sample_key']:r for r in values if r['split']==split and int(r['round'])==k}
                    keys=sorted(first.keys()&later.keys())
                    if not keys:continue
                    deltas=np.array([float(later[a]['psnr'])-float(first[a]['psnr']) for a in keys])
                    boot=deltas[rng.integers(len(keys),size=(2000,len(keys)))].mean(1)
                    intervals.append(dict(arm=path.stem,split=split,round=k,count=len(keys),psnr_delta=float(deltas.mean()),
                        psnr_degraded=int((deltas<0).sum()),ci95=np.quantile(boot,[.025,.975]).tolist()))
        write_json_atomic(self.run/'paired_bootstrap.json',intervals)
        if (self.run/'split.json').exists():self.inspection(rows)
        text=['# V5 observation constrained recursive SR results','',
            'Training 1024 sources, validation 128, new test 256. F0 first round fixed; raw candidates reported.',
            'Continuation holdout excludes original 3000 training HR groups and prior inspected holdout; F0 pretraining independence is not established.',
            '','| Arm | Split | Round | Images | PSNR | SSIM | LPIPS | DISTS |','|---|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if '_check' in r['arm']:continue
            text.append(f'| {r["arm"]} | {r["split"]} | {r["round"]} | {r["count"]} | {r["psnr"]:.4f} | {r["ssim"]:.5f} | {r["lpips"]:.5f} | {r["dists"]:.5f} |')
        text+=['','See status.json for stage completion, paired_bootstrap.json for paired uncertainty, correction/*/selection.json for validation-only selection and trajectories/* for full512 images.',
            'See inspection/psnr_rounds.png and inspection/*_fixed_test.png for curves and the first four predetermined test sources. Technical protocol: ../../docs/observation_loop_v5.md.',
            'No CLIPIQA/CLIPIQA+/MUSIQ/LPIPS/DISTS as training loss. No HR enters inference or per-image stopping. A failed quality gate is reported as failed; keeping F0 does not count as recursive gain.']
        (self.run/'report.md').write_text('\n'.join(text)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--stage',required=True);args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());exp=Experiment(cfg,args.run)
    if args.stage not in set(cfg['stages'])|{'report'}:raise ValueError('Unknown stage')
    lock=(exp.run/f'{args.stage}.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This experiment stage already has an active worker')
    getattr(exp,args.stage)()


if __name__=='__main__':main()
