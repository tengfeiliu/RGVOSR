"""Run one resumable stage of paired self-trajectory/conservative SR validation.

Generation always uses FLUX.2 Klein native image concatenation, the uploaded
36k F0, 25-NFE proposals and actual previous-round images. No HR at inference.
The supervisor invokes each stage in a fresh process to release CUDA memory.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml
from PIL import Image

from rl_sr.artifact_store import read_jsonl, write_json_atomic, write_jsonl_atomic
from rl_sr.schema import adapter_content_sha256, build_sample_key, content_sha256, identity_sha256
from rl_sr.self_refine import quality_verdict, source_group, split_sources, states_from_trajectory

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('psnr', 'ssim', 'lpips', 'dists')


def case_seed(sample_key, target_round, seed):
    return int(identity_sha256(dict(sample=sample_key, round=target_round, seed=seed))[:12],16) % (2**31-1)


def _json(path):
    return json.loads(Path(path).read_text())


def _write_yaml(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(yaml.safe_dump(value,sort_keys=False)); temp.replace(path)


class Experiment:
    def __init__(self, config, run_dir, level=None, batch_size=None):
        self.config = copy.deepcopy(config)
        self.root_run = Path(run_dir).resolve()
        self.level = int(level) if level else int(config.get('train_sources',128))
        self.run = self.root_run if not level else self.root_run/f'level_{level}'
        self.run.mkdir(parents=True,exist_ok=True)
        self.baseline = Path(config['baseline_run_dir'])
        self.base_repo = Path(config.get('baseline_repo',ROOT))
        self.batch_override = batch_size
        code_files=('tools/run_sr_self_refine_experiment.py','rl_sr/self_refine.py',
                    'models/sr_update_controller.py','rl_sr/conservative.py','train_sr_update_controller.py',
                    'dataloaders/sr_update_dataset.py','train_rg_flux_refiner_sft.py','rl_sr/training.py',
                    'models/flux2_klein_sr_artist.py','models/text_embedding_cache.py',
                    'models/prompt_builder.py','models/router_condition.py','rg_flux_fm.py')
        code_files += tuple(getattr(self, 'extra_code_files', ()))
        code={name:content_sha256(ROOT/name) for name in code_files}
        self.signature = identity_sha256(dict(config=config,level=self.level,source_code_sha256=code))
        signature_path = self.run/'experiment.json'
        if signature_path.exists() and _json(signature_path)['config_sha256'] != self.signature:
            raise ValueError('Existing experiment config differs; choose a new run directory')
        if not signature_path.exists():
            commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
            write_json_atomic(signature_path,dict(config=config,config_sha256=self.signature,level=self.level,
                                                 source_code_sha256=code,git_commit=commit))

    def journal(self, message):
        line=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line,flush=True)
        with (self.run/'journal.md').open('a') as f: f.write(line+'\n\n')

    def sources(self, names=('train','validation','test')):
        split=_json(self.run/'split.json')
        core={k:[{a:b for a,b in r.items() if a!='_f0_path'} for r in v] for k,v in split.items()}
        if core!=_json(self.run/'split_core.json'):
            raise ValueError('Immutable split/prompt/HR assignment changed')
        return [r for name in names for r in split[name]]

    def prepare(self):
        rows=read_jsonl(self.config['source_jsonl'])
        inputs=(self.baseline/'00_inputs/train_lq_inputs.txt').read_text().splitlines()
        split=split_sources(rows,inputs,self.level,int(self.config.get('validation_sources',32)),
                            int(self.config.get('test_sources',32)),int(self.config.get('seed',42)))
        from models.prompt_builder import build_sr_prompt
        from models.router_condition import extract_router_condition, ROUTER_CONDITION_VERSION
        cache={str(Path(r['source_path']).resolve()):r['rounds'][0]['path']
               for r in read_jsonl(self.baseline/'01_f0_round1_train_state/sample_lineage.jsonl')}
        for name,sources in split.items():
            for row in sources:
                for field in ('lq_path','hq_path'):
                    row[field]=str(Path(row[field]).resolve())
                    if not Path(row[field]).is_file(): raise FileNotFoundError(row[field])
                lr_hash=content_sha256(row['lq_path'])
                row['_sample_key']=build_sample_key('lsdir_precrop512_v1',f'original_lr/sha256/{lr_hash}',lr_hash)
                profile=(row.get('unipercept_raw') or {}).get('profile') or {}
                row['_prompt']=build_sr_prompt(profile,use_prompt=True,use_suggestions=True,
                                               prompt_variant='iqa_suggestion',include_caption=True)
                row['_router_condition']=extract_router_condition(profile,version=ROUTER_CONDITION_VERSION).as_dict()
                row['_f0_path']=cache.get(row['lq_path'])
                row['_split']=name
                if name=='train' and not row['_f0_path']: raise ValueError('Missing cached training F0')
        write_json_atomic(self.run/'split.json',split)
        write_json_atomic(self.run/'split_core.json',
                          {k:[{a:b for a,b in r.items() if a!='_f0_path'} for r in v] for k,v in split.items()})
        write_json_atomic(self.run/'provenance.json',dict(
            baseline=str(self.baseline),config_sha256=self.signature,
            source_jsonl_sha256=content_sha256(self.config['source_jsonl']),
            baseline_inputs_sha256=content_sha256(self.baseline/'00_inputs/train_lq_inputs.txt'),
            split_counts={k:len(v) for k,v in split.items()},
            unique_original_hr_counts={k:len({source_group(r) for r in v}) for k,v in split.items()},
            holdout_excludes_all_baseline_hr_groups=True,
            scope='Holdout of continuation training; overlap with 36k pretraining is not established',
            hr_groups={k:[source_group(r) for r in v] for k,v in split.items()}))
        self.journal(f'Prepared {self.level} train + validation/test outside all baseline HR groups')

    def runtime_config(self):
        config=yaml.safe_load((self.baseline/'launch_config.yaml').read_text())
        config['rl_sr']['init_adapter']=str(self.f0_path(config))
        config['text_encoding'].update(mode='cached',cache_dir=str(self.root_run/'text_cache'),dtype='bf16',strict=True)
        # Budget counts are optimizer updates, not accumulated microbatches.
        config['rl_sr']['grad_accum_steps']=1
        config['training']['grad_accum_steps']=1
        recorded=self.run/'generator_batch.json'
        matched=_json(recorded)['batch_size'] if recorded.exists() else None
        config['rl_sr']['batch_size']=int(matched or self.batch_override or self.config.get('generator_batch_size',2))
        config['rl_sr']['num_workers']=2
        config['rl_sr']['log_every']=10
        config['model']['gradient_checkpointing']=bool(self.config.get('gradient_checkpointing',True))
        return config

    def f0_path(self, config=None):
        config=config or yaml.safe_load((self.baseline/'launch_config.yaml').read_text())
        path=Path(config['rl_sr']['init_adapter'])
        path=path if path.is_absolute() else self.base_repo/path
        expected=self.config.get('expected_f0_sha256')
        if expected and not getattr(self,'_f0_verified',False):
            actual=adapter_content_sha256(path)
            if actual!=expected: raise ValueError(f'Uploaded F0 checkpoint hash differs: {actual} != {expected}')
            self._f0_verified=True
        return path

    def snapshot_c0(self):
        target=self.root_run/'C0'
        source=self.baseline/'03_g_sft'
        summary=_json(source/'summary.json')
        weights=source/'rg_flux_adapters'
        actual=adapter_content_sha256(weights)
        if actual != summary['adapter_sha256']: raise ValueError('Baseline C incomplete/checkpoint hash mismatch')
        if not target.exists():
            temp=target.with_name('C0.copying')
            if temp.exists(): shutil.rmtree(temp)
            shutil.copytree(source,temp,ignore=shutil.ignore_patterns('config.json'))
            temp.replace(target)
        if adapter_content_sha256(target/'rg_flux_adapters') != actual:
            raise ValueError('Immutable C0 differs from completed baseline C')
        return target/'rg_flux_adapters'

    def cache_text(self):
        import torch
        from diffusers import Flux2KleinPipeline
        from models.flux2_klein_sr_artist import Flux2KleinSRArtist
        from models.text_embedding_cache import TextEmbeddingCache,validate_online_prompt_lengths
        from types import SimpleNamespace
        config=self.runtime_config()
        cache=TextEmbeddingCache(self.root_run/'text_cache',config,dtype='bf16')
        pending=[r for r in self.sources() if not cache.find_record(r['_prompt'],r['_sample_key'],validate_file=True)]
        if not pending: return
        pipe=Flux2KleinPipeline.from_pretrained(config['model']['flux_model_path'],
                        transformer=None,vae=None,torch_dtype=torch.float32).to('cuda')
        for name in ('text_encoder','text_encoder_2'):
            module=getattr(pipe,name,None)
            if module is not None: module.requires_grad_(False); module.eval()
        proxy=SimpleNamespace(text_pipeline=pipe,text_encoder_device='cuda',
                  max_prompt_sequence_length=config['model']['max_prompt_sequence_length'],weight_dtype=torch.bfloat16)
        proxy.prompt_token_lengths=lambda prompts: Flux2KleinSRArtist.prompt_token_lengths(proxy,prompts)
        lengths=[]
        for i,row in enumerate(pending):
            validate_online_prompt_lengths(proxy,[row['_prompt']],config)
            values=Flux2KleinSRArtist.encode_prompts(proxy,[row['_prompt']],device='cpu',dtype=torch.bfloat16)
            if values[0].shape[-1] != 7680 or values[0].shape[1] != 512:
                raise ValueError(f'Unexpected FLUX.2 prompt encoding shape {values[0].shape}')
            cache.save_embedding(row['_prompt'],row['_sample_key'],row['lq_path'],row['hq_path'],
                         dict(zip(('prompt_embeds','pooled_prompt_embeds','text_ids'),values)))
            lengths.extend(proxy.prompt_token_lengths([row['_prompt']]))
            if i%16==0: self.journal(f'Prompt cache {i+1}/{len(pending)} (Qwen3 FP32, 512x7680 BF16 cache)')
        write_json_atomic(self.run/'text_cache_audit.json',dict(count=len(pending),max_tokens=max(lengths),
                  shape=[1,512,7680],max_memory_bytes=torch.cuda.max_memory_allocated()))

    def artist(self, adapter):
        import torch
        from models.rg_flux_artist_factory import build_rg_flux_artist
        model=build_rg_flux_artist(self.runtime_config()).to(device='cuda')
        model.load_trainable(adapter,is_trainable=False)
        model.align_inference_dtype(dtype=torch.bfloat16)
        model.requires_grad_(False); model.eval(); model.set_moe_inference_schedule()
        return model

    @staticmethod
    def image(path, size=None):
        import torch
        import numpy as np
        with Image.open(path) as im:
            im=im.convert('RGB')
            if size and im.size!=size: im=im.resize(size,Image.Resampling.BICUBIC)
            array=np.array(im,dtype=np.float32)/127.5-1
        return torch.from_numpy(array).permute(2,0,1).unsqueeze(0)

    @staticmethod
    def save_image(image,path):
        import numpy as np
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        array=(image.detach().float().cpu()[0].permute(1,2,0).clamp(-1,1).numpy()+1)*127.5
        temp=path.with_suffix('.tmp.png')
        Image.fromarray(np.rint(array).astype(np.uint8)).save(temp); temp.replace(path)

    def proposal(self, model, row, current_path, target_round, f0=False):
        import torch
        from models.text_embedding_cache import get_text_embedding_cache,resolve_prompt_embeddings
        from rl_sr.conditioning import state_router_condition_tensors
        from rg_flux_fm import sample_multistep_fm
        config=self.runtime_config(); dtype=torch.bfloat16
        current=self.image(current_path).to('cuda',dtype=dtype)
        anchor=self.image(row['lq_path'],size=(current.shape[-1],current.shape[-2])).to('cuda',dtype=dtype)
        seed=case_seed(row['_sample_key'],target_round,int(self.config.get('seed',42)))
        with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            z_current=model.encode_images(current,sample=False).to('cuda',dtype=dtype)
            z_anchor=None if f0 else model.encode_images(anchor,sample=False).to('cuda',dtype=dtype)
            embeds=resolve_prompt_embeddings(model,[row['_prompt']],[row['_sample_key']],config,'cuda',dtype,
                                             cache=get_text_embedding_cache(config,dtype=dtype))
            condition,mask,confidence=state_router_condition_tensors([row['_router_condition']],config,'cuda',dtype)
            latent=sample_multistep_fm(model,z_current.shape,embeds[0],embeds[1],text_ids=embeds[2],
                      z_lr=z_current,z_anchor_lr=z_anchor,refinement_round=None if f0 else target_round,
                      lr_cond_mode='flux2_image_concat',router_condition=condition,
                      router_condition_mask=mask,router_condition_confidence=confidence,
                      num_steps=25,schedule='linear',init_mode='pure_noise',sigma_start=1.,device='cuda',dtype=dtype)
            output=model.decode_latents(latent).float().clamp(-1,1)
        return output,seed

    def ensure_f0(self):
        missing=[r for r in self.sources() if not r['_f0_path']]
        if not missing: return
        model=self.artist(self.f0_path())
        f0_hash=adapter_content_sha256(self.f0_path())
        for i,row in enumerate(missing):
            path=self.root_run/'F0'/f"{row['_sample_key']}.png"
            meta=path.with_suffix('.json')
            expected=dict(actor_sha256=f0_hash,lr_sha256=content_sha256(row['lq_path']),
                    prompt_sha256=identity_sha256({'prompt':row['_prompt']}),
                    seed=case_seed(row['_sample_key'],1,self.config.get('seed',42)),nfe=25)
            cached=_json(meta) if meta.exists() else {}
            valid=path.exists() and all(cached.get(k)==v for k,v in expected.items()) and cached.get('output_sha256')==content_sha256(path)
            if not valid:
                # LQ is already resized to 512 by the source degradation pipeline.
                output,seed=self.proposal(model,row,row['lq_path'],1,f0=True)
                self.save_image(output,path)
                write_json_atomic(meta,{**expected,'output_sha256':content_sha256(path)})
            row['_f0_path']=str(path)
            self.journal(f'Independent heldout F0 {i+1}/{len(missing)}')
        split=_json(self.run/'split.json')
        lookup={r['_sample_key']:r['_f0_path'] for r in missing}
        for group in split.values():
            for row in group:
                if row['_sample_key'] in lookup: row['_f0_path']=lookup[row['_sample_key']]
        write_json_atomic(self.run/'split.json',split)
        del model; gc.collect()
        import torch
        torch.cuda.empty_cache()

    def rollout(self, arm, adapter, names=('train','validation','test'), controller=None,
                fixed_alpha=None, identity=False, existing_model=None):
        import torch
        from rl_sr.conservative import fuse_rgb
        rows=self.sources(names)
        directory=self.run/'trajectories'/arm
        directory.mkdir(parents=True,exist_ok=True)
        adapter_hash=adapter_content_sha256(adapter)
        policy=dict(actor_sha256=adapter_hash,controller_sha256=content_sha256(controller) if controller else None,
                    fixed_alpha=fixed_alpha,noop_threshold=float(self.config.get('noop_threshold',.001)),identity=identity,
                    sampler=dict(nfe=25,schedule='linear',init='pure_noise',sigma_start=1.,
                                 seed=int(self.config.get('seed',42))),source_keys=[r['_sample_key'] for r in rows])
        signature=identity_sha256(policy)
        if (directory/'policy.json').exists() and _json(directory/'policy.json')['policy_sha256']!=signature:
            raise ValueError(f'Changed old policy cannot reuse trajectories {arm}')
        write_json_atomic(directory/'policy.json',{**policy,'policy_sha256':signature})
        model=existing_model or self.artist(adapter)
        gate=None
        if controller:
            from models.sr_update_controller import SRUpdateController
            gate=SRUpdateController.load(controller,device='cuda').eval()
        outputs=[]
        for i,row in enumerate(rows):
            current=row['hq_path'] if identity else row['_f0_path']
            outputs.append(dict(sample_key=row['_sample_key'],round=1,path=current,split=row['_split']))
            for k in range(2,5):
                if identity: current=row['hq_path'] # actual identity proposals, each target-round condition
                path=directory/f"{row['_sample_key']}_r{k}.png"
                meta=path.with_suffix('.json'); raw=path.with_name(path.stem+'_proposal.png')
                expected=dict(policy_sha256=signature,parent_sha256=content_sha256(current),
                    lr_sha256=content_sha256(row['lq_path']),prompt_sha256=identity_sha256({'prompt':row['_prompt']}),round=k)
                cached=_json(meta) if meta.exists() else {}
                valid=(path.exists() and raw.exists() and
                       all(cached.get(a)==b for a,b in expected.items()) and
                       cached.get('output_sha256')==content_sha256(path) and
                       cached.get('proposal_sha256')==content_sha256(raw))
                if valid:
                    info=_json(meta)
                else:
                    begin=time.monotonic(); proposal,seed=self.proposal(model,row,current,k)
                    self.save_image(proposal,raw)
                    weight=None; noop=False
                    if gate is not None or fixed_alpha is not None:
                        parent=self.image(current).to('cuda',dtype=torch.float32)
                        # Match SRUpdateDataset interpolation exactly, independent of the VAE anchor path.
                        import torch.nn.functional as F
                        lr=self.image(row['lq_path']).to('cuda')
                        if lr.shape[-2:]!=parent.shape[-2:]:
                            lr=F.interpolate(lr,size=parent.shape[-2:],mode='bicubic',align_corners=False).clamp(-1,1)
                        with torch.no_grad():
                            weight=gate(parent,lr,proposal,k)['weight'] if gate is not None else torch.full_like(parent[:,:1],fixed_alpha)
                            cutoff=float(self.config.get('noop_threshold',.001))
                            noop=bool(weight.max().item() <= cutoff)
                            output=fuse_rgb(parent,proposal,weight,noop_threshold=cutoff)
                    else: output=proposal
                    if noop:
                        shutil.copyfile(current,path) # preserves original uint8 bytes, no re-encoding
                    else: self.save_image(output,path)
                    info={**expected,'seed':seed,'seconds':time.monotonic()-begin,
                          'weight_mean':weight.mean().item() if weight is not None else 1.,
                          'weight_max':weight.max().item() if weight is not None else 1.,'noop':noop,
                          'path':str(path),'proposal':str(raw),
                          'output_sha256':content_sha256(path),'proposal_sha256':content_sha256(raw),
                          'max_memory_bytes':torch.cuda.max_memory_allocated()}
                    write_json_atomic(meta,info)
                outputs.append({**info,'sample_key':row['_sample_key'],'round':k,'path':str(path),
                                'current_sr':current,'split':row['_split']})
                current=str(path)
            if i%8==0: self.journal(f'{arm} actual chain {i+1}/{len(rows)}')
        write_jsonl_atomic(directory/'outputs.jsonl',outputs)
        if existing_model is None:
            del model,gate; gc.collect(); torch.cuda.empty_cache()
        return outputs,signature

    def old_states(self):
        selected={r['lq_path'] for r in self.sources(('train',))}
        records=[r for r in read_jsonl(self.baseline/'02_states_for_c.jsonl')
                 if str(Path(r['runtime']['original_lr']).resolve()) in selected]
        if len(records)!=len(selected)*3: raise ValueError(f'Old buffer missing mixed rounds: {len(records)}')
        for record in records:
            if content_sha256(record['runtime']['current_sr'])!=record['current_output_sha256']:
                raise ValueError('Old buffer image hash mismatch; cannot train on stale states')
        write_jsonl_atomic(self.run/'states_old.jsonl',records)
        return records

    def self_states(self, arm, adapter):
        outputs=read_jsonl(self.run/'trajectories'/arm/'outputs.jsonl')
        records=states_from_trajectory(self.sources(('train',)),outputs,adapter_content_sha256(adapter),
                      adapter_content_sha256(self.f0_path()),identity_sha256(dict(nfe=25,schedule='linear',seed=self.config.get('seed',42))))
        path=self.run/f'states_{arm}.jsonl'; write_jsonl_atomic(path,records)
        return path,records

    def collect_C0(self):
        c0=self.snapshot_c0(); self.ensure_f0()
        self.rollout('C0',c0)
        self.old_states(); self.self_states('C0',c0)
        self.evaluate('C0')

    def generator_budget(self):
        values=self.config.get('scale_generator_updates',{})
        return int(values.get(self.level,values.get(str(self.level),self.config.get('generator_updates',100))))

    def sft(self, arm, manifest, init_adapter, steps):
        config=self.runtime_config(); config['rl_sr']['init_adapter']=str(init_adapter)
        directory=self.run/'generators'/arm
        inputs=dict(manifest_sha256=content_sha256(manifest),init_adapter_sha256=adapter_content_sha256(init_adapter),
                    steps=steps,config_sha256=identity_sha256({k:v for k,v in config.items() if k!='_runtime'}))
        if (directory/'summary.json').exists():
            summary=_json(directory/'summary.json')
            if ((directory/'pilot_inputs.json').exists() and _json(directory/'pilot_inputs.json')==inputs and int(summary['global_step'])==steps and
                adapter_content_sha256(directory/'rg_flux_adapters')==summary['adapter_sha256']):
                return directory/'rg_flux_adapters'
        directory.mkdir(parents=True,exist_ok=True)
        write_json_atomic(directory/'pilot_inputs.json',inputs)
        _write_yaml(directory/'runtime_config.yaml',config)
        self.journal(f'{arm}: {steps} optimizer updates, batch{config["rl_sr"]["batch_size"]}, frozen backbone')
        from train_rg_flux_refiner_sft import train
        train(config,state_jsonl=str(manifest),output_dir=str(directory),max_steps=steps)
        return directory/'rg_flux_adapters'

    def train_A(self):
        cycles=int(self.config.get('refresh_cycles',2)); budget=self.generator_budget()
        initial=self.snapshot_c0()
        for cycle in range(cycles):
            steps=budget//cycles+(cycle<budget%cycles)
            initial=self.sft(f'A_cycle{cycle}',self.run/'states_old.jsonl',initial,steps)
        write_json_atomic(self.run/'generator_A.json',dict(adapter=str(initial),cycles=cycles,updates=budget))
        write_json_atomic(self.run/'generator_batch.json',dict(batch_size=int(self.batch_override or self.config.get('generator_batch_size',2)),
                                reason='Same number of images per optimizer update for matched A/B arms'))

    def train_B(self):
        import torch
        cycles=int(self.config.get('refresh_cycles',2))
        budget=self.generator_budget()
        if cycles<1 or cycles>budget: raise ValueError('Invalid refresh_cycles')
        old=read_jsonl(self.run/'states_old.jsonl'); initial=self.snapshot_c0(); path=self.run/'states_C0.jsonl'
        fraction=float(self.config.get('self_state_fraction',.5))
        if not 0<fraction<=1: raise ValueError('self_state_fraction must be in (0,1]')
        for cycle in range(cycles):
            own=read_jsonl(path)
            # One state per source/round: equal coverage, deterministic mixture,
            # no duplicate source weighting or omitted images across the arms.
            import random
            mixture=[]
            for k in (2,3,4):
                a=[r for r in old if r['round_index']==k]; b=[r for r in own if r['round_index']==k]
                old_index={r['runtime']['original_lr']:r for r in a}
                own_index={r['runtime']['original_lr']:r for r in b}
                if set(old_index)!=set(own_index): raise ValueError('Old/self buffers cover different source sets')
                keys=sorted(old_index); random.Random(self.config.get('seed',42)+cycle*17+k).shuffle(keys)
                self_keys=set(keys[:round(len(keys)*fraction)])
                mixture.extend(own_index[key] if key in self_keys else old_index[key] for key in keys)
            manifest=self.run/f'states_B_cycle{cycle}.jsonl'; write_jsonl_atomic(manifest,mixture)
            steps=budget//cycles+(cycle<budget%cycles)
            initial=self.sft(f'B_cycle{cycle}',manifest,initial,steps)
            gc.collect(); torch.cuda.empty_cache()
            # Freeze the newly optimized snapshot, then visit its actual own states.
            if cycle+1<cycles:
                arm=f'B_refresh{cycle}'
                self.rollout(arm,initial,names=('train',)); path,_=self.self_states(arm,initial)
            self.journal(f'Old/new refresh cycle {cycle+1}/{cycles} complete; new={adapter_content_sha256(initial)}')
        write_json_atomic(self.run/'generator_B.json',dict(adapter=str(initial),cycles=cycles,updates=budget))

    def actor_path(self, arm):
        if arm=='C0': return self.snapshot_c0()
        if arm=='A': return Path(_json(self.run/'generator_A.json')['adapter'])
        if arm=='B': return Path(_json(self.run/'generator_B.json')['adapter'])
        raise ValueError(arm)

    def eval_A(self):
        self.rollout('A',self.actor_path('A'),names=('validation','test')); self.evaluate('A')

    def eval_B(self):
        self.rollout('B',self.actor_path('B'),names=('validation','test')); self.evaluate('B')

    def evaluate(self, arm):
        import torch
        import pyiqa
        torch.set_num_threads(min(8,os.cpu_count() or 1))
        # Use the official package weights already installed on this server.
        # This avoids an implicit network download from urllib during long runs.
        paths={}
        for name,module,relative in [('lpips','lpips','weights/v0.1/alex.pth'),
                                      ('dists','DISTS_pytorch','weights.pt')]:
            spec=importlib.util.find_spec(module)
            if spec is not None:
                candidate=Path(spec.origin).parent/relative
                if candidate.is_file(): paths[name]=candidate
        metrics={name:pyiqa.create_metric(name,device='cpu',
                        **({'pretrained_model_path':str(paths[name])} if name in paths else {}),
                        **({'test_y_channel':False} if name in {'psnr','ssim'} else {})) for name in METRICS}
        weights={name:dict(path=str(path),sha256=content_sha256(path)) for name,path in paths.items()}
        write_json_atomic(self.run/'metrics/weights.json',weights)
        hr={r['_sample_key']:r for r in self.sources()}
        results=[]
        outputs=read_jsonl(self.run/'trajectories'/arm/'outputs.jsonl')
        cache_path=self.run/'metrics'/f'{arm}.csv'; cache_path.parent.mkdir(exist_ok=True)
        previous={}
        if cache_path.exists():
            with cache_path.open() as f: previous={(r['sample_key'],int(r['round'])):r for r in csv.DictReader(f)}
        for i,row in enumerate(outputs):
            source=hr[row['sample_key']]
            if source['_split']=='train': continue
            digest=content_sha256(row['path']); gt_digest=content_sha256(source['hq_path'])
            cached=previous.get((row['sample_key'],row['round']))
            if cached and cached['output_sha256']==digest and cached['hr_sha256']==gt_digest:
                result={**cached,**{m:float(cached[m]) for m in METRICS},'round':row['round']}
            else:
                output=(self.image(row['path'])+1)/2; target=(self.image(source['hq_path'])+1)/2
                if output.shape!=target.shape: raise ValueError('Paired metric shape mismatch')
                with torch.no_grad(): values={name:float(metric(output,target).item()) for name,metric in metrics.items()}
                if not all(math.isfinite(v) for v in values.values()): raise ValueError('Nonfinite quality metric')
                result=dict(arm=arm,sample_key=row['sample_key'],split=source['_split'],round=row['round'],
                            output_sha256=digest,hr_sha256=gt_digest,**values)
            results.append(result)
            # Small pilot durable progress; metric retries reuse completed rows.
            if len(results)%16==0: self.write_csv(cache_path,results)
        self.write_csv(cache_path,results)
        summary={}
        for split in ('validation','test'):
            summary[split]={str(k):{m:sum(r[m] for r in results if r['split']==split and r['round']==k)/
                sum(1 for r in results if r['split']==split and r['round']==k) for m in METRICS} for k in range(1,5)}
        write_json_atomic(self.run/'metrics'/f'{arm}.json',summary)
        self.journal(f'{arm}: paired PSNR/SSIM/LPIPS/DISTS complete; validation r4={summary["validation"]["4"]}')
        return summary

    @staticmethod
    def write_csv(path,rows):
        temp=path.with_suffix('.tmp.csv')
        with temp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        temp.replace(path)

    def select_actor(self):
        base=_json(self.run/'metrics/C0.json')['validation']['4']
        a=_json(self.run/'metrics/A.json')['validation']['4']; b=_json(self.run/'metrics/B.json')['validation']['4']
        thresholds=self.config.get('quality_thresholds',{})
        # Fidelity and both perceptual metrics on validation only; test is never used for selection.
        useful_b=quality_verdict(a,b,b,thresholds)['improves_baseline']
        b_fidelity=(b['psnr']>=base['psnr']-thresholds.get('psnr_tolerance',.05) and
                    b['ssim']>=base['ssim']-thresholds.get('ssim_tolerance',.002))
        selected='B' if useful_b and b_fidelity else 'A' if quality_verdict(base,a,a,thresholds)['improves_baseline'] else 'C0'
        result=dict(selected_actor=selected,adapter=str(self.actor_path(selected)),
                    self_trajectory_beats_matched_A=useful_b,validation_only=True)
        write_json_atomic(self.run/'selected_actor.json',result)
        return result

    def collect_controller(self):
        selected=self.select_actor(); adapter=Path(selected['adapter']); arm='controller_train'
        outputs,policy=self.rollout(arm,adapter,names=('train',))
        manifest=[]
        sources={r['_sample_key']:r for r in self.sources(('train',))}
        for output in outputs:
            if output['round']==1: continue
            source=sources[output['sample_key']]
            manifest.append(dict(sample_key=output['sample_key'],original_lr=source['lq_path'],
                current_sr=output['current_sr'],proposal=output['proposal'],hr=source['hq_path'],
                target_round=output['round'],producer=dict(policy_sha256=policy,seed=output['seed'],nfe=25),identity=False))
        # Actual HR-conditioned proposals, not relabelled ordinary proposals.
        fraction=float(self.config.get('identity_fraction',.10))
        subset=self.sources(('train',))[:max(1,round(self.level*fraction))]
        model=self.artist(adapter)
        actor_hash=adapter_content_sha256(adapter)
        for i,source in enumerate(subset):
            for k in (2,3,4):
                path=self.run/'identity'/f"{source['_sample_key']}_r{k}.png"
                metadata=path.with_suffix('.json')
                expected=dict(actor_sha256=actor_hash,hr_sha256=content_sha256(source['hq_path']),
                              lr_sha256=content_sha256(source['lq_path']),round=k,
                              seed=case_seed(source['_sample_key'],k,self.config.get('seed',42)))
                cached=_json(metadata) if metadata.exists() else {}
                valid=path.exists() and all(cached.get(a)==b for a,b in expected.items()) and cached.get('proposal_sha256')==content_sha256(path)
                if not valid:
                    proposal,seed=self.proposal(model,source,source['hq_path'],k)
                    self.save_image(proposal,path)
                    write_json_atomic(metadata,{**expected,'proposal_sha256':content_sha256(path)})
                manifest.append(dict(sample_key=source['_sample_key'],original_lr=source['lq_path'],
                    current_sr=source['hq_path'],proposal=str(path),hr=source['hq_path'],target_round=k,
                    producer=dict(actor_sha256=actor_hash,nfe=25,
                                  seed=case_seed(source['_sample_key'],k,self.config.get('seed',42)),current_is_hr=True),identity=True))
        write_jsonl_atomic(self.run/'controller_manifest.jsonl',manifest)
        self.journal(f'Controller real proposals={len(manifest)} incl {len(subset)*3} verified identity rows')

    def train_controller(self, full):
        from train_sr_update_controller import parse_args,train
        name='full' if full else 'plain'
        root=self.run/'controllers'/name; root.mkdir(parents=True,exist_ok=True)
        attempts=sorted(root.glob('attempt_*'))
        completed=[p for p in attempts if (p/'summary.json').exists() and _json(p/'summary.json')['status']=='complete']
        if completed:
            write_json_atomic(self.run/f'controller_{name}.json',dict(checkpoint=str(completed[-1]/'controller.pt'))); return
        # Preserve failed zero-update attempts; later resume is exact and permits smaller batches.
        resume=attempts[-1] if attempts and (attempts[-1]/'trainer_state.pt').exists() else None
        directory=resume or root/f'attempt_{len(attempts)+1:02d}'
        updates=self.config.get('scale_controller_updates',{}).get(self.level,self.config.get('controller_updates',200))
        flags=['--manifest',str(self.run/'controller_manifest.jsonl'),'--output_dir',str(directory),
            '--max_steps',str(updates),'--batch_size',str(self.batch_override or self.config.get('controller_batch_size',16)),
            '--device','cuda','--lr',str(self.config.get('controller_lr',.001)),
            '--crop_size',str(self.config.get('controller_crop_size',256)),'--seed',str(self.config.get('seed',42)),
            '--channels',str(self.config.get('controller_channels',32)),
            '--lpips_weight',str(self.config.get('controller_lpips_weight',.05)),
            '--regress_weight',str(self.config.get('controller_regress_weight',1.) if full else 0),
            '--identity_weight',str(self.config.get('controller_identity_weight',1.) if full else 0),
            '--smooth_weight',str(self.config.get('controller_smooth_weight',.001))]
        if resume: flags += ['--resume']
        train(parse_args(flags))
        write_json_atomic(self.run/f'controller_{name}.json',dict(checkpoint=str(directory/'controller.pt')))

    def train_controller_plain(self): self.train_controller(False)
    def train_controller_full(self): self.train_controller(True)

    def eval_controls(self):
        selected=_json(self.run/'selected_actor.json'); adapter=Path(selected['adapter'])
        for name in ('plain','full'):
            controller=_json(self.run/f'controller_{name}.json')['checkpoint']
            self.rollout(f'gate_{name}',adapter,names=('validation','test'),controller=controller)
            self.evaluate(f'gate_{name}')
        self.rollout('fixed_alpha',adapter,names=('validation','test'),fixed_alpha=float(self.config.get('fixed_alpha',.25)))
        self.evaluate('fixed_alpha')

    def verdict(self):
        selected=_json(self.run/'selected_actor.json'); actor=selected['selected_actor']
        thresholds=self.config.get('quality_thresholds',{})
        scores={name:_json(self.run/f'metrics/{name}.json') for name in ('C0','A','B','gate_plain','gate_full','fixed_alpha')}
        # Compare full controller to its raw actor and same-actor early-stop baseline.
        val=quality_verdict(scores[actor]['validation']['4'],scores['gate_full']['validation']['4'],
                            scores['gate_full']['validation']['2'],thresholds)
        val_vs_c0=quality_verdict(scores['C0']['validation']['4'],scores['gate_full']['validation']['4'],
                            scores['gate_full']['validation']['2'],thresholds)
        test=quality_verdict(scores[actor]['test']['4'],scores['gate_full']['test']['4'],
                            scores['gate_full']['test']['2'],thresholds)
        # A weak gated round2 is not enough: beat the actual full-strength
        # raw refiner's round2 early-stop output on the same images as well.
        val_raw_early=quality_verdict(scores[actor]['validation']['2'],scores['gate_full']['validation']['4'],
                            scores[actor]['validation']['2'],thresholds)
        comparisons={}
        for label,first,last,early in [('self_vs_matched','A','B',4),('gate_vs_actor',actor,'gate_full',4),
                                       ('late_vs_round2','gate_full','gate_full',2),
                                       ('late_vs_raw_round2',actor,'gate_full',2)]:
            comparisons[label]=self.paired_gain(first,early,last,4,'validation')
        confident=all(v['lpips_ci95'][0]>0 and v['dists_ci95'][0]>0 for v in comparisons.values())
        scale_ok=(selected['selected_actor']=='B' and selected['self_trajectory_beats_matched_A'] and
                  val['scale_ok'] and val_vs_c0['scale_ok'] and val_raw_early['scale_ok'] and confident)
        levels=[int(n) for n in self.config.get('scale_levels',[512,3000])]
        next_level=next((n for n in levels if n>self.level),None) if scale_ok and self.config.get('auto_scale',True) else None
        result=dict(level=self.level,selected_actor=selected,validation=val,validation_vs_C0=val_vs_c0,
                    validation_vs_raw_round2_stop=val_raw_early,
                    independent_test=test,scale_ok=scale_ok,next_scale_sources=next_level,
                    paired_validation_bootstrap=comparisons,confidence_ok=confident,
                    status='positive_validation' if scale_ok else 'validation_failed_no_scale',
                    metric_protocol='RGB [0,1], full512 images, no border crop, paired pyiqa; no NR reward tuning')
        write_json_atomic(self.run/'verdict.json',result)
        lines=['# Experiment report','',f'Status: {result["status"]}; training sources: {self.level}.','',
               'Selection and expansion use validation only. Test is a separately reported diagnostic.',
               'Holdout excludes all baseline continuation-training HR groups; 36k pretraining overlap remains unknown.','',
               '| Arm | Round | PSNR | SSIM | LPIPS | DISTS |','|---|---:|---:|---:|---:|---:|']
        for name,score in scores.items():
            for k in ('2','3','4'):
                s=score['validation'][k]
                lines.append(f'| {name} | {k} | {s["psnr"]:.4f} | {s["ssim"]:.5f} | {s["lpips"]:.5f} | {s["dists"]:.5f} |')
        lines += ['',f'Next training scale: {next_level}.','',
                  'Plateau caused by repeated no-op alone is insufficient for scale-up. See raw per-image CSVs and trajectories.',
                  'A negative pilot preserves the fixed-alpha and plain-controller controls for diagnosis; no automatic large run.']
        (self.run/'report.md').write_text('\n'.join(lines)+'\n')
        self.journal(f'Verdict: {result["status"]}; next_scale_sources={next_level}')
        return result

    def paired_gain(self, first, first_round, last, last_round, split):
        import numpy as np
        def load(arm,k):
            with (self.run/f'metrics/{arm}.csv').open() as f:
                return {r['sample_key']:r for r in csv.DictReader(f) if r['split']==split and int(r['round'])==k}
        a,b=load(first,first_round),load(last,last_round)
        if set(a)!=set(b) or not a: raise ValueError('Paired comparison sample identities differ')
        keys=sorted(a); rng=np.random.default_rng(self.config.get('seed',42))
        indices=rng.integers(0,len(keys),size=(2000,len(keys)))
        result={'count':len(keys),'first':first,'first_round':first_round,'last':last,'last_round':last_round}
        for name in METRICS:
            delta=np.array([float(b[k][name])-float(a[k][name]) for k in keys])
            gain=-delta if name in {'lpips','dists'} else delta
            result[f'{name}_mean_gain']=float(gain.mean())
            result[f'{name}_ci95']=[float(v) for v in np.quantile(gain[indices].mean(axis=1),[.025,.975])]
            result[f'{name}_nondegradation_fraction']=float((gain>=-1e-8).mean())
        return result

    def artifact_paths(self,stage):
        """Fingerprint durable outputs; later stages may add unrelated artifacts."""
        paths=[]
        def tree(path):
            if path.exists(): paths.extend(p for p in path.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
        if stage=='prepare':
            paths.extend([self.run/'provenance.json',self.run/'split_core.json'])
            paths.append(Path(self.config['source_jsonl']))
            for r in self.sources(): paths.extend(Path(r[f]) for f in ('lq_path','hq_path'))
        elif stage=='cache_text':
            from models.text_embedding_cache import get_text_embedding_cache
            cache=get_text_embedding_cache(self.runtime_config(),dtype='bf16')
            for r in self.sources():
                record=cache.find_record(r['_prompt'],r['_sample_key'],validate_file=True)
                if record is None: raise ValueError('Prompt cache incomplete')
                paths.append(cache.cache_dir/record['embedding_path'])
        elif stage=='collect_C0':
            tree(self.run/'trajectories/C0'); tree(self.root_run/'C0')
            paths.extend([self.run/'states_old.jsonl',self.run/'states_C0.jsonl',self.run/'metrics/C0.csv',self.run/'metrics/C0.json'])
            for row in read_jsonl(self.run/'trajectories/C0/outputs.jsonl'):
                if row['round']==1: paths.append(Path(row['path']))
        elif stage in {'train_A','train_B'}:
            letter=stage[-1]
            paths.append(self.run/f'generator_{letter}.json')
            if letter=='A': paths.append(self.run/'generator_batch.json')
            for directory in (self.run/'generators').glob(f'{letter}_cycle*'): tree(directory)
            if letter=='B':
                for directory in (self.run/'trajectories').glob('B_refresh*'): tree(directory)
                paths.extend(self.run.glob('states_B_*.jsonl'))
        elif stage in {'eval_A','eval_B'}:
            letter=stage[-1]; tree(self.run/f'trajectories/{letter}')
            paths.extend([self.run/f'metrics/{letter}.csv',self.run/f'metrics/{letter}.json'])
        elif stage=='collect_controller':
            tree(self.run/'trajectories/controller_train'); tree(self.run/'identity')
            paths.extend([self.run/'selected_actor.json',self.run/'controller_manifest.jsonl'])
        elif stage.startswith('train_controller_'):
            name=stage.removeprefix('train_controller_')
            tree(self.run/f'controllers/{name}'); paths.append(self.run/f'controller_{name}.json')
        elif stage=='eval_controls':
            for name in ('gate_plain','gate_full','fixed_alpha'):
                tree(self.run/f'trajectories/{name}'); paths.extend([self.run/f'metrics/{name}.csv',self.run/f'metrics/{name}.json'])
        elif stage=='verdict': paths.extend([self.run/'verdict.json',self.run/'report.md'])
        return sorted(set(paths))

    def run_stage(self, stage):
        marker=self.run/'completed'/f'{stage}.json'
        if marker.exists():
            saved=_json(marker)
            if saved['config_sha256']!=self.signature: raise ValueError('Stage config mismatch')
            if all(Path(p).is_file() and content_sha256(p)==h for p,h in saved.get('artifacts',{}).items()):
                return
            self.journal(f'Stage {stage} artifact changed; invalidate marker and recover actual outputs')
            marker.unlink()
        if stage not in {'prepare','cache_text','collect_C0','train_A','train_B','eval_A','eval_B',
            'collect_controller','train_controller_plain','train_controller_full','eval_controls','verdict',
            *getattr(self, 'extra_stages', ())}:
            raise ValueError(f'Unknown stage {stage}')
        started=time.monotonic(); self.journal(f'START {stage}')
        result=getattr(self,stage)()
        artifacts={str(p):content_sha256(p) for p in self.artifact_paths(stage)}
        write_json_atomic(marker,dict(stage=stage,config_sha256=self.signature,seconds=time.monotonic()-started,
                                     result=result,artifacts=artifacts))
        self.journal(f'DONE {stage}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--run_dir',required=True)
    parser.add_argument('--stage',required=True); parser.add_argument('--level',type=int)
    parser.add_argument('--batch_size',type=int)
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    Experiment(config,args.run_dir,args.level,args.batch_size).run_stage(args.stage)


if __name__=='__main__': main()
