"""Small F0-initialized pilot with verified reuse and actual SR validation."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import math
import os
from pathlib import Path
import shutil

import yaml
from PIL import Image

from rl_sr.artifact_store import read_jsonl, write_json_atomic, write_jsonl_atomic
from rl_sr.schema import adapter_content_sha256, content_sha256, identity_sha256
from rl_sr.stable_reuse import mixed_states, passes_first_round_gate, validate_teacher_state
from tools.run_sr_self_refine_experiment import Experiment, METRICS, _json, case_seed


class StableExperiment(Experiment):
    extra_stages = ('diagnose_real4x',)
    extra_code_files = ('tools/run_sr_stable_experiment.py', 'tools/sr_v3_real_diagnostic.py',
                        'rl_sr/stable_reuse.py', 'rl_sr/stable_training.py',
                        'inference_rg_flux_sr.py', 'tools/run_rg_flux_iterative_inference.py')

    def f0_path(self, config=None):
        path = self.config.get('f0_adapter')
        if not path:
            return super().f0_path(config)
        result = Path(path).resolve()
        if not getattr(self, '_f0_verified', False):
            if adapter_content_sha256(result) != self.config['expected_f0_sha256']:
                raise ValueError('Uploaded 36k F0 hash differs')
            self._f0_verified = True
        return result

    def snapshot_c0(self):
        # C0 keeps the old runner API name; in v3 it is the uploaded F0.
        source = self.f0_path()
        if (source/'rg_flux_adapters').is_dir(): source = source/'rg_flux_adapters'
        target = self.root_run/'C0/rg_flux_adapters'
        digest = adapter_content_sha256(source)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name('rg_flux_adapters.copying')
            if temporary.exists(): shutil.rmtree(temporary)
            shutil.copytree(source, temporary)
            temporary.replace(target)
        if adapter_content_sha256(target) != digest:
            raise ValueError('Immutable initial policy differs from uploaded F0')
        write_json_atomic(target.parent/'initial_policy.json', dict(policy='uploaded_36k_F0', adapter_sha256=digest))
        return target

    def runtime_config(self):
        config = super().runtime_config()
        settings = config['rl_sr']
        settings.update(sft_lora_lr=float(self.config.get('sft_lora_lr', 1e-5)),
                        sft_refinement_lr=float(self.config.get('sft_refinement_lr', 1e-4)),
                        sft_train_lora=bool(self.config.get('sft_train_lora', True)),
                        sft_checkpoint_every=int(self.config.get('checkpoint_every', 25)),
                        lr_warmup_steps=int(self.config.get('warmup_updates', 5)))
        return config

    def sources(self, names=('train','validation','test')):
        rows = super().sources(names)
        if getattr(self, '_teacher_subset', None) is not None:
            rows=[r for r in rows if r['_sample_key'] in self._teacher_subset]
        if getattr(self, '_include_real_for_cache', False):
            rows += _json(self.run/'real_split.json')
        return rows

    def prepare(self):
        super().prepare()
        f0_hash = adapter_content_sha256(self.f0_path())
        # Only completed F0 generation records can establish reuse provenance.
        manifest = _json(self.baseline/'01_f0_round1_train_state/iterative_manifest.json')
        if (manifest['status'] != 'completed' or manifest['upscale_policy']['round_1'] != 1 or
            adapter_content_sha256(manifest['checkpoint_path']) != f0_hash):
            raise ValueError('Cached paired F0 provenance is incompatible')
        for k in (1,2,3):
            sampling=_json(self.baseline/f'02_f0_bootstrap_trajectory/round_{k:02d}/inference_manifest.json')['sampling']
            expected=dict(num_inference_steps=25,schedule='linear',init_mode='pure_noise',sigma_start=1.,
                          seed=manifest['base_seed'],input_upscale=1,uses_refinement_adapter=False)
            if any(sampling.get(a)!=b for a,b in expected.items()):
                raise ValueError('Teacher sampler/conditioning provenance differs; cannot reuse this buffer')
        old_states = read_jsonl(self.baseline/'02_states_for_c.jsonl')
        index = {str(Path(r['runtime']['original_lr']).resolve()):r for r in old_states if r['round_index']==2}
        reuse = []; split = _json(self.run/'split.json')
        prior = Path(self.config['reuse_run_dir'])
        prior_split = _json(prior/'split.json') if (prior/'split.json').exists() else {}
        prior_rows = {r['_sample_key']:r for rows in prior_split.values() for r in rows}
        for name, rows in split.items():
            for row in rows:
                row['_f0_path'] = None
                try:
                    if name == 'train':
                        state = validate_teacher_state(index[row['lq_path']], row, f0_hash)
                        path = Path(state['runtime']['current_sr'])
                        producer_seed = manifest['base_seed']
                    else:
                        previous = prior_rows[row['_sample_key']]
                        if any(previous[k] != row[k] for k in ('lq_path','hq_path','_prompt','_router_condition')):
                            raise ValueError('Prior heldout F0 source/prompt differs')
                        path = Path(previous['_f0_path']); meta = _json(path.with_suffix('.json'))
                        expected = dict(actor_sha256=f0_hash, lr_sha256=content_sha256(row['lq_path']),
                            prompt_sha256=identity_sha256({'prompt':row['_prompt']}), nfe=25,
                            seed=case_seed(row['_sample_key'],1,self.config.get('seed',42)))
                        if any(meta.get(k)!=v for k,v in expected.items()) or meta['output_sha256']!=content_sha256(path):
                            raise ValueError('Prior heldout F0 content or sampler differs')
                        producer_seed = expected['seed']
                    with Image.open(path) as im:
                        if im.size!=(512,512): raise ValueError('F0 geometry differs')
                    row['_f0_path'] = str(path)
                    reuse.append(dict(sample_key=row['_sample_key'], path=str(path), status='accepted',
                                      sha256=content_sha256(path), producer_seed=producer_seed))
                except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                    reuse.append(dict(sample_key=row['_sample_key'],status='rejected',reason=str(exc)))
        write_json_atomic(self.run/'split.json', split)
        write_json_atomic(self.run/'reuse_audit.json',dict(f0=reuse, invalid_cache_action='regenerate; never change model initialization'))
        from tools.sr_v3_real_diagnostic import prepare_real_split
        prepare_real_split(self)
        self.reuse_text_cache(prior)
        self.journal(f'Reused {sum(r["status"]=="accepted" for r in reuse)}/{len(reuse)} verified paired F0 images')

    def reuse_text_cache(self, prior):
        from models.text_embedding_cache import TextEmbeddingCache
        import torch
        torch.set_num_threads(min(8,os.cpu_count() or 1))
        config = self.runtime_config()
        old = TextEmbeddingCache(prior/'text_cache', config, dtype='bf16')
        cache = TextEmbeddingCache(self.root_run/'text_cache', config, dtype='bf16')
        accepted = []; rejected = []
        marker=prior/'completed/cache_text.json'
        recorded=_json(marker).get('artifacts',{}) if marker.exists() else {}
        for row in self.sources():
            record = old.find_record(row['_prompt'],row['_sample_key'],validate_file=True)
            if record is None:
                rejected.append(dict(sample_key=row['_sample_key'],reason='missing/incompatible prompt or encoder signature')); continue
            values = old.load_embedding_file(record)
            source=old.embedding_path(record); target=cache.cache_dir/record['embedding_path']
            if (values['prompt_embeds'].shape[-2:]!=(512,7680) or values['prompt_embeds'].dtype!=torch.bfloat16 or
                not all(torch.isfinite(v).all().item() for v in values.values() if torch.is_tensor(v)) or
                recorded.get(str(source.resolve()))!=content_sha256(source)):
                if target.is_file() and content_sha256(target)==content_sha256(source):
                    target.unlink()
                rejected.append(dict(sample_key=row['_sample_key'],reason='prompt shape/dtype/finite/durable SHA differs')); continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and content_sha256(target)!=content_sha256(source):
                raise ValueError('Text cache target content differs')
            if not target.exists():
                try: os.link(source,target)
                except OSError: shutil.copyfile(source,target)
            if not cache.find_record(row['_prompt'],row['_sample_key'],validate_file=True):
                cache.register_existing_embedding(record,row['_prompt'],row['_sample_key'],row['lq_path'],row['hq_path'])
            accepted.append(dict(sample_key=row['_sample_key'],sha256=content_sha256(target)))
        write_json_atomic(self.run/'text_reuse_audit.json',dict(accepted=accepted,rejected=rejected,
            shape=[1,512,7680], cache_dtype='bf16', online_encoder_dtype='fp32'))

    def cache_text(self):
        self._include_real_for_cache = True
        try:
            import torch
            torch.set_num_threads(min(8,os.cpu_count() or 1))
            from models.text_embedding_cache import TextEmbeddingCache
            cache=TextEmbeddingCache(self.root_run/'text_cache',self.runtime_config(),dtype='bf16')
            integrity=self.run/'text_cache_integrity.json'
            hashes=_json(integrity) if integrity.exists() else {}
            for row in self.sources():
                record=cache.find_record(row['_prompt'],row['_sample_key'],validate_file=True)
                if record is None: continue
                path=cache.embedding_path(record); values=cache.load_embedding_file(record)
                expected=hashes.get(str(path))
                valid=((not expected or expected==content_sha256(path)) and
                       values['prompt_embeds'].shape==(1,512,7680) and values['prompt_embeds'].dtype==torch.bfloat16 and
                       all(torch.isfinite(v).all().item() for v in values.values() if torch.is_tensor(v)))
                if not valid:
                    self.journal(f'Regenerate corrupted prompt cache {row["_sample_key"]}')
                    path.unlink() # sever this cache link; never overwrite the old cache
            super().cache_text()
            cache=TextEmbeddingCache(self.root_run/'text_cache',self.runtime_config(),dtype='bf16')
            hashes={}
            for row in self.sources():
                record=cache.find_record(row['_prompt'],row['_sample_key'],validate_file=True)
                if record is None: raise ValueError('Prompt cache remains incomplete')
                values=cache.load_embedding_file(record)
                if values['prompt_embeds'].shape!=(1,512,7680) or values['prompt_embeds'].dtype!=torch.bfloat16:
                    raise ValueError('Prompt cache must be [1,512,7680] BF16')
                path=cache.embedding_path(record); hashes[str(path)]=content_sha256(path)
            write_json_atomic(integrity,hashes)
        finally: self._include_real_for_cache = False

    def diagnose_real4x(self):
        from tools.sr_v3_real_diagnostic import run_real_diagnostic
        return run_real_diagnostic(self)

    def old_states(self):
        sources={r['lq_path']:r for r in self.sources(('train',))}
        f0_hash=adapter_content_sha256(self.f0_path())
        records=[]; invalid=set()
        for record in read_jsonl(self.baseline/'02_states_for_c.jsonl'):
            row=sources.get(str(Path(record['runtime']['original_lr']).resolve()))
            if row is None: continue
            try: records.append(validate_teacher_state(record,row,f0_hash))
            except (ValueError,KeyError,OSError): invalid.add(row['_sample_key'])
        lineage={(r['sample_key'],r['round_index']):r for r in records}
        for row in sources.values():
            for k in (2,3,4):
                record=lineage.get((row['_sample_key'],k))
                previous=lineage.get((row['_sample_key'],k-1))
                expected_parent=None if k==2 else previous['state_id'] if previous else 'missing'
                if record is None or record['parent_state_id']!=expected_parent: invalid.add(row['_sample_key'])
        if invalid:
            from rl_sr.self_refine import states_from_trajectory
            self._teacher_subset=invalid
            try:
                self.journal(f'Regenerate F0 teacher chains for {len(invalid)} incompatible sources')
                outputs,_=self.rollout('teacher_regenerated',self.snapshot_c0(),names=('train',))
                regenerated=states_from_trajectory(self.sources(('train',)),outputs,f0_hash,f0_hash,
                    identity_sha256(dict(num_steps=25,schedule='linear',init_mode='pure_noise',sigma_start=1.)))
                # Use the same portable router identity convention as baseline
                # states and recompute parent-linked identities after conversion.
                from rl_sr.schema import StateRecord
                normalized=[]; parents={}
                for record in regenerated:
                    value=record.as_dict(); value['state_id']=''
                    value['router_condition_sha256']=value['metadata']['router_condition']['source_hash']
                    value['parent_state_id']=parents.get(value['sample_key'])
                    value=StateRecord.from_dict(value).as_dict()
                    parents[value['sample_key']]=value['state_id']; normalized.append(value)
                regenerated=normalized
            finally: self._teacher_subset=None
            records=[r for r in records if r['sample_key'] not in invalid]+regenerated
        pairs={(r['sample_key'],r['round_index']) for r in records}
        if len(records)!=len(sources)*3 or len(pairs)!=len(records): raise ValueError('Incomplete teacher coverage')
        records.sort(key=lambda r:(r['round_index'],r['runtime']['original_lr']))
        lineage={(r['sample_key'],r['round_index']):r for r in records}
        for record in records:
            expected_parent=None if record['round_index']==2 else lineage[(record['sample_key'],record['round_index']-1)]['state_id']
            if record['parent_state_id']!=expected_parent: raise ValueError('Teacher state parent lineage differs')
        write_jsonl_atomic(self.run/'states_old.jsonl',records)
        return records

    def collect_C0(self):
        self.snapshot_c0(); self.ensure_f0()
        records=self.old_states()
        # At initialization old and new are exactly F0; its training states are already available.
        write_jsonl_atomic(self.run/'states_C0.jsonl',records)
        self.rollout('C0',self.snapshot_c0(),names=('validation','test'))
        self.evaluate('C0')

    def paired_scores(self, outputs):
        import torch
        import pyiqa
        torch.set_num_threads(min(8,os.cpu_count() or 1))
        local={}
        for name,module,relative in [('lpips','lpips','weights/v0.1/alex.pth'),('dists','DISTS_pytorch','weights.pt')]:
            spec=importlib.util.find_spec(module)
            if spec:
                path=Path(spec.origin).parent/relative
                if path.is_file(): local[name]=path
        metrics={m:pyiqa.create_metric(m,device='cpu',
            **({'pretrained_model_path':str(local[m])} if m in local else {}),
            **({'test_y_channel':False} if m in ('psnr','ssim') else {})) for m in METRICS}
        results=[]
        for row,path in outputs:
            prediction=(self.image(path)+1)/2; hr=(self.image(row['hq_path'])+1)/2
            if prediction.shape!=hr.shape: raise ValueError('Candidate metric shape differs')
            with torch.no_grad(): values={m:float(f(prediction,hr).item()) for m,f in metrics.items()}
            if not all(math.isfinite(v) for v in values.values()): raise ValueError('Candidate metric is nonfinite')
            results.append(dict(sample_key=row['_sample_key'],output_sha256=content_sha256(path),
                                hr_sha256=content_sha256(row['hq_path']),**values))
        return results, {m:sum(r[m] for r in results)/len(results) for m in METRICS}

    def validate_checkpoint(self, arm, adapter):
        import torch
        directory=self.run/'checkpoint_validation'/arm
        directory.mkdir(parents=True,exist_ok=True)
        digest=adapter_content_sha256(adapter); model=None; outputs=[]
        for row in self.sources(('validation',)):
            path=directory/f'{row["_sample_key"]}.png'; meta=path.with_suffix('.json')
            expected=dict(adapter_sha256=digest,parent_sha256=content_sha256(row['_f0_path']),
                          lr_sha256=content_sha256(row['lq_path']),prompt_sha256=identity_sha256({'prompt':row['_prompt']}),
                          seed=case_seed(row['_sample_key'],2,self.config.get('seed',42)),nfe=25)
            saved=_json(meta) if meta.exists() else {}
            valid=path.exists() and all(saved.get(k)==v for k,v in expected.items()) and saved.get('output_sha256')==content_sha256(path)
            if not valid:
                if model is None: model=self.artist(adapter)
                output,seed=self.proposal(model,row,row['_f0_path'],2)
                self.save_image(output,path); write_json_atomic(meta,{**expected,'output_sha256':content_sha256(path)})
            outputs.append((row,path))
        del model; gc.collect(); torch.cuda.empty_cache()
        rows,score=self.paired_scores(outputs)
        self.write_csv(directory/'metrics.csv',rows)
        write_json_atomic(directory/'score.json',dict(adapter=str(adapter),adapter_sha256=digest,metrics=score,
                         selection_split='validation',round=2,nfe=25))
        return score

    def sft(self, arm, manifest, init_adapter, steps):
        directory=self.run/'generators'/arm
        selected=directory/'selected_checkpoint.json'
        if selected.exists():
            result=_json(selected)
            if (result['manifest_sha256']==content_sha256(manifest) and
                result['init_sha256']==adapter_content_sha256(init_adapter) and result['updates']==steps and
                adapter_content_sha256(result['adapter'])==result['adapter_sha256']): return Path(result['adapter'])
            raise ValueError('Selected checkpoint provenance changed')
        # Both arms have an exactly identical initial buffer/update sequence.
        # Share this computation, then branch into static vs refreshed states.
        common=self.run/'generators/A_cycle0/selected_checkpoint.json'
        if arm=='B_cycle0' and common.exists():
            result=_json(common)
            if (result['manifest_sha256']==content_sha256(manifest) and result['updates']==steps and
                result['init_sha256']==adapter_content_sha256(init_adapter) and
                adapter_content_sha256(result['adapter'])==result['adapter_sha256']):
                directory.mkdir(parents=True,exist_ok=True)
                write_json_atomic(selected,{**result,'reused_identical_A_cycle0':True})
                self.journal('B cycle0 reuses the exactly identical validated A cycle0 computation')
                return Path(result['adapter'])
        final=super().sft(arm,manifest,init_adapter,steps)
        # Accelerate/BF16 wrappers may form cycles; release the training model
        # before loading a separate inference model for actual rollout checks.
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        candidates=sorted(directory.glob('checkpoint-*/rg_flux_adapters'))
        if adapter_content_sha256(final) not in {adapter_content_sha256(p) for p in candidates}:
            candidates.append(final)
        scores=[]; baseline=_json(self.run/'metrics/C0.json')['validation']['1']
        threshold=self.config.get('quality_thresholds',{})
        for adapter in candidates:
            score=self.validate_checkpoint(f'{arm}_{adapter.parent.name}',adapter)
            fidelity=score['psnr']>=baseline['psnr']-threshold.get('psnr_tolerance',.05) and score['ssim']>=baseline['ssim']-threshold.get('ssim_tolerance',.002)
            scores.append(dict(adapter=str(adapter),metrics=score,fidelity_ok=fidelity))
        eligible=[r for r in scores if r['fidelity_ok'] and r['metrics']['lpips']<=baseline['lpips'] and r['metrics']['dists']<=baseline['dists']]
        best=min(eligible,key=lambda r:r['metrics']['lpips']+r['metrics']['dists']) if eligible else None
        chosen=Path(best['adapter']) if best else Path(init_adapter)
        result=dict(adapter=str(chosen),adapter_sha256=adapter_content_sha256(chosen),
                    manifest_sha256=content_sha256(manifest),init_sha256=adapter_content_sha256(init_adapter),
                    updates=steps,candidates=scores,validation_first_round=baseline,
                    selected_trained_checkpoint=best is not None,
                    decision='validated_candidate' if best else 'retain_initializer_candidates_degraded')
        write_json_atomic(selected,result)
        self.journal(f'{arm} checkpoint decision: {result["decision"]}')
        return chosen

    def train_B(self):
        import torch
        cycles=int(self.config.get('refresh_cycles',2)); budget=self.generator_budget()
        old=read_jsonl(self.run/'states_old.jsonl'); initial=self.snapshot_c0(); own=read_jsonl(self.run/'states_C0.jsonl')
        for cycle in range(cycles):
            mixture=mixed_states(old,own,float(self.config.get('self_state_fraction',.5)),int(self.config.get('seed',42))+17*cycle)
            manifest=self.run/f'states_B_cycle{cycle}.jsonl'; write_jsonl_atomic(manifest,mixture)
            initial=self.sft(f'B_cycle{cycle}',manifest,initial,budget//cycles+(cycle<budget%cycles))
            gc.collect(); torch.cuda.empty_cache()
            if cycle+1<cycles:
                arm=f'B_refresh{cycle}'
                self.rollout(arm,initial,names=('train',)); path,_=self.self_states(arm,initial)
                own=read_jsonl(path)
        write_json_atomic(self.run/'generator_B.json',dict(adapter=str(initial),cycles=cycles,updates=budget,
                          shared_initial_computation=True))

    def select_actor(self):
        threshold=self.config.get('quality_thresholds',{})
        scores={a:_json(self.run/f'metrics/{a}.json')['validation'] for a in ('C0','A','B')}
        first=scores['C0']['1']
        def useful(a): return passes_first_round_gate(first,scores[a]['4'],scores[a]['2'],threshold)
        b_vs_a=(scores['B']['4']['lpips']<scores['A']['4']['lpips'] and scores['B']['4']['dists']<scores['A']['4']['dists'])
        selected='B' if useful('B') and b_vs_a else 'A' if useful('A') else 'C0'
        result=dict(selected_actor=selected,adapter=str(self.actor_path(selected)),
                    self_trajectory_beats_matched_A=b_vs_a,validation_only=True,
                    raw_actor_beats_F0_first_round={a:useful(a) for a in ('A','B')},
                    fallback_reason='retain_F0_when_raw_training_fails_first_round_gate' if selected=='C0' else None)
        write_json_atomic(self.run/'selected_actor.json',result)
        return result

    def verdict(self):
        result=super().verdict()
        actor=result['selected_actor']['selected_actor']; threshold=self.config.get('quality_thresholds',{})
        scores={a:_json(self.run/f'metrics/{a}.json') for a in ('C0','A','B','gate_full')}
        gate=passes_first_round_gate(scores['C0']['validation']['1'],scores['gate_full']['validation']['4'],
                                    scores[actor]['validation']['2'],threshold)
        gains=self.paired_gain('C0',1,'gate_full',4,'validation')
        confident=gains['lpips_ci95'][0]>0 and gains['dists_ci95'][0]>0
        result.update(validation_beats_F0_first_round=gate,paired_vs_F0_first_round=gains,
                      scale_ok=result['scale_ok'] and gate and confident)
        result['status']='positive_validation' if result['scale_ok'] else 'validation_failed_no_scale'
        if not result['scale_ok']: result['next_scale_sources']=None
        write_json_atomic(self.run/'verdict.json',result)
        report=self.run/'report.md'
        text=report.read_text()
        text=text.replace(text.splitlines()[2],f'Status: {result["status"]}; training sources: {self.level}.')
        report.write_text(text)
        with report.open('a') as f:
            f.write('\n## Mandatory F0 first-round reference\n\n')
            f.write(f'Final v3 status: {result["status"]}. Expansion: {result["next_scale_sources"]}.\n\n')
            f.write(f'F0 round1 validation: {scores["C0"]["validation"]["1"]}\n\n')
            f.write(f'Gate-full round4 versus F0 round1 passes: {gate}; paired confidence: {confident}.\n')
        return result

    def artifact_paths(self, stage):
        if stage=='diagnose_real4x':
            paths=[p for p in (self.run/'real4x').rglob('*') if p.is_file()]
            paths += [self.run/'real_split.json',Path(self.config['real_source_jsonl'])]
            paths += [Path(r['lq_path']) for r in _json(self.run/'real_split.json')]
            paths += [p for p in Path(self.config['negative_adapter']).rglob('*') if p.is_file()]
            return sorted(set(paths))
        paths=super().artifact_paths(stage)
        # Every stage binds its immutable data assignment as well as its own
        # outputs; a regenerated split cannot silently reuse later markers.
        paths += [self.run/'split_core.json',Path(self.config['source_jsonl'])]
        if stage=='prepare': paths += [self.run/'reuse_audit.json',self.run/'text_reuse_audit.json',self.run/'real_split.json']
        if stage=='collect_C0':
            for record in read_jsonl(self.run/'states_old.jsonl'):
                paths += [Path(record['runtime'][f]) for f in ('current_sr','original_lr','hr')]
            paths += [p for p in (self.run/'trajectories/teacher_regenerated').rglob('*') if p.is_file()]
        if stage=='cache_text':
            paths.append(self.run/'text_cache_integrity.json')
            from models.text_embedding_cache import get_text_embedding_cache
            cache=get_text_embedding_cache(self.runtime_config(),dtype='bf16')
            for row in _json(self.run/'real_split.json'):
                record=cache.find_record(row['_prompt'],row['_sample_key'],validate_file=True)
                if record is None: raise ValueError('Real diagnostic prompt cache incomplete')
                paths.append(cache.embedding_path(record))
        if stage in ('train_A','train_B'):
            for directory in (self.run/'checkpoint_validation').glob(f'{stage[-1]}_cycle*'):
                paths += [p for p in directory.rglob('*') if p.is_file()]
            selected=_json(self.run/f'generator_{stage[-1]}.json')['adapter']
            paths += [p for p in Path(selected).rglob('*') if p.is_file()]
        return sorted(set(paths))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--run_dir',required=True)
    parser.add_argument('--stage',required=True); parser.add_argument('--level',type=int)
    parser.add_argument('--batch_size',type=int)
    args=parser.parse_args()
    StableExperiment(yaml.safe_load(Path(args.config).read_text()),args.run_dir,args.level,args.batch_size).run_stage(args.stage)


if __name__=='__main__': main()
