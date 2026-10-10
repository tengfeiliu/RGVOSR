import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from rl_sr.schema import content_sha256


def state_and_source(tmp_path):
    from rl_sr.schema import StateRecord, RuntimePaths, build_sample_key, identity_sha256
    paths = {}
    for name in ('lr', 'hr', 'current'):
        path = tmp_path / f'{name}.png'
        Image.new('RGB', (512, 512), {'lr':'gray', 'hr':'white', 'current':'red'}[name]).save(path)
        paths[name] = str(path)
    digest = content_sha256(paths['lr'])
    source = dict(lq_path=paths['lr'], hq_path=paths['hr'], _prompt='restore',
                  _sample_key=build_sample_key('lsdir_precrop512_v1', f'original_lr/sha256/{digest}', digest))
    source['_router_condition'] = dict(source_hash='route',values=[.5])
    state = StateRecord(dataset_id='lsdir_precrop512_v1',original_lr_key=f'original_lr/sha256/{digest}',
                 round_index=2, sample_key=source['_sample_key'], prompt='restore',
                 producer_adapter_sha256='f0', original_lr_sha256=digest,
                 hr_sha256=content_sha256(paths['hr']), current_output_sha256=content_sha256(paths['current']),
                 parent_state_id=None,geometry_sha256=identity_sha256(dict(mode='RGB',width=512,height=512)),
                 sampler_sha256=identity_sha256(dict(num_steps=25,schedule='linear',init_mode='pure_noise',sigma_start=1.)),
                 router_condition_sha256='route',metadata=dict(router_condition=source['_router_condition']),
                 runtime=RuntimePaths(original_lr=paths['lr'], hr=paths['hr'], current_sr=paths['current'])).as_dict()
    return source, state


def test_reuse_checks_hr_prompt_and_policy(tmp_path):
    from rl_sr.stable_reuse import validate_teacher_state
    source, state = state_and_source(tmp_path)
    assert validate_teacher_state(state, source, 'f0') == state
    for field, value in [('prompt','changed'), ('producer_adapter_sha256','failed-C'), ('hr_sha256','bad'),
                         ('sampler_sha256','bad'),('router_condition_sha256','bad'),('state_id','bad')]:
        bad = copy.deepcopy(state); bad[field] = value
        with pytest.raises(ValueError): validate_teacher_state(bad, source, 'f0')
    Image.new('RGB',(512,512),'blue').save(state['runtime']['current_sr'])
    with pytest.raises(ValueError): validate_teacher_state(state, source, 'f0')


def test_first_refresh_has_exact_static_state_order():
    from rl_sr.stable_reuse import mixed_states
    old = [dict(round_index=k, runtime={'original_lr':str(i)}, version='old') for k in (2,3,4) for i in range(8)]
    assert mixed_states(old, copy.deepcopy(old), .5, 42) == old
    own = [{**r, 'version':'own'} for r in old]
    mixed = mixed_states(old, own, .5, 42)
    assert [r['runtime']['original_lr'] for r in mixed] == [r['runtime']['original_lr'] for r in old]
    assert sum(r['version']=='own' for r in mixed) == 12
    with pytest.raises(ValueError): mixed_states(old, own[:-1], .5, 42)


def test_scale_requires_f0_first_round_and_actual_early_stop():
    from rl_sr.stable_reuse import passes_first_round_gate
    first=dict(psnr=20,ssim=.5,lpips=.2,dists=.1)
    failed=dict(psnr=18,ssim=.4,lpips=.4,dists=.3)
    apparent=dict(psnr=19,ssim=.45,lpips=.3,dists=.2)
    assert not passes_first_round_gate(first, apparent, failed, {})
    good=dict(psnr=20,ssim=.5,lpips=.19,dists=.09)
    assert passes_first_round_gate(first, good, first, {})
    assert not passes_first_round_gate(first, good, dict(psnr=20,ssim=.5,lpips=.18,dists=.08), {})


def test_v3_initial_snapshot_is_uploaded_f0_without_completed_c(tmp_path, monkeypatch):
    from tools.run_sr_stable_experiment import StableExperiment
    experiment=object.__new__(StableExperiment)
    experiment.root_run=tmp_path
    f0=tmp_path/'uploaded'; f0.mkdir(); (f0/'adapter.pt').write_bytes(b'uploaded36k')
    monkeypatch.setattr(experiment, 'f0_path', lambda: f0)
    adapter=experiment.snapshot_c0()
    assert adapter == tmp_path/'C0/rg_flux_adapters'
    assert (adapter/'adapter.pt').read_bytes() == b'uploaded36k'


def test_two_cycle_training_reads_refreshed_state_records_as_dicts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rl_sr.artifact_store import write_jsonl_atomic, read_jsonl
    from tools.run_sr_stable_experiment import StableExperiment
    experiment=object.__new__(StableExperiment)
    experiment.run=tmp_path
    experiment.config=dict(refresh_cycles=2,self_state_fraction=.5,seed=42)
    old=[dict(round_index=k,runtime={'original_lr':str(i)},version='old') for k in (2,3,4) for i in range(8)]
    own=[{**r,'version':'self'} for r in old]
    write_jsonl_atomic(tmp_path/'states_old.jsonl',old)
    write_jsonl_atomic(tmp_path/'states_C0.jsonl',old)
    path=tmp_path/'refreshed.jsonl'; write_jsonl_atomic(path,own)
    calls=[]
    monkeypatch.setattr(experiment,'generator_budget',lambda:100)
    monkeypatch.setattr(experiment,'snapshot_c0',lambda:tmp_path/'F0')
    monkeypatch.setattr(experiment,'rollout',lambda *a,**kw:None)
    monkeypatch.setattr(experiment,'self_states',lambda *a:(path,[SimpleNamespace() for _ in own]))
    def fake_sft(arm,manifest,initial,steps):
        calls.append((arm,read_jsonl(manifest),steps)); return initial
    monkeypatch.setattr(experiment,'sft',fake_sft)
    experiment.train_B()
    assert calls[0][1]==old
    assert sum(r['version']=='self' for r in calls[1][1])==12
    assert calls[0][2]==calls[1][2]==50


def test_corrupt_teacher_rebuilds_complete_dict_lineage(tmp_path, monkeypatch):
    from rl_sr.schema import StateRecord
    from rl_sr.artifact_store import write_jsonl_atomic
    import tools.run_sr_stable_experiment as runner
    source,state=state_and_source(tmp_path)
    experiment=object.__new__(runner.StableExperiment)
    experiment.run=tmp_path/'run'; experiment.run.mkdir()
    experiment.baseline=tmp_path/'baseline'; experiment.baseline.mkdir()
    experiment.config={'seed':42}
    Image.new('RGB',(512,512),'blue').save(state['runtime']['current_sr'])
    write_jsonl_atomic(experiment.baseline/'02_states_for_c.jsonl',[state])
    monkeypatch.setattr(experiment,'sources',lambda names=('train',):[source])
    monkeypatch.setattr(experiment,'f0_path',lambda:tmp_path)
    monkeypatch.setattr(experiment,'snapshot_c0',lambda:tmp_path)
    monkeypatch.setattr(experiment,'journal',lambda text:None)
    monkeypatch.setattr(runner,'adapter_content_sha256',lambda path:'f0')
    trajectories=[]
    for k in (1,2,3,4):
        path=tmp_path/f'new{k}.png'; Image.new('RGB',(512,512),'gray').save(path)
        trajectories.append(dict(sample_key=source['_sample_key'],round=k,path=str(path)))
    monkeypatch.setattr(experiment,'rollout',lambda *a,**kw:(trajectories,'policy'))
    rebuilt=experiment.old_states()
    assert len(rebuilt)==3
    from rl_sr.stable_reuse import validate_teacher_state
    for i,state in enumerate(rebuilt):
        assert isinstance(state,dict)
        validate_teacher_state(state,source,'f0')
        assert state['parent_state_id']==(None if i==0 else rebuilt[i-1]['state_id'])


@pytest.mark.parametrize('bad_value',[1.0,float('nan')])
def test_text_reuse_requires_prior_durable_bytes_and_finite_values(tmp_path, monkeypatch, bad_value):
    import os
    import torch
    from models.text_embedding_cache import TextEmbeddingCache
    from rl_sr.artifact_store import write_json_atomic
    from tools.run_sr_stable_experiment import StableExperiment
    source,_=state_and_source(tmp_path)
    config=dict(model=dict(flux_backend='flux2_klein',max_prompt_sequence_length=512,text_encoder_dtype='fp32'),
                condition=dict(prompt_variant='iqa_suggestion',include_caption=True))
    prior=tmp_path/'prior'
    old=TextEmbeddingCache(prior/'text_cache',config,dtype='bf16')
    record=old.save_embedding(source['_prompt'],source['_sample_key'],source['lq_path'],source['hq_path'],
        dict(prompt_embeds=torch.zeros(1,512,7680,dtype=torch.bfloat16),pooled_prompt_embeds=torch.zeros(1,0),text_ids=torch.zeros(512,4)))
    path=old.embedding_path(record)
    write_json_atomic(prior/'completed/cache_text.json',dict(artifacts={str(path.resolve()):content_sha256(path)}))
    experiment=object.__new__(StableExperiment)
    experiment.root_run=experiment.run=tmp_path/'new'; experiment.run.mkdir()
    monkeypatch.setattr(experiment,'sources',lambda:[source])
    monkeypatch.setattr(experiment,'runtime_config',lambda:config)
    target=experiment.root_run/'text_cache'/record['embedding_path']; target.parent.mkdir(parents=True)
    os.link(path,target)
    values=torch.load(path,weights_only=False); values['prompt_embeds'][0,0,0]=bad_value
    torch.save(values,path)
    experiment.reuse_text_cache(prior)
    audit=json.loads((experiment.run/'text_reuse_audit.json').read_text())
    assert audit['accepted']==[] and len(audit['rejected'])==1
    assert not target.exists()
