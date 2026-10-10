import importlib.util
import pytest
import torch


def data_module():
    assert importlib.util.find_spec('rl_sr.observation_loop_data') is not None, 'data module not implemented'
    from rl_sr import observation_loop_data
    return observation_loop_data


def model_module():
    assert importlib.util.find_spec('models.sr_observation_loop') is not None, 'spatial correction module not implemented'
    from models import sr_observation_loop
    return sr_observation_loop


def test_grouped_split_excludes_all_previous_holdout_and_baseline_crops():
    rows=[dict(lq_path=f'/tmp/lr{i}',hq_path=f'/tmp/hr{i}',source_hq_path=f'/tmp/source{i//2}') for i in range(30)]
    split=data_module().select_sources(rows,[r['lq_path'] for r in rows[:8]],{'/tmp/source4'},(4,3,3),17)
    groups={k:{r['source_hq_path'] for r in v} for k,v in split.items()}
    assert [len(split[k]) for k in ('train','validation','test')]==[4,3,3]
    assert not (groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
    assert '/tmp/source4' not in groups['validation']|groups['test']
    assert not ({f'/tmp/source{i}' for i in range(4)} & (groups['validation']|groups['test']))
    with pytest.raises(ValueError): data_module().select_sources(rows,[],set(),(1,2,2),17)


def test_spatial_zero_exit_preserves_hidden_but_has_initial_gradient():
    branch=model_module().SpatialLRResidual(12,latent_channels=4,attention_width=8,grid=4)
    h=torch.randn(2,16,12); lr=torch.randn(2,4,4,4)
    delta=branch(h,lr,(4,4))
    assert torch.count_nonzero(delta)==0
    loss=((h+delta)*torch.randn_like(h)).sum();loss.backward()
    assert branch.output.weight.grad.abs().sum()>0


def test_spatial_lr_changes_after_update_and_supports_target_geometry():
    branch=model_module().SpatialLRResidual(12,latent_channels=4,attention_width=8,grid=4)
    h=torch.randn(2,16,12); lr=torch.randn(2,4,4,4)
    opt=torch.optim.Adam(branch.parameters(),lr=.01)
    ((h+branch(h,lr,(4,4)))*torch.randn_like(h)).sum().backward(); opt.step()
    assert not torch.allclose(branch(h,lr,(4,4)),branch(h,lr.flip(0),(4,4)))
    with pytest.raises(ValueError): branch(h,lr,(3,4))


def test_shared_corrector_identity_gradient_and_actual_recursion():
    torch.manual_seed(4)
    m=model_module().ObservationLoop(width=16,depth=1,heads=4)
    current=torch.randn(2,3,32,32).clamp(-1,1); lr=torch.randn_like(current)
    output=m(current,lr,torch.tensor([2,3]))
    assert torch.equal(output['image'],current)
    optimizer=torch.optim.Adam(m.parameters(),lr=.01)
    (output['image']-torch.zeros_like(current)).square().mean().backward()
    assert m.residual_head.weight.grad.abs().sum()>0
    optimizer.step();optimizer.zero_grad()
    out2=m(current,lr,2)
    out3=m(out2['image'],lr,3,state=out2['state'])
    assert out3['image'].shape==current.shape and torch.isfinite(out3['image']).all()
    out3['image'].square().mean().backward()
    assert m.current_encoder[0].weight.grad.abs().sum()>0
    changed=m(current,lr.flip(0),2)
    assert not torch.allclose(out2['image'],changed['image'])
    assert not torch.allclose(m(current,lr,2)['image'],m(current,lr,4)['image'])


def test_corrector_state_validation_and_per_case_isolation():
    m=model_module().ObservationLoop(width=16,depth=1,heads=4)
    m.residual_head.weight.data.normal_(std=.01)
    x=torch.rand(2,3,32,32);lr=torch.rand_like(x)
    batch=m(x,lr,torch.tensor([2,4]))
    separate=torch.cat([m(x[i:i+1],lr[i:i+1],k)['image'] for i,k in enumerate([2,4])])
    torch.testing.assert_close(batch['image'],separate,atol=1e-6,rtol=1e-5)
    with pytest.raises(ValueError): m(x,lr,0)
    with pytest.raises(ValueError): m(x,lr[:1],2)
    with pytest.raises(ValueError): m(x,lr,2,state=torch.zeros(2,3,3,3))


def test_selector_only_ranks_validation_and_records_failure():
    mod=data_module()
    base=dict(psnr=20.,ssim=.6,lpips=.3,dists=.2)
    candidates=[dict(name='bad',split='validation',psnr=18.,ssim=.5,lpips=.1,dists=.1),
                dict(name='good',split='validation',psnr=20.1,ssim=.61,lpips=.299,dists=.199)]
    selected=mod.select_checkpoint(base,candidates)
    assert selected['selected']=='good' and selected['quality_passed']
    assert not mod.select_checkpoint(base,candidates[:1])['quality_passed']
    with pytest.raises(ValueError):mod.select_checkpoint(base,[{**candidates[1],'split':'test'}])


def test_flux_hook_bypasses_f0_and_only_updates_target_positions():
    mod=model_module()
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.transformer=torch.nn.Module();self.transformer.x_embedder=torch.nn.Linear(4,12)
        def forward(self,z_t,z_lr,z_anchor_lr=None,refinement_round=None):
            return self.transformer.x_embedder(torch.cat([z_t.flatten(2).transpose(1,2),z_lr.flatten(2).transpose(1,2)],1))
    actor=Actor();branch=mod.SpatialLRResidual(12,4,8,4)
    branch.output.weight.data.normal_(std=.1)
    z=torch.randn(2,4,4,4);prev=torch.randn_like(z);anchor=torch.randn_like(z)
    base=actor(z,prev);hook=mod.FluxSpatialHooks(actor,branch)
    assert torch.equal(actor(z,prev,z_anchor_lr=anchor),base)
    refined=actor(z,prev,z_anchor_lr=anchor,refinement_round=2)
    assert not torch.allclose(refined[:,:16],base[:,:16])
    assert torch.equal(refined[:,16:],base[:,16:])
    hook.close();assert torch.equal(actor(z,prev,z_anchor_lr=anchor,refinement_round=2),base)


def test_actual_unroll_supervision_backpropagates_without_ground_truth_input():
    assert importlib.util.find_spec('tools.sr_observation_loop_training') is not None,'unroll trainer not implemented'
    from tools.sr_observation_loop_training import unroll_loss, rollout_tensor
    model=model_module().ObservationLoop(width=16,depth=1,heads=4)
    y=torch.rand(2,3,32,32)*2-1;lr=torch.rand_like(y);gt=torch.rand_like(y)
    loss,outputs=unroll_loss(model,y,lr,gt,3,dict(lr_mode='spatial',use_round=True,use_state=True),.05,.1)
    assert len(outputs)==3
    loss.backward();assert model.residual_head.weight.grad.abs().sum()>0
    with torch.no_grad(): prediction=rollout_tensor(model,y,lr,3,dict(lr_mode='spatial',use_round=True,use_state=True))
    for actual,expected in zip(outputs,prediction):torch.testing.assert_close(actual,expected)


def test_refreshed_states_keep_round_labels_and_stop_at_round_four():
    from tools.sr_observation_loop_training import rollout_tensor
    class Step(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=[]
        def forward(self,current,original_lr,rounds,state=None,**kwargs):
            self.calls.append(rounds.clone())
            return dict(image=current+.1,state=current)
    model=Step();y=torch.zeros(2,3,16,16)
    outputs=rollout_tensor(model,y,y,3,{},start_rounds=torch.tensor([3,4]),maximum_round=4)
    torch.testing.assert_close(outputs[-1][:,0,0,0],torch.tensor([.2,.1]))
    assert model.calls[0].tolist()==[3,4] and max(int(v.max()) for v in model.calls)<=4


def test_oom_backoff_preserves_effective_training_batch():
    assert importlib.util.find_spec('tools.watch_sr_observation_loop') is not None,'supervisor not implemented'
    from tools.watch_sr_observation_loop import backoff
    cfg=dict(loop_batch=256,loop_micro_batch=256,loop_eval_batch=16,flux_batch=20,flux_micro_batch=20,inference_batch=8)
    assert backoff(cfg,'correctors') and cfg['loop_batch']==256 and cfg['loop_micro_batch']==128
    assert backoff(cfg,'flux_spatial') and cfg['flux_batch']==20 and cfg['flux_micro_batch']==10 and cfg['inference_batch']==4
    cfg.update(loop_micro_batch=1,loop_eval_batch=1)
    assert not backoff(cfg,'correctors')


def test_checkpoint_selection_considers_recursive_terminal_quality():
    mod=data_module();base=dict(psnr=20.,ssim=.6,lpips=.3,dists=.2)
    candidates=[dict(name='early_only',split='validation',psnr=20.3,ssim=.61,lpips=.299,dists=.199,
                    round4=dict(psnr=19.,ssim=.5,lpips=.4,dists=.3)),
                dict(name='recursive',split='validation',psnr=20.1,ssim=.61,lpips=.299,dists=.199,
                    round4=dict(psnr=20.15,ssim=.61,lpips=.298,dists=.198))]
    assert mod.select_checkpoint(base,candidates)['selected']=='recursive'


def test_artist_uses_device_aware_to_for_unregistered_flux_vae(monkeypatch,tmp_path):
    from tools.run_sr_observation_loop import Experiment
    from models import rg_flux_artist_factory
    class Spy(torch.nn.Module):
        def __init__(self):super().__init__();self.device_to=False
        def to(self,*args,**kwargs):self.device_to=kwargs.get('device')=='cuda';return self
        def cuda(self,*args,**kwargs):return self # nn.Module.cuda does not call artist.to
        def load_trainable(self,*args,**kwargs):pass
        def align_inference_dtype(self,**kwargs):pass
        def set_moe_inference_schedule(self):pass
    spy=Spy();monkeypatch.setattr(rg_flux_artist_factory,'build_rg_flux_artist',lambda cfg:spy)
    exp=Experiment(dict(f0_adapter='unused'),tmp_path);exp.runtime=lambda:{}
    assert exp.artist().device_to,'VAE must move via Flux2KleinSRArtist.to'


def test_phase_recovery_and_expected_refresh_checkpoint():
    from tools.sr_observation_loop_training import pending_work, expected_refresh_update
    assert pending_work(400,1200,200,400,[],None)==['evaluate','refresh']
    assert pending_work(400,1200,200,400,[{'update':400}],400)==[]
    assert pending_work(500,1200,200,400,[{'update':400}],None)==['refresh']
    assert expected_refresh_update(1200,1200,400)==800


def test_refresh_checks_exact_producer_sources_content_and_hidden():
    from tools.sr_observation_loop_training import refresh_payload,validate_refresh
    images=torch.rand(2,3,32,32);hidden=torch.rand(2,16,2,2).bfloat16();rounds=torch.tensor([2,3])
    payload=refresh_payload(images,hidden,rounds,400,'producer','split',['a','b'])
    validate_refresh(payload,400,'producer','split',['a','b'],16,32)
    for changed in ({'producer_sha256':'wrong'},{'source_keys':['b','a']},{'hidden':hidden+1},{'rounds':torch.tensor([2,4])}):
        with pytest.raises(ValueError):validate_refresh({**payload,**changed},400,'producer','split',['a','b'],16,32)


def test_refresh_hidden_state_enters_first_recursive_step():
    from tools.sr_observation_loop_training import rollout_tensor
    class Step(torch.nn.Module):
        def forward(self,current,lr,rounds,state=None,**kwargs):
            assert state is not None
            return {'image':current+state,'state':state}
    x=torch.zeros(1,3,16,16);s=torch.ones_like(x)*.1
    result=rollout_tensor(Step(),x,x,1,{},start_rounds=torch.tensor([4]),initial_state=s)
    torch.testing.assert_close(result[0],s)


def test_completed_chain_reused_without_rewriting_and_corruption_rejected(tmp_path):
    from tools.sr_observation_loop_training import commit_chain, read_chain
    folder=tmp_path/'arm';folder.mkdir();signature={'checkpoint':'policy','lr':'observation'}
    images=[torch.zeros(3,16,16),torch.ones(3,16,16)*.1]
    rows=commit_chain(folder,'sample',images,signature)
    before=[__import__('pathlib').Path(r['path']).stat().st_mtime_ns for r in rows]
    assert read_chain(folder,'sample',signature)==rows
    assert before==[__import__('pathlib').Path(r['path']).stat().st_mtime_ns for r in rows]
    with pytest.raises(ValueError):read_chain(folder,'sample',{'checkpoint':'different','lr':'observation'})
    __import__('pathlib').Path(rows[0]['path']).write_bytes(b'corrupted')
    with pytest.raises(ValueError):read_chain(folder,'sample',signature)


def test_supervisor_process_identity_rejects_reused_pid(tmp_path):
    import os
    from tools.watch_sr_observation_loop import process_identity, same_child
    info=process_identity(os.getpid())
    assert same_child(info)
    assert not same_child({**info,'start_ticks':info['start_ticks']+1})
    assert not same_child({**info,'command':['other-command']})


def test_round_conditioning_training_schedule_covers_all_later_rounds():
    from tools.sr_observation_loop_training import single_step_rounds
    assert set(single_step_rounds(9,1))=={2,3,4}
    assert single_step_rounds(9,1)==single_step_rounds(9,1)
    assert single_step_rounds(9,2)!=single_step_rounds(9,1)


def test_terminal_nonfinite_cannot_enter_checkpoint_ranking():
    base=dict(psnr=20.,ssim=.6,lpips=.3,dists=.2)
    candidate=dict(name='bad',split='validation',**base,round4={**base,'psnr':float('nan')})
    with pytest.raises(ValueError):data_module().select_checkpoint(base,[candidate])


def test_flux_spatial_injection_can_be_disabled_for_frozen_ablation():
    mod=model_module()
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.transformer=torch.nn.Module();self.transformer.x_embedder=torch.nn.Linear(4,12)
        def forward(self,z_t,z_lr,z_anchor_lr=None,refinement_round=None):
            return self.transformer.x_embedder(z_t.flatten(2).transpose(1,2))
    actor=Actor();branch=mod.SpatialLRResidual(12,4,8,4);branch.output.weight.data.normal_(std=.1)
    z=torch.randn(1,4,4,4);base=actor(z,z);hook=mod.FluxSpatialHooks(actor,branch)
    assert not torch.allclose(actor(z,z,z_anchor_lr=z,refinement_round=2),base)
    hook.enabled=False
    torch.testing.assert_close(actor(z,z,z_anchor_lr=z,refinement_round=2),base,rtol=0,atol=0)
    hook.close()


def test_new_holdout_includes_frozen_generic_multiround_control(tmp_path):
    from tools.run_sr_observation_loop import Experiment
    exp=Experiment({'f0_sha256':'f0'},tmp_path);seen={}
    exp.artist=lambda:torch.nn.Linear(1,1)
    exp.embeds=lambda:{}
    exp.rows=lambda split:[{'split':split}]
    def generate(model,arm,rows,embeds,max_round):
        seen.update(arm=arm,max_round=max_round,rows=rows,checkpoint=exp.active_checkpoint)
        return []
    exp.generate=generate;exp.score=lambda arm,outputs:[]
    exp.flux_generic()
    assert seen==dict(arm='G_generic',max_round=4,rows=[{'split':'validation'},{'split':'test'}],checkpoint=None)
    assert (tmp_path/'flux/generic/result.json').exists()
