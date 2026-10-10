"""Frozen FLUX.2: matched weak-text versus image-space LR adapters."""
from pathlib import Path
import json
import random
import time
import torch
from torch.nn import functional as F

from models.sr_observation_loop import SpatialLRResidual,FluxSpatialHooks
from models.sr_round_state_adapter import hold_moe_routing
from rl_sr.artifact_store import write_json_atomic
from rl_sr.observation_loop_data import select_checkpoint
from rl_sr.schema import content_sha256
from tools.run_sr_round_state_quick import atomic_torch,image


def configure(exp,mode):
    model=exp.artist();hook=None
    if mode=='spatial':
        branch=SpatialLRResidual(model.transformer.x_embedder.out_features,
            attention_width=exp.cfg['flux_attention_width'],grid=exp.cfg['flux_spatial_grid']).cuda()
        hook=FluxSpatialHooks(model,branch)
        parameters=list(branch.parameters())
    else:
        branch=model.refinement_condition_adapter
        branch.to(dtype=torch.float32);branch.round_embed.to(dtype=torch.bfloat16)
        parameters=[p for name,p in branch.named_parameters() if name.startswith('anchor_')]
    for parameter in parameters:parameter.requires_grad_(True)
    return model,branch,hook,parameters


def load_branch(exp,model,branch,path,mode):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['f0_sha256']!=exp.cfg['f0_sha256'] or payload['split_sha256']!=exp.split_sha() or payload['mode']!=mode:
        raise ValueError('FLUX branch checkpoint provenance mismatch')
    branch.load_state_dict(payload['branch'])
    return payload


def train_flux(exp,mode):
    from rg_flux_fm import build_flow_matching_inputs
    from rl_sr.conditioning import state_router_condition_tensors
    folder=exp.run/'flux'/mode;folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/'result.json'
    if result_path.exists():return
    torch.manual_seed(exp.cfg['seed']);torch.cuda.manual_seed_all(exp.cfg['seed'])
    model,branch,hook,parameters=configure(exp,mode)
    optimizer=torch.optim.AdamW(parameters,lr=exp.cfg['flux_lr'],weight_decay=.01)
    embeds=exp.embeds();rows=exp.rows('train');start=0;candidates=[];history=[]
    resume=folder/'resume.pt'
    if resume.exists():
        payload=load_branch(exp,model,branch,resume,mode);optimizer.load_state_dict(payload['optimizer'])
        start=payload['update'];candidates=payload['candidates'];history=payload['history']
        torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state(payload['cuda_rng'])
    def save(path,step,resumable=False):
        payload=dict(branch={k:v.detach().cpu() for k,v in branch.state_dict().items()},mode=mode,update=step,
            f0_sha256=exp.cfg['f0_sha256'],split_sha256=exp.split_sha(),parameters=sum(p.numel() for p in parameters))
        if resumable:payload.update(optimizer=optimizer.state_dict(),candidates=candidates,history=history,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        atomic_torch(path,payload)
    def finish_evaluation(step):
        if not step or (step%exp.cfg['flux_checkpoint_every'] and step!=exp.cfg['flux_updates']):return
        if any(c['update']==step for c in candidates):return
        checkpoint=folder/f'checkpoint-{step:05d}.pt'
        if not checkpoint.exists():save(checkpoint,step)
        exp.active_checkpoint=checkpoint;exp.active_mode=mode
        outputs=exp.generate(model,f'G_{mode}_check{step}',exp.rows('validation')[:exp.cfg['flux_eval_sources']],embeds,max_round=2)
        metrics=exp.score(f'G_{mode}_check{step}',outputs)
        candidates.append(dict(name=str(checkpoint),split='validation',update=step,**exp.average(metrics,k=2)))
        write_json_atomic(folder/'candidate_scores.json',candidates);save(resume,step,True)
    finish_evaluation(start)
    began=time.perf_counter()
    for step in range(start+1,exp.cfg['flux_updates']+1):
        exp.check_time();rng=random.Random(exp.cfg['seed']+step)
        batch=[rows[rng.randrange(len(rows))] for _ in range(exp.cfg['flux_batch'])]
        rounds=torch.tensor([2+i%3 for i in range(len(batch))],device='cuda')
        current=torch.stack([exp.latent(model,r['_teacher_paths'][str(2+i%3)]) for i,r in enumerate(batch)]).cuda()
        original=torch.stack([exp.latent(model,r['lq_path']) for r in batch]).cuda()
        hr=torch.stack([exp.latent(model,r['hq_path']) for r in batch]).cuda()
        target_images=torch.stack([image(r['hq_path']) for r in batch]).cuda()
        identity=torch.tensor([rng.random()<exp.cfg['identity_fraction'] for _ in batch],device='cuda')
        current[identity]=hr[identity]
        sigma=torch.rand(len(batch),device='cuda',dtype=torch.bfloat16)
        noise=torch.randn_like(hr);zt,velocity=build_flow_matching_inputs(hr,noise,sigma)
        condition,mask,confidence=state_router_condition_tensors([r['_router_condition'] for r in batch],exp.runtime(),'cuda',torch.bfloat16)
        optimizer.zero_grad(set_to_none=True);micro=exp.cfg['flux_micro_batch'];total=0.;torch.cuda.reset_peak_memory_stats()
        for offset in range(0,len(batch),micro):
            stop=min(offset+micro,len(batch));fraction=(stop-offset)/len(batch)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(z_t=zt[offset:stop],timestep=sigma[offset:stop],
                    prompt_embeds=embeds['prompt_embeds'].expand(stop-offset,-1,-1),
                    pooled_prompt_embeds=embeds['pooled_prompt_embeds'].expand(stop-offset,-1),
                    text_ids=embeds['text_ids'].expand(stop-offset,-1,-1),z_lr=current[offset:stop],
                    z_anchor_lr=original[offset:stop],refinement_round=rounds[offset:stop],
                    lr_cond_mode='flux2_image_concat',router_condition=condition[offset:stop],
                    router_condition_mask=mask[offset:stop],router_condition_confidence=confidence[offset:stop])
                restored=zt[offset:stop].float()-sigma[offset:stop].float().view(-1,1,1,1)*prediction.float()
                loss=F.mse_loss(prediction.float(),velocity[offset:stop].float())+.1*F.smooth_l1_loss(restored,hr[offset:stop].float())
                loss=loss+exp.spatial_losses(model,restored,target_images[offset:stop])['loss_image']
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite FLUX branch loss')
            dispatch=model._last_moe_router_output['alpha_used']
            with hold_moe_routing(model.transformer,dispatch):(loss*fraction).backward()
            total+=float(loss.detach())*fraction
        gradient=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite FLUX branch gradient')
        optimizer.step()
        if step%10==0 or step==1:
            record=dict(update=step,loss=total,grad_norm=float(gradient),batch=len(batch),micro_batch=micro,
                peak_gib=torch.cuda.max_memory_allocated()/2**30)
            if hook and hook.last_delta is not None:record['image_injection_rms']=float(hook.last_delta.detach().float().square().mean().sqrt())
            history.append(record);write_json_atomic(folder/'loss_history.json',history)
            exp.log(f'FLUX {mode}: update={step}/{exp.cfg["flux_updates"]} loss={total:.5f} peak={record["peak_gib"]:.1f}GiB')
        if step%20==0 or step==exp.cfg['flux_updates']:save(resume,step,True)
        finish_evaluation(step)
    selection_rows=exp.rows('validation')[:exp.cfg['flux_eval_sources']]
    base=exp.average(exp.score_f0(selection_rows),k=1);choice=select_checkpoint(base,candidates)
    write_json_atomic(folder/'selection.json',choice);load_branch(exp,model,branch,choice['selected'],mode)
    exp.active_checkpoint=Path(choice['selected']);exp.active_mode=mode
    # Full validation/test, not the smaller calibration subset, establish final quality.
    arm=f'G_{mode}';outputs=exp.generate(model,arm,exp.rows('validation')+exp.rows('test'),embeds,max_round=4)
    ablation_outputs={};ablation_rows=exp.rows('validation')[:exp.cfg['flux_ablation_sources']]
    for ablation in ('no_lr','shuffle_lr'):
        exp.active_mode=f'{mode}_{ablation}';name=f'G_{mode}_{ablation}'
        if ablation=='no_lr':
            if hook:hook.enabled=False
            else:
                previous_gate=branch.anchor_gate.detach().clone()
                with torch.no_grad():branch.anchor_gate.zero_()
            selected=ablation_rows
        else:
            selected=[{**r,'lq_path':ablation_rows[(i+1)%len(ablation_rows)]['lq_path']} for i,r in enumerate(ablation_rows)]
        ablation_outputs[name]=exp.generate(model,name,selected,embeds,max_round=2)
        if ablation=='no_lr':
            if hook:hook.enabled=True
            else:
                with torch.no_grad():branch.anchor_gate.copy_(previous_gate)
    if hook:hook.close()
    del model,branch,hook;torch.cuda.empty_cache()
    metrics=exp.score(arm,outputs)
    for name,value in ablation_outputs.items():exp.score(name,value)
    full_base=exp.average(exp.score_f0(exp.rows('validation')),k=1)
    recursive_check=select_checkpoint(full_base,[dict(name=arm,split='validation',
        **exp.average(metrics,k=2),round4=exp.average(metrics,k=4))])
    write_json_atomic(result_path,dict(mode=mode,checkpoint=choice['selected'],quality_passed=choice['quality_passed'],
        selection_gate_scope='round2 on validation calibration subset',recursive_quality_passed=recursive_check['quality_passed'],
        trainable_parameters=sum(p.numel() for p in parameters),updates=exp.cfg['flux_updates'],
        sample_exposures=exp.cfg['flux_updates']*exp.cfg['flux_batch'],seconds=time.perf_counter()-began,
        frozen_original_moe=True,raw_full_generation=True,matched_source_update_objective=True,
        validation_round2=exp.average(metrics,k=2),validation_round4=exp.average(metrics,k=4)))
