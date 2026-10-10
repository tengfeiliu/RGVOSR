"""Actual-output supervision, shared unroll and cheap old-policy refresh."""
from pathlib import Path
import json
import hashlib
import random
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from models.sr_observation_loop import ObservationLoop
from models.sr_round_state_adapter import gradient_consistency
from rl_sr.artifact_store import write_json_atomic
from rl_sr.observation_loop_data import select_checkpoint
from rl_sr.schema import content_sha256,identity_sha256
from tools.run_sr_round_state_quick import atomic_torch

ARMS={
    'C0_weak':dict(lr_mode='weak',use_round=False,use_state=False),
    'C1_spatial':dict(lr_mode='spatial',use_round=False,use_state=False),
    'C2_round_state':dict(lr_mode='spatial',use_round=True,use_state=False),
    'C3_recursive':dict(lr_mode='spatial',use_round=True,use_state=True),
    'C4_refresh':dict(lr_mode='spatial',use_round=True,use_state=True),
}


def expected_refresh_update(update,total,interval):
    return (min(update,total-1)//interval)*interval


def pending_work(update,total,checkpoint_every,refresh_every,candidates,buffer_update):
    work=[]
    if update and (update%checkpoint_every==0 or update==total) and not any(c['update']==update for c in candidates):work.append('evaluate')
    expected=expected_refresh_update(update,total,refresh_every)
    if expected and buffer_update!=expected:work.append('refresh')
    return work


def single_step_rounds(count,update):
    return [2+(i+update)%3 for i in range(count)]


def tensor_sha(value):
    value=value.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest()


def refresh_payload(images,hidden,rounds,update,producer_sha,split_sha,source_keys):
    tensors=dict(states=images.cpu().float(),hidden=hidden.cpu().bfloat16(),rounds=rounds.cpu().long())
    return dict(**tensors,update=update,producer_sha256=producer_sha,split_sha256=split_sha,source_keys=source_keys,
        tensor_sha256={k:tensor_sha(v) for k,v in tensors.items()},protocol='continuous RGB float32 and corresponding old hidden BF16')


def validate_refresh(buf,update,producer_sha,split_sha,source_keys,width,image_hw=512):
    if any(buf.get(k)!=v for k,v in dict(update=update,producer_sha256=producer_sha,
        split_sha256=split_sha,source_keys=source_keys).items()):raise ValueError('Old buffer producer/source identity mismatch')
    n=len(source_keys)
    shapes=dict(states=(n,3,image_hw,image_hw),hidden=(n,width,image_hw//16,image_hw//16),rounds=(n,))
    for key,shape in shapes.items():
        value=buf[key]
        if tuple(value.shape)!=shape or not torch.isfinite(value).all() or tensor_sha(value)!=buf['tensor_sha256'][key]:
            raise ValueError('Old buffer tensor integrity/geometry mismatch')
    if buf['states'].dtype!=torch.float32 or buf['hidden'].dtype!=torch.bfloat16 or buf['rounds'].dtype!=torch.long:
        raise ValueError('Old buffer tensor precision mismatch')
    if not torch.equal(buf['rounds'],torch.tensor([2+i%2 for i in range(n)])):raise ValueError('Old buffer round labels mismatch')


def read_chain(folder,key,signature):
    path=folder/f'{key}.chain.json'
    if not path.exists():return None
    meta=json.loads(path.read_text())
    if meta['signature']!=signature:raise ValueError('Completed image chain checkpoint/input identity differs')
    for row in meta['outputs']:
        if not Path(row['path']).exists() or content_sha256(row['path'])!=row['output_sha256']:
            raise ValueError('Completed image chain content changed')
    return meta['outputs']


def commit_chain(folder,key,predictions,signature):
    from tools.run_sr_round_state_quick import save_image
    existing=read_chain(folder,key,signature)
    if existing is not None:return existing
    # A final manifest commits the chain. Interrupted, uncommitted PNGs can be replaced.
    outputs=[]
    for k,value in enumerate(predictions,2):
        path=folder/f'{key}_r{k}.png';temporary=folder/f'.{key}_r{k}.pending.png'
        save_image(value,temporary);temporary.replace(path)
        outputs.append(dict(round=k,path=str(path),output_sha256=content_sha256(path)))
    write_json_atomic(folder/f'{key}.chain.json',dict(signature=signature,outputs=outputs))
    return outputs


def rollout_tensor(model,current,original_lr,loops,options,start_rounds=None,maximum_round=None,initial_state=None):
    result=[];state=initial_state
    starts=torch.full((len(current),),2,device=current.device,dtype=torch.long) if start_rounds is None else start_rounds
    for offset in range(loops):
        rounds=starts+offset
        active=torch.ones_like(rounds,dtype=torch.bool) if maximum_round is None else rounds<=maximum_round
        passed_rounds=rounds if maximum_round is None else rounds.clamp(max=maximum_round)
        out=model(current,original_lr,passed_rounds,state=state,**options)
        current=torch.where(active[:,None,None,None],out['image'].clamp(-1,1),current)
        state=out['state'] if state is None else torch.where(active[:,None,None,None],out['state'],state)
        result.append(current)
    return result


def unroll_loss(model,current,original_lr,target,loops,options,gradient_weight,low_frequency_weight,start_rounds=None,initial_state=None):
    outputs=rollout_tensor(model,current,original_lr,loops,options,start_rounds,maximum_round=4,initial_state=initial_state);losses=[]
    for offset,value in enumerate(outputs):
        local_target=target
        if start_rounds is not None:
            active=start_rounds+offset<=4
            if not active.any():continue
            value=value[active];local_target=target[active]
        value=value.float();local_target=local_target.float()
        charb=((value-local_target).square()+1e-6).sqrt().mean()
        gradient=gradient_consistency(value,local_target)
        low=F.l1_loss(F.avg_pool2d(value,4),F.avg_pool2d(local_target,4))
        losses.append(charb+gradient_weight*gradient+low_frequency_weight*low)
    return torch.stack(losses).mean(),outputs


class ImageBank:
    """Read uint8 once. Float tensors are produced only for sampled batches."""
    def __init__(self,rows,device='cuda'):
        self.rows=rows;self.data={}
        for field in ('_f0_path','lq_path','hq_path'):
            tensors=[]
            for r in rows:
                with Image.open(r[field]) as im:
                    tensors.append(torch.from_numpy(np.array(im.convert('RGB'),dtype=np.uint8)).permute(2,0,1))
            self.data[field]=torch.stack(tensors).to(device)
        for k in (3,4):
            tensors=[]
            for r in rows:
                with Image.open(r['_teacher_paths'][str(k)]) as im:
                    tensors.append(torch.from_numpy(np.array(im.convert('RGB'),dtype=np.uint8)).permute(2,0,1))
            self.data[f'teacher{k}']=torch.stack(tensors).to(device)

    def sample(self,indices,crop,rng,rounds=None):
        index=torch.tensor(indices,device=self.data['_f0_path'].device)
        values=[self.data[k][index].float()/127.5-1 for k in ('_f0_path','lq_path','hq_path')]
        if rounds is not None:
            for k in (3,4):
                chosen=rounds==k
                values[0][chosen]=self.data[f'teacher{k}'][index[chosen]].float()/127.5-1
        if crop!=512:
            out=[[] for _ in values]
            for i in range(len(indices)):
                y=rng.randrange(513-crop);x=rng.randrange(513-crop)
                for bucket,v in zip(out,values):bucket.append(v[i,:,y:y+crop,x:x+crop])
            values=[torch.stack(v) for v in out]
        return values


def model_for(exp):
    return ObservationLoop(width=exp.cfg['loop_width'],depth=exp.cfg['loop_depth'],heads=exp.cfg['loop_heads']).cuda()


def load_correction(exp,checkpoint):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload['f0_sha256']!=exp.cfg['f0_sha256'] or payload['split_sha256']!=exp.split_sha():
        raise ValueError('Correction checkpoint F0/data identity mismatch')
    model=model_for(exp);model.load_state_dict(payload['model']);model.eval()
    return model,payload


@torch.no_grad()
def evaluate_correction(exp,model,arm,options,rows,max_round=4,ablation=None):
    from tools.run_sr_round_state_quick import image
    folder=exp.run/'trajectories'/arm;folder.mkdir(parents=True,exist_ok=True)
    outputs=[];batch_size=exp.cfg['loop_eval_batch'];model.eval();pending=[]
    started=time.perf_counter();actual_options=dict(options)
    if ablation=='no_lr':actual_options['lr_mode']='off'
    if ablation=='no_round':actual_options['use_round']=False
    if ablation=='no_state':actual_options['use_state']=False
    checkpoint_sha=getattr(exp,'active_correction_sha',None)
    if checkpoint_sha is None:raise ValueError('Inference requires a pinned correction checkpoint')
    for i,r in enumerate(rows):
        original=rows[(i+1)%len(rows)]['lq_path'] if ablation=='shuffle_lr' else r['lq_path']
        signature=dict(checkpoint_sha256=checkpoint_sha,split_sha256=exp.split_sha(),
            f0_sha256=content_sha256(r['_f0_path']),lr_sha256=content_sha256(original),
            options=actual_options,max_round=max_round,ablation=ablation,protocol='float-RGB-hidden-v1')
        prefix=dict(arm=arm,sample_key=r['_sample_key'],split=r['_split'])
        outputs.append(dict(**prefix,round=1,path=r['_f0_path']))
        prior=read_chain(folder,r['_sample_key'],signature)
        if prior is not None:outputs.extend(dict(**prefix,**out) for out in prior)
        else:pending.append((r,original,signature))
    for offset in range(0,len(pending),batch_size):
        batch=pending[offset:offset+batch_size];exp.check_time()
        current=torch.stack([image(r['_f0_path']) for r,_,_ in batch]).cuda()
        original=torch.stack([image(path) for _,path,_ in batch]).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            predictions=rollout_tensor(model,current,original,max_round-1,actual_options)
        for i,(r,_,signature) in enumerate(batch):
            committed=commit_chain(folder,r['_sample_key'],[p[i] for p in predictions],signature)
            prefix=dict(arm=arm,sample_key=r['_sample_key'],split=r['_split'])
            outputs.extend(dict(**prefix,**out) for out in committed)
    from rl_sr.artifact_store import write_jsonl_atomic
    write_jsonl_atomic(folder/'outputs.jsonl',outputs)
    write_json_atomic(folder/'inference.json',dict(options=actual_options,ablation=ablation,seconds=time.perf_counter()-started,
        sources=len(rows),generated_chains=len(pending),max_round=max_round,state_protocol='continuous floating RGB + hidden state within each chain; PNG quantization for metric/export only',
        checkpoint_sha256=checkpoint_sha,ground_truth_used_as_input=False))
    return exp.score(arm,outputs)


def train_correctors(exp):
    torch.set_num_threads(4)
    bank=ImageBank(exp.rows('train'));base_rows=exp.rows('validation')[:exp.cfg['loop_validation_sources']]
    base=exp.average(exp.score_f0(base_rows),k=1)
    all_results=[]
    for arm,options in ARMS.items():
        destination=exp.run/'correction'/arm;destination.mkdir(parents=True,exist_ok=True)
        result_file=destination/'result.json'
        if result_file.exists():all_results.append(json.loads(result_file.read_text()));continue
        torch.manual_seed(exp.cfg['seed']);torch.cuda.manual_seed_all(exp.cfg['seed'])
        model=model_for(exp);optimizer=torch.optim.AdamW(model.parameters(),lr=exp.cfg['loop_lr'],weight_decay=.01)
        start=0;resume=destination/'resume.pt';history=[];candidates=[];refresh=None;refresh_round=None
        if resume.exists():
            payload=torch.load(resume,map_location='cpu',weights_only=True)
            if payload['f0_sha256']!=exp.cfg['f0_sha256'] or payload['split_sha256']!=exp.split_sha():raise ValueError('Resume provenance differs')
            model.load_state_dict(payload['model']);optimizer.load_state_dict(payload['optimizer']);start=payload['update']
            history=payload['history'];candidates=payload['candidates']
            torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state(payload['cuda_rng'])
        refresh_path=destination/'old_buffer.pt';refresh_hidden=None
        source_keys=[r['_sample_key'] for r in bank.rows]
        buffer_update=None
        parameters=sum(p.numel() for p in model.parameters());began=time.perf_counter()
        total_calls=exp.cfg['loop_batch']*sum(min(3,1+(s-1)//exp.cfg['refresh_every']) if arm in ('C3_recursive','C4_refresh') else 1 for s in range(1,start+1))
        def save(path,step,include_optimizer=False):
            payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,options=options,
                update=step,f0_sha256=exp.cfg['f0_sha256'],split_sha256=exp.split_sha(),parameters=parameters)
            if include_optimizer:payload.update(optimizer=optimizer.state_dict(),history=history,candidates=candidates,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            atomic_torch(path,payload)
        def ensure_checkpoint(step):
            checkpoint=destination/f'checkpoint-{step:05d}.pt'
            if not checkpoint.exists():save(checkpoint,step)
            else:
                old=torch.load(checkpoint,map_location='cpu',weights_only=True)
                if old['update']!=step or old['split_sha256']!=exp.split_sha() or old['f0_sha256']!=exp.cfg['f0_sha256']:
                    raise ValueError('Existing correction checkpoint identity mismatch')
            return checkpoint
        def finish_work(step):
            nonlocal refresh,refresh_round,refresh_hidden,buffer_update
            if step and (step%exp.cfg['loop_checkpoint_every']==0 or step==exp.cfg['loop_updates']):
                checkpoint=ensure_checkpoint(step)
                if not any(c['update']==step for c in candidates):
                    exp.active_correction_sha=content_sha256(checkpoint)
                    metric=evaluate_correction(exp,model,f'{arm}_check{step}',options,base_rows)
                    candidates.append(dict(name=str(checkpoint),split='validation',update=step,
                        round4=exp.average(metric,k=4),**exp.average(metric,k=2)))
                    write_json_atomic(destination/'candidate_scores.json',candidates)
                    save(resume,step,True)
            expected=expected_refresh_update(step,exp.cfg['loop_updates'],exp.cfg['refresh_every']) if arm=='C4_refresh' else 0
            if not expected or buffer_update==expected:return
            producer=destination/f'checkpoint-{expected:05d}.pt'
            if not producer.exists():
                if expected!=step:raise ValueError('Required old policy checkpoint missing')
                ensure_checkpoint(step)
            producer_sha=content_sha256(producer)
            buf=torch.load(refresh_path,map_location='cpu',weights_only=True) if refresh_path.exists() else None
            if buf is not None and buf['update']==expected:
                validate_refresh(buf,expected,producer_sha,exp.split_sha(),source_keys,exp.cfg['loop_width'])
            else:
                # Reconstruct a missed refresh from the exact old snapshot, never from later weights.
                live={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(torch.load(producer,map_location='cpu',weights_only=True)['model']);model.eval()
                states=[];hidden=[];labels=[]
                with torch.no_grad():
                    for offset in range(0,len(bank.rows),exp.cfg['loop_eval_batch']):
                        index=list(range(offset,min(offset+exp.cfg['loop_eval_batch'],len(bank.rows))))
                        y,x,_=bank.sample(index,512,random.Random(0));state=None;ys=[];hs=[]
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            for k in (2,3):
                                out=model(y,x,k,state=state,**options);y=out['image'].clamp(-1,1);state=out['state']
                                ys.append(y);hs.append(state)
                        r=torch.tensor([2+i%2 for i in index],device='cuda')
                        states.append(torch.stack([ys[int(k)-2][j] for j,k in enumerate(r)]).cpu())
                        hidden.append(torch.stack([hs[int(k)-2][j] for j,k in enumerate(r)]).cpu())
                        labels.append(r.cpu())
                buf=refresh_payload(torch.cat(states),torch.cat(hidden),torch.cat(labels),expected,
                    producer_sha,exp.split_sha(),source_keys)
                validate_refresh(buf,expected,producer_sha,exp.split_sha(),source_keys,exp.cfg['loop_width'])
                atomic_torch(refresh_path,buf);model.load_state_dict(live)
                exp.log(f'{arm}: old snapshot refreshed at update {expected}, states={len(buf["states"])}')
            refresh=buf['states'].cuda();refresh_hidden=buf['hidden'].cuda();refresh_round=buf['rounds'].cuda()
            buffer_update=expected
            del buf
        # Recover post-update work before moving to the next optimizer update.
        finish_work(start)
        exp.log(f'{arm}: {parameters} trainable params, {len(bank.rows)} sources, batch={exp.cfg["loop_batch"]}, crop={exp.cfg["loop_crop"]}')
        for step in range(start+1,exp.cfg['loop_updates']+1):
            exp.check_time();rng=random.Random(exp.cfg['seed']+step)
            ids=[rng.randrange(len(bank.rows)) for _ in range(exp.cfg['loop_batch'])]
            rounds=torch.tensor(single_step_rounds(len(ids),step) if arm in ('C0_weak','C1_spatial','C2_round_state') else [2]*len(ids),device='cuda')
            current,lr,target=bank.sample(ids,exp.cfg['loop_crop'],rng,rounds=rounds)
            initial_state=None
            loops=1
            if arm in ('C3_recursive','C4_refresh'):
                loops=min(3,1+(step-1)//exp.cfg['refresh_every'])
            if refresh is not None and exp.cfg['loop_crop']==512:
                use=torch.tensor([rng.random()<exp.cfg['refresh_fraction'] for _ in ids],device='cuda')
                i=torch.tensor(ids,device='cuda');current[use]=refresh[i[use]]
                initial_state=torch.zeros(len(ids),exp.cfg['loop_width'],32,32,device='cuda',dtype=torch.bfloat16)
                initial_state[use]=refresh_hidden[i[use]]
                rounds[use]=refresh_round[i[use]]+1
            # Identity states are passed through the actual correction operator.
            identity=torch.tensor([rng.random()<exp.cfg['identity_fraction'] for _ in ids],device='cuda')
            current[identity]=target[identity]
            if initial_state is not None:initial_state[identity]=0
            model.train()
            optimizer.zero_grad(set_to_none=True);micro=exp.cfg['loop_micro_batch'];total=0.
            torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
            for offset in range(0,len(ids),micro):
                stop=min(offset+micro,len(ids));weight=(stop-offset)/len(ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,_=unroll_loss(model,current[offset:stop],lr[offset:stop],target[offset:stop],loops,options,
                        exp.cfg['gradient_weight'],exp.cfg['low_frequency_weight'],start_rounds=rounds[offset:stop],initial_state=None if initial_state is None else initial_state[offset:stop])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite correction loss')
                (loss*weight).backward();total+=float(loss.detach())*weight
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite correction gradient')
            optimizer.step();total_calls+=len(ids)*loops
            if step%20==0 or step==1:
                torch.cuda.synchronize();history.append(dict(update=step,loss=total,grad_norm=float(norm),batch=len(ids),loops=loops,
                    identity_count=int(identity.sum()),seconds=time.perf_counter()-t0,peak_gib=torch.cuda.max_memory_allocated()/2**30))
                write_json_atomic(destination/'loss_history.json',history)
                exp.log(f'{arm} update={step}/{exp.cfg["loop_updates"]} loops={loops} loss={total:.5f} peak={history[-1]["peak_gib"]:.1f}GiB')
            if step%100==0 or step==exp.cfg['loop_updates']:save(resume,step,True)
            finish_work(step)
            model.train()
        choice=select_checkpoint(base,candidates);write_json_atomic(destination/'selection.json',choice)
        del model;torch.cuda.empty_cache();model,payload=load_correction(exp,choice['selected'])
        exp.active_correction_sha=content_sha256(choice['selected'])
        metrics=evaluate_correction(exp,model,arm,options,exp.rows('validation')+exp.rows('test'))
        result=dict(arm=arm,checkpoint=choice['selected'],checkpoint_sha256=exp.active_correction_sha,
            quality_passed=choice['quality_passed'],options=options,parameters=parameters,
            updates=exp.cfg['loop_updates'],sample_exposures=exp.cfg['loop_updates']*exp.cfg['loop_batch'],
            correction_calls=total_calls,training_seconds=time.perf_counter()-began,
            validation_round2=exp.average(metrics,k=2),validation_round4=exp.average(metrics,k=4))
        write_json_atomic(result_file,result);all_results.append(result);del model;torch.cuda.empty_cache()
    # Choose the model on full validation only, then inspect frozen-model ablations.
    full_base=exp.average(exp.score_f0(exp.rows('validation')),k=1)
    final_choice=select_checkpoint(full_base,[dict(name=r['arm'],split='validation',round4=r['validation_round4'],
        **r['validation_round2']) for r in all_results])
    best=next(r for r in all_results if r['arm']==final_choice['selected'])
    model,payload=load_correction(exp,best['checkpoint']);exp.active_correction_sha=best['checkpoint_sha256']
    for ablation in ('no_lr','shuffle_lr','no_round','no_state'):
        evaluate_correction(exp,model,f'{best["arm"]}_{ablation}',best['options'],exp.rows('validation'),ablation=ablation)
    write_json_atomic(exp.run/'correction_results.json',dict(arms=all_results,validation_selected_arm=best['arm'],selection=final_choice))
