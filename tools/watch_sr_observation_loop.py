"""Durable, own-process supervision with memory admission and checkpoint retries."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import yaml
from rl_sr.artifact_store import write_json_atomic
from rl_sr.schema import content_sha256

ROOT=Path(__file__).resolve().parents[1]


def backoff(cfg,stage):
    changed=False
    if stage=='correctors':
        keys=('loop_micro_batch','loop_eval_batch')
    elif stage=='flux_generic':keys=('inference_batch',)
    elif stage.startswith('flux_'):
        keys=('flux_micro_batch','inference_batch')
    elif stage=='f0':keys=('inference_batch',)
    else:keys=()
    for key in keys:
        if cfg[key]>1:cfg[key]=max(1,cfg[key]//2);changed=True
    return changed


def alive(pid,needle=None):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split()
        command=Path(f'/proc/{pid}/cmdline').read_bytes()
        return stat[2]!='Z' and (needle is None or needle.encode() in command)
    except (FileNotFoundError,ProcessLookupError):return False


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        command=Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        return dict(pid=pid,start_ticks=int(fields[19]),command=[v.decode() for v in command])
    except (FileNotFoundError,ProcessLookupError):return None


def same_child(record):
    actual=process_identity(record['pid']) if record else None
    return actual is not None and all(actual[k]==record[k] for k in ('pid','start_ticks','command'))


def gpu_free():
    result=subprocess.run(['/home/xd/.local/bin/nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
        capture_output=True,text=True,timeout=15)
    try:return float(result.stdout.strip().splitlines()[0])/1024
    except (ValueError,IndexError):return 0.


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    lock=(run/'supervisor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This experiment already has an active supervisor')
    cfg=yaml.safe_load(Path(args.config).read_text());original_config_sha=content_sha256(args.config)
    files=['tools/run_sr_observation_loop.py','tools/sr_observation_loop_training.py','tools/sr_observation_loop_flux.py',
        'tools/watch_sr_observation_loop.py','models/sr_observation_loop.py','rl_sr/observation_loop_data.py',
        'models/flux2_klein_sr_artist.py','models/lora_moe.py','rg_flux_fm.py','models/sr_refinement_condition.py',
        'models/text_embedding_cache.py','tools/run_sr_round_state_quick.py','models/sr_round_state_adapter.py',
        'models/prompt_builder.py','models/router_condition.py','rl_sr/conditioning.py',
        'tools/run_sr_self_refine_experiment.py','rl_sr/stable_reuse.py']
    pin=dict(config_sha256=original_config_sha,source_sha256={f:content_sha256(ROOT/f) for f in files},
        external_sha256={cfg['base_config']:content_sha256(cfg['base_config']),cfg['source_jsonl']:content_sha256(cfg['source_jsonl'])})
    signature=run/'launch_provenance.json'
    if signature.exists() and json.loads(signature.read_text())!=pin:raise SystemExit('Code/config/source pins changed; use a new run')
    write_json_atomic(signature,pin);(run/'supervisor.pid').write_text(str(os.getpid())+'\n')
    logs=run/'logs';logs.mkdir(exist_ok=True);done=run/'completed';done.mkdir(exist_ok=True)
    runtime=run/'runtime_config.yaml'
    if runtime.exists():
        previous=yaml.safe_load(runtime.read_text())
        for k in ('loop_micro_batch','loop_eval_batch','flux_micro_batch','inference_batch'):cfg[k]=min(cfg[k],previous[k])
    runtime.write_text(yaml.safe_dump(cfg,sort_keys=False))
    history=json.loads((run/'status.json').read_text()).get('history',[]) if (run/'status.json').exists() else []
    started=time.time();failure=None
    env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
        'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1'}
    def status(state,**details):
        write_json_atomic(run/'status.json',dict(status=state,time=datetime.datetime.now().astimezone().isoformat(),
            supervisor_pid=os.getpid(),elapsed_seconds=time.time()-started,history=history,**details))
    status('starting')
    child_record=run/'active_child.json'
    if child_record.exists():
        old=json.loads(child_record.read_text())
        if 'tools.run_sr_observation_loop' not in old['command'] or str(run) not in old['command']:
            raise ValueError('Stored child does not belong to this experiment')
        while same_child(old):
            status('recovering_live_child',stage=old['stage'],child_pid=old['pid'])
            if (run/'user_stop.json').exists():
                os.killpg(old['pid'],signal.SIGTERM)
                status('user_stopped');return
            time.sleep(10)
        # Its exit status is unavailable after reparenting. Safely rerun the resumable stage.
        history.append(dict(stage=old['stage'],recovered_child=old['pid'],exit_code='unknown'))
        child_record.unlink()
    # A one-off prompt preparation was started while implementation was finishing.
    preparation=run/'preparation.json'
    if preparation.exists():
        previous=json.loads(preparation.read_text())
        while alive(previous['pid'],'tools.run_sr_observation_loop'):
            status('waiting_for_preparation',preparation_pid=previous['pid'],stage=previous.get('stage'))
            if (run/'user_stop.json').exists():status('user_stopped');return
            time.sleep(10)
    for stage in cfg['stages']:
        if stage=='report':continue
        if (done/f'{stage}.json').exists():continue
        if (run/'user_stop.json').exists():failure='user_stopped';break
        if any(content_sha256(ROOT/f)!=sha for f,sha in pin['source_sha256'].items()):failure='source_changed';break
        for attempt in range(1,cfg['max_attempts']+1):
            required=cfg['minimum_free_gib'].get(stage,0)
            while required:
                try:free=gpu_free()
                except (OSError,subprocess.TimeoutExpired):free=0.
                if free>=required:break
                status('waiting_for_memory',stage=stage,free_gib=free,required_gib=required)
                if (run/'user_stop.json').exists():failure='user_stopped';break
                time.sleep(cfg['poll_seconds'])
            if failure:break
            began=time.time();command=[sys.executable,'-u','-m','tools.run_sr_observation_loop',
                '--config',str(runtime),'--run',str(run),'--stage',stage]
            with (logs/f'{stage}.log').open('ab',buffering=0) as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
                identity=process_identity(child.pid)
                if identity is not None:write_json_atomic(child_record,{**identity,'stage':stage})
                status('running',stage=stage,attempt=attempt,child_pid=child.pid)
                while child.poll() is None:
                    status('running',stage=stage,attempt=attempt,child_pid=child.pid)
                    if (run/'user_stop.json').exists():
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        failure='user_stopped';break
                    time.sleep(10)
            if child_record.exists():child_record.unlink()
            entry=dict(stage=stage,attempt=attempt,exit_code=child.returncode,seconds=time.time()-began,
                loop_micro_batch=cfg['loop_micro_batch'],flux_micro_batch=cfg['flux_micro_batch'],inference_batch=cfg['inference_batch'])
            history.append(entry)
            # Refresh a partial report after every stage, even if a later stage fails.
            with (logs/'report.log').open('ab') as log:
                try:subprocess.run([sys.executable,'-m','tools.run_sr_observation_loop','--config',str(runtime),
                    '--run',str(run),'--stage','report'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                except subprocess.TimeoutExpired:pass
            if child.returncode==0:write_json_atomic(done/f'{stage}.json',entry);break
            if failure:break
            tail=(logs/f'{stage}.log').read_text(errors='replace')[-12000:]
            if attempt==cfg['max_attempts']:failure=f'{stage}_retry_exhausted';break
            if 'out of memory' in tail.lower():
                if not backoff(cfg,stage):failure=f'{stage}_oom_at_minimum_batch';break
                runtime.write_text(yaml.safe_dump(cfg,sort_keys=False));status('retrying_smaller_microbatch',stage=stage)
            else:status('retrying',stage=stage,error_tail=tail[-2000:])
            time.sleep(5)
        if failure:break
    if failure is None:
        with (logs/'report.log').open('ab') as log:
            result=subprocess.run([sys.executable,'-m','tools.run_sr_observation_loop','--config',str(runtime),
                '--run',str(run),'--stage','report'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:failure='final_report_failed'
    status(failure or 'completed')


if __name__=='__main__':main()
