#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
exec /home/xd/miniconda3/envs/sr-flux2/bin/python - "$@" <<'PY'
import datetime,json,os,subprocess,sys
from pathlib import Path
from tools.watch_sr_observation_loop import alive
root=Path.cwd();config=root/'configs/rl_sr_observation_loop.yaml'
run=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else root/'exp_rg_flux_self_refine/observation_loop_v5_20261010'
run.mkdir(parents=True,exist_ok=True);pidfile=run/'supervisor.pid'
if pidfile.exists() and alive(int(pidfile.read_text().strip()),'tools.watch_sr_observation_loop'):
    print(f'Already running: {run}/status.json');raise SystemExit(0)
command=[sys.executable,'-u','-m','tools.watch_sr_observation_loop','--config',str(config),'--run',str(run)]
with (run/'supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(run/'launch.json').write_text(json.dumps(dict(pid=child.pid,command=command,
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    time=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
print(f'Started supervisor PID {child.pid}\nRun: {run}\nStatus: {run}/status.json')
PY
