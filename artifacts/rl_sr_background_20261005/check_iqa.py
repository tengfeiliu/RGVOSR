"""Warm and verify the configured IQA models on CPU before the long job needs them."""
from pathlib import Path
import csv,json,math,sys,time
import torch,yaml,pyiqa
from urllib.error import URLError
import pyiqa.utils.download_util as download_util

# Warm the exact production cache filenames. Retry transient TLS failures,
# then use the official GitHub release source provided by this PyIQA version.
original_download = download_util.download_url_to_file
def reliable_download(url, destination, *args, **kwargs):
    alternatives=[url,url]
    if 'huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/' in url:
        alternatives.append(url.replace(
            'https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/',
            'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/'))
    for attempt, source in enumerate(alternatives):
        try:
            return original_download(source,destination,*args,**kwargs)
        except URLError:
            if attempt == len(alternatives)-1: raise
            print('RETRY_WEIGHT',source,flush=True)
            time.sleep(1)
download_util.download_url_to_file = reliable_download
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from metrics.rg_sr_metrics import metric_input,score_to_float
metadata=json.loads((ROOT/'artifacts/rl_sr_background_20261005/launch_metadata.json').read_text())
config=yaml.safe_load(Path(metadata['config']).read_text())
path=Path(metadata['run_dir'])/'00_inputs/train_lq_inputs.txt'
image=Path(path.read_text().splitlines()[0])
from PIL import Image
with Image.open(image) as value: width,height=value.size
row={'path':str(image),'width':width,'height':height}
torch.set_num_threads(8)
results=[]
for name in config['rl_sr']['evaluation']['metrics']:
    started=time.perf_counter()
    print('VERIFY',name,flush=True)
    try:
        model=pyiqa.create_metric(name,device='cpu').eval()
        with torch.no_grad(): score=score_to_float(model(metric_input(row,name,torch)))
        if not math.isfinite(score): raise RuntimeError('Nonfinite metric result')
        results.append({'metric':name,'status':'passed','score':score,'seconds':time.perf_counter()-started,'error':''})
        print('PASS',name,score,flush=True)
        del model
    except Exception as error:
        results.append({'metric':name,'status':'failed','score':'','seconds':time.perf_counter()-started,'error':repr(error)})
        print('FAIL',name,repr(error),flush=True)
    with (ROOT/'artifacts/rl_sr_background_20261005/iqa_check.csv').open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=['metric','status','score','seconds','error'])
        writer.writeheader();writer.writerows(results)
if any(row['status']=='failed' for row in results): raise SystemExit(1)
print('IQA_CHECK_PASS',flush=True)
