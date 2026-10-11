"""Real 36k training plumbing probe; no convergence or quality claim."""
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import os
import sys
import time

import numpy as np
from PIL import Image
import torch
import yaml
from diffusers import Flux2KleinPipeline

ROOT = Path('/media/xd/data/code/RGVOSR')
OUT = ROOT / 'artifacts/rl_sr_fixes_20261005/gpu'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from models.rg_flux_artist_factory import build_rg_flux_artist
from models.prompt_builder import build_sr_prompt
from models.router_condition import extract_router_condition
from models.text_embedding_cache import TextEmbeddingCache, stack_embedding_states
from rg_flux_fm import sample_multistep_fm
from rl_sr.artifact_store import write_jsonl_atomic, read_jsonl
from rl_sr.schema import StateRecord, RolloutRecord, RuntimePaths, content_sha256, adapter_content_sha256
from rl_sr.snapshots import is_rl_trainable_name
import train_rg_flux_refiner_sft as stage_c
import train_rg_flux_output_rl as stage_e

torch.set_num_threads(8)
config = yaml.safe_load((ROOT / 'configs/rl_sr_refinement_flux2_klein_moe.yaml').read_text())
config['text_encoding']['mode'] = 'cached'
config['text_encoding']['cache_dir'] = str(OUT / 'text_cache')
config['text_encoding']['mode'] = os.environ.get('RL_SR_VERIFY_TEXT_MODE', 'cached')
config['rl_sr']['num_workers'] = 0
config['rl_sr']['log_every'] = 1
config['rl_sr']['sft_output_dir'] = str(OUT / 'stage_c')
config['rl_sr']['rl_output_dir'] = str(OUT / 'stage_e')
config['rl_sr']['init_adapter'] = str(ROOT / config['rl_sr']['init_adapter'])
(OUT / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
rows = [json.loads(line) for line in (ROOT / 'artifacts/gain_validation_20261005/manifest_4096.jsonl').open()
        if json.loads(line)['split'] == 'val'][:2]
prompts = [build_sr_prompt(row['unipercept_raw']['profile'], use_prompt=True, use_suggestions=True,
                          prompt_variant='iqa_suggestion', include_caption=True) for row in rows]
cache = TextEmbeddingCache(config['text_encoding']['cache_dir'], config, dtype=torch.bfloat16)
if cache.records_by_prompt:
    # Repair only this diagnostic's initial pooled=None payload; Klein's actual
    # encode_prompts/cache contract uses a [B,0] tensor.
    for record in cache.records_by_prompt.values():
        path = cache.embedding_path(record)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['pooled_prompt_embeds'] is None:
            payload['pooled_prompt_embeds'] = torch.zeros(1, 0, dtype=torch.bfloat16)
            torch.save(payload, path)
    prompt_embeds, _, text_ids = stack_embedding_states([
        cache.load_embedding(prompt, device='cpu', dtype=torch.bfloat16) for prompt in prompts
    ], device='cuda', dtype=torch.bfloat16)
    print('Reusing real FP32-Qwen prompt cache', flush=True)
else:
    print('Encoding two real prompts with FP32 Qwen3', flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(config['model']['flux_model_path'], transformer=None, vae=None,
                                            torch_dtype=torch.float32, local_files_only=True).to('cuda')
    with torch.inference_mode():
        prompt_embeds, text_ids = pipe.encode_prompt(prompts, device='cuda', max_sequence_length=512)
    prompt_embeds = prompt_embeds.to(torch.bfloat16)
    text_ids = text_ids.to(torch.bfloat16)
    for index, row in enumerate(rows):
        cache.save_embedding(prompts[index], row['sample_id'], row['lq_path'], row['hq_path'], {
            'prompt_embeds': prompt_embeds[index:index+1], 'text_ids': text_ids[index:index+1],
            'pooled_prompt_embeds': prompt_embeds.new_zeros(1, 0)})
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    

def read_rgb(path):
    with Image.open(path) as image:
        array = np.array(image.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1).cuda() * 2 - 1

original = torch.stack([read_rgb(row['lq_path']) for row in rows])
conditions = [extract_router_condition(row['unipercept_raw']['profile']) for row in rows]
router = dict(router_condition=torch.tensor([c.values for c in conditions], device='cuda', dtype=torch.bfloat16),
              router_condition_mask=torch.tensor([c.valid_mask for c in conditions], device='cuda', dtype=torch.bfloat16),
              router_condition_confidence=torch.tensor([c.confidence for c in conditions], device='cuda', dtype=torch.bfloat16))
state_path = OUT / 'states.jsonl'
if state_path.exists():
    states = list(read_jsonl(state_path, StateRecord))
    assert len(states) == 6 and {state.round_index for state in states} == {2,3,4}
    print('Reusing six actual 25-NFE F0 states', flush=True)
else:
    print('Generating actual F0 rounds 1/2/3 with 25 NFE each', flush=True)
    artist = build_rg_flux_artist(config).to('cuda')
    artist.load_trainable(config['rl_sr']['init_adapter'], is_trainable=False)
    artist.align_inference_dtype(torch.bfloat16)
    artist.requires_grad_(False).eval()
    artist.set_moe_inference_schedule()
    states, parents = [], [None, None]
    current = original
    with torch.inference_mode():
        for round_index in range(1, 4):
            torch.manual_seed(42+round_index)
            z_current = artist.encode_images(current, sample=False).to(torch.bfloat16)
            z0 = sample_multistep_fm(artist=artist, shape=z_current.shape, prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=prompt_embeds.new_zeros(2, 0), text_ids=text_ids, z_lr=z_current,
                num_steps=25, device='cuda', dtype=torch.bfloat16, lr_cond_mode='flux2_image_concat',
                schedule='linear', **router)
            current = artist.decode_latents(z0).float().clamp(-1, 1)
            for index, row in enumerate(rows):
                image_path = OUT / f"f0_{row['sample_id']}_r{round_index}.png"
                array = ((current[index].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
                Image.fromarray(array).save(image_path)
                record = StateRecord(dataset_id='36k_branch_fix_smoke', sample_key=row['sample_id'],
                    original_lr_key=f"lr/{row['sample_id']}.png", original_lr_sha256=content_sha256(row['lq_path']),
                    current_output_key=image_path.name, current_output_sha256=content_sha256(image_path),
                    round_index=round_index+1, parent_state_id=parents[index], geometry_sha256='512x512_rgb',
                    producer_adapter_sha256=adapter_content_sha256(config['rl_sr']['init_adapter']), sampler_sha256='linear25',
                    hr_key=f"hr/{row['sample_id']}.png", hr_sha256=content_sha256(row['hq_path']), prompt=prompts[index],
                    metadata={'router_condition': conditions[index].as_dict()},
                    runtime=RuntimePaths(original_lr=row['lq_path'], current_sr=str(image_path), hr=row['hq_path']))
                states.append(record)
                parents[index] = record.state_id
            # Continue from the saved RGB, matching the branch's state cache.
            current = torch.stack([read_rgb(OUT / f"f0_{row['sample_id']}_r{round_index}.png") for row in rows])
            print(f'F0 round {round_index} complete', flush=True)
    del artist, z0, z_current, current
    gc.collect()
    torch.cuda.empty_cache()
    state_path = OUT / 'states.jsonl'
    write_jsonl_atomic(state_path, [state.as_dict() for state in states])
    

captured = {}
def capture_factory(cfg):
    model = build_rg_flux_artist(cfg)
    original_load = model.load_trainable
    def load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        captured['versions'] = {name: parameter._version for name, parameter in model.named_parameters()
            if not is_rl_trainable_name(name) or name.startswith('moe_router.')}
        return result
    model.load_trainable = load
    captured['artist'] = model
    return model

def frozen_check():
    model = captured['artist']
    parameters = dict(model.named_parameters())
    for name, version in captured['versions'].items():
        assert parameters[name]._version == version, name
        assert parameters[name].grad is None and not parameters[name].requires_grad, name
    assert model.moe_routing_mode == 'topk' and model.moe_temperature == .7
    assert all(torch.isfinite(parameter).all() for name, parameter in parameters.items()
               if is_rl_trainable_name(name))
    return len(captured['versions'])

stage_c.build_rg_flux_artist = capture_factory
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
print('Running real Stage C: 2 updates, batch4, spatial64, Charbonnier+LPIPS', flush=True)
summary_c = stage_c.train(config, str(state_path), max_steps=2)
report = {'scope': 'Two-source plumbing/gradient probe; not convergence or quality evidence',
          'text_encoding_mode': config['text_encoding']['mode'],
          'stage_c': {'summary': summary_c, 'seconds': time.perf_counter()-start,
                      'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                      'frozen_parameters_checked': frozen_check()}}
model = captured['artist']
assert model.refinement_condition_adapter.anchor_gate.item() != 0
report['stage_c']['anchor_gate'] = model.refinement_condition_adapter.anchor_gate.item()
(OUT / 'verification_partial.json').write_text(json.dumps(report, indent=2))
print('C verified', json.dumps({key: value for key, value in report['stage_c'].items() if key != 'summary'}), flush=True)

print('Collecting actual post-C actions for NFT; 1 NFE diagnostic only', flush=True)
policy_dir = Path(config['rl_sr']['sft_output_dir']) / 'rg_flux_adapters'
policy_hash = adapter_content_sha256(policy_dir)
rollouts = []
model.eval()
with torch.no_grad():
    for index, state in enumerate(states):
        torch.manual_seed(100+index)
        row_index = index % 2
        z_current = model.encode_images(read_rgb(state.runtime.current_sr).unsqueeze(0), sample=False).to(torch.bfloat16)
        z_anchor = model.encode_images(original[row_index:row_index+1], sample=False).to(torch.bfloat16)
        context = {key: value[row_index:row_index+1] for key, value in router.items()}
        z0 = sample_multistep_fm(model, z_current.shape, prompt_embeds[row_index:row_index+1],
            prompt_embeds.new_zeros(1, 0), text_ids=text_ids[row_index:row_index+1], z_lr=z_current,
            z_anchor_lr=z_anchor, refinement_round=state.round_index, num_steps=1, device='cuda',
            dtype=torch.bfloat16, lr_cond_mode='flux2_image_concat', schedule='linear',
            **context)
        path = OUT / f'action_{index}.pt'
        torch.save(z0[0].cpu(), path)
        # Controlled scores exercise E precision, not an IQA/quality experiment.
        rollouts.append(RolloutRecord(state_id=state.state_id, policy_adapter_sha256=policy_hash,
            candidate_index=0, sample_latent_sha256=content_sha256(path), raw_sr_sha256='diagnostic',
            executed_sr_sha256='diagnostic', sampler_sha256='linear1_diagnostic',
            reward_version='controlled_precision_probe', total_reward=0., optimality_probability=.4995,
            confidence=1., runtime=RuntimePaths(sample_latent=str(path))))
rollout_path = OUT / 'scored_rollouts.jsonl'
write_jsonl_atomic(rollout_path, [row.as_dict() for row in rollouts])
captured.clear()
del model, z0, z_current, z_anchor
gc.collect()
torch.cuda.empty_cache()
print('Before E CUDA memory', torch.cuda.mem_get_info(), flush=True)

stage_e.build_rg_flux_artist = capture_factory
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
print('Running real Stage E: 1 update, batch4 + replay batch4', flush=True)
args = SimpleNamespace(output_dir=config['rl_sr']['rl_output_dir'], state_jsonl=str(state_path),
    scored_rollout_jsonl=str(rollout_path), old_adapter=str(policy_dir), reference_adapter=str(policy_dir),
    policy_adapter=str(policy_dir), sft_state_jsonl=str(state_path), max_steps=1)
summary_e = stage_e.train(config, args)
report['stage_e'] = {'summary': summary_e, 'seconds': time.perf_counter()-start,
                    'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                    'frozen_parameters_checked': frozen_check()}
(OUT / 'verification.json').write_text(json.dumps(report, indent=2))
print('GPU_VERIFY_PASS', json.dumps({key: {k:v for k,v in value.items() if k != 'summary'}
      for key,value in report.items() if isinstance(value,dict)}), flush=True)
