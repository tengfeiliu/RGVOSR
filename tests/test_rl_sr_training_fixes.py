"""Behavioral regressions for the audited RL-SR training defects."""
from types import SimpleNamespace
import json
import os
import subprocess
import sys

import pytest
import torch
from torch import nn

from losses.output_nft import nft_velocity_loss, reference_velocity_loss
from models.flux2_klein_sr_artist import Flux2KleinSRArtist
from models.sr_refinement_condition import SRRefinementConditionAdapter
from rl_sr.snapshots import configure_rl_trainables


def small_artist():
    # Avoid the external 4B load; keep the real routing and checkpoint methods.
    artist = Flux2KleinSRArtist.__new__(Flux2KleinSRArtist)
    nn.Module.__init__(artist)
    artist.config = {'model': {'lora_moe': {
        'inference_routing_mode': 'topk', 'inference_temperature': .7, 'inference_top_k': 2,
    }}}
    artist.use_lora = True
    artist.lora_backend = 'moe'
    artist.moe_router = nn.Linear(2, 2)
    artist.transformer = nn.Linear(2, 2)
    artist.lora_adapter = nn.Parameter(torch.zeros(1))
    artist.moe_routing_mode = 'soft'
    artist.moe_temperature = 2.
    artist.moe_router_noise_std = .1
    return artist


def test_frozen_router_uses_rollout_schedule():
    artist = small_artist()
    configure_rl_trainables(artist, train_router=False)
    assert not artist.moe_router.weight.requires_grad
    assert artist.moe_routing_mode == 'topk'
    assert artist.moe_temperature == .7
    assert artist.moe_top_k == 2
    assert artist.moe_router_noise_std == 0


def test_nft_keeps_weak_fp32_preference_with_bf16_velocities():
    current = torch.ones(1, 1, 2, 2, dtype=torch.bfloat16, requires_grad=True)
    old = torch.ones_like(current, requires_grad=True)
    loss = nft_velocity_loss(current, old, torch.zeros_like(current), torch.tensor([.4995]))
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.allclose(current.grad.float(), torch.full((1, 1, 2, 2), -.0005), atol=1e-5)
    assert old.grad is None


def test_reference_loss_uses_fp32_residuals_and_detached_reference():
    current = torch.full((1, 2), 1.015625, dtype=torch.bfloat16, requires_grad=True)
    reference = torch.ones_like(current, requires_grad=True)
    loss = reference_velocity_loss(current, reference)
    loss.backward()
    assert loss.dtype == torch.float32
    assert loss.item() == .000244140625
    assert reference.grad is None


@pytest.mark.parametrize('accumulation_steps', [1, 2])
def test_replay_forward_waits_for_policy_backward_and_preserves_summed_gradient(accumulation_steps):
    from rl_sr.training import backward_policy_then_replay
    parameter = nn.Parameter(torch.tensor(2.))
    policy_loss = (parameter - 1).square()
    order = []
    policy_loss.register_hook(lambda gradient: order.append('policy_backward'))

    def replay_forward():
        assert order == ['policy_backward']
        assert parameter.grad.item() == 2 / accumulation_steps
        return {'loss': .5 * (parameter - 3).square(), 'source': 'paired_replay'}

    total, replay = backward_policy_then_replay(
        policy_loss, replay_forward, lambda loss: (loss / accumulation_steps).backward())
    # d[(p-1)^2 + .5*(p-3)^2]/dp at p=2 is 1.
    assert parameter.grad.item() == 1 / accumulation_steps
    assert total.item() == 1.5 and not total.requires_grad
    assert replay['source'] == 'paired_replay'
    # A second backward would only succeed if the large policy graph was retained.
    with pytest.raises(RuntimeError, match='second time'):
        policy_loss.backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA AMP regression')
def test_topk_router_scatter_accepts_cuda_autocast_softmax_dtype():
    from models.lora_moe import route_logits
    logits = torch.tensor([[1., 2., 3., 4.]], device='cuda', dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        alpha = route_logits(logits, mode='topk', top_k=2, temperature=.7)
    assert (alpha != 0).sum().item() == 2
    assert torch.allclose(alpha.sum(dim=-1).float(), torch.ones(1, device='cuda'), atol=1e-6)
    (alpha * torch.tensor([[1., 3., 2., 4.]], device='cuda')).sum().backward()
    assert logits.grad.abs().sum() > 0 and torch.isfinite(logits.grad).all()


def test_c_rejects_missing_f0_before_loading_data_or_gpu():
    from train_rg_flux_refiner_sft import _require_refinement_config
    with pytest.raises(ValueError, match='init_adapter'):
        _require_refinement_config({'model': {'flux_backend': 'flux2_klein'},
            'training': {'freeze_flux_transformer': True},
            'condition': {'refinement': {'enabled': True}}})


def test_spatial_anchor_distinguishes_left_and_right_and_starts_as_noop():
    adapter = SRRefinementConditionAdapter(4, 6, anchor_tokens=4, anchor_mode='spatial', attention_dim=4)
    prompt = torch.randn(1, 5, 6)
    left = torch.zeros(1, 4, 4, 4)
    right = left.clone()
    left[:, :, :, :2] = 1
    right[:, :, :, 2:] = 1
    assert torch.equal(adapter(prompt, left, 2), prompt)
    with torch.no_grad():
        adapter.anchor_gate.fill_(1)
    first = adapter(prompt, left, 2)
    second = adapter(prompt, right, 2)
    assert not torch.allclose(first, second)
    first.square().mean().backward()
    assert adapter.anchor_query.weight.grad.abs().sum() > 0


def test_spatial_anchor_requires_square_grid():
    with pytest.raises(ValueError, match='square'):
        SRRefinementConditionAdapter(4, 6, anchor_tokens=8, anchor_mode='spatial')


def test_round_sampler_mixes_rounds_and_is_epoch_reproducible():
    from dataloaders.sr_refinement_state_dataset import RoundBalancedBatchSampler
    rounds = [2] * 9 + [3] * 2 + [4] * 5
    sampler = RoundBalancedBatchSampler(rounds, batch_size=4, seed=42)
    first = list(sampler)
    assert len(first) == 4
    assert all({rounds[i] for i in batch} == {2, 3, 4} for batch in first)
    assert first == list(RoundBalancedBatchSampler(rounds, batch_size=4, seed=42))
    sampler.set_epoch(1)
    assert list(sampler) != first


@pytest.mark.parametrize('rounds,batch', [([2, 2, 2, 2], 4), ([2, 3], 1)])
def test_round_sampler_rejects_fake_multiround_batches(rounds, batch):
    from dataloaders.sr_refinement_state_dataset import RoundBalancedBatchSampler
    with pytest.raises(ValueError, match='round|batch'):
        RoundBalancedBatchSampler(rounds, batch_size=batch)


def test_round_loader_understands_scored_rollout_states():
    from dataloaders.sr_refinement_state_dataset import refinement_round_indices
    dataset = SimpleNamespace(records=[SimpleNamespace(state_id='a'), SimpleNamespace(state_id='b')],
                              states={'a': SimpleNamespace(round_index=2), 'b': SimpleNamespace(round_index=3)})
    assert refinement_round_indices(dataset) == [2, 3]


def test_round_sampler_covers_late_rounds_when_rounds_exceed_batch_size():
    from dataloaders.sr_refinement_state_dataset import RoundBalancedBatchSampler
    rounds = list(range(2, 10))
    sampler = RoundBalancedBatchSampler(rounds, batch_size=4)
    assert {rounds[i] for batch in sampler for i in batch} == set(rounds)


def test_round_sampler_rotates_dropped_round_slots_across_epochs():
    from dataloaders.sr_refinement_state_dataset import RoundBalancedBatchSampler
    rounds = list(range(2, 12))
    sampler = RoundBalancedBatchSampler(rounds, batch_size=4)
    seen = set()
    for epoch in range(2):
        sampler.set_epoch(epoch)
        seen.update(rounds[i] for batch in sampler for i in batch)
    assert seen == set(rounds)


def test_image_loss_backpropagates_through_frozen_decoder():
    from rl_sr.training import refinement_image_losses
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1, bias=False)
            with torch.no_grad():
                self.conv.weight.copy_(torch.eye(3).reshape(3, 3, 1, 1))
            self.requires_grad_(False)
        def decode_latents_for_loss(self, z):
            return self.conv(z)
    decoder = Decoder()
    latent = torch.zeros(1, 3, 8, 8, requires_grad=True)
    result = refinement_image_losses(decoder, {'rl_sr': {'image_loss': {'charb_weight': .1}}},
                                     0, latent, torch.ones_like(latent))
    assert result['loss_image'].item() == pytest.approx(.10000005, abs=1e-6)
    result['loss_image'].backward()
    assert latent.grad.abs().sum() > 0
    assert decoder.conv.weight.grad is None


def test_disabled_image_losses_do_not_need_decoder():
    from rl_sr.training import refinement_image_losses
    latent = torch.randn(1, 3, 8, 8, requires_grad=True)
    result = refinement_image_losses(None, {}, 0, latent, torch.zeros_like(latent))
    assert result['loss_image'].item() == 0


def test_checkpoint_roundtrip_keeps_spatial_condition_semantics(tmp_path):
    artist = small_artist()
    artist.refinement_condition_adapter = SRRefinementConditionAdapter(4, 6, anchor_tokens=4,
        anchor_mode='spatial', attention_dim=4)
    with torch.no_grad():
        artist.refinement_condition_adapter.anchor_gate.fill_(.5)
    artist.save_trainable(tmp_path)
    metadata = json.loads((tmp_path / 'rg_flux_checkpoint_meta.json').read_text())
    assert metadata['refinement_condition']['anchor_mode'] == 'spatial'
    restored = small_artist()
    restored.refinement_condition_adapter = SRRefinementConditionAdapter(4, 6, anchor_tokens=4,
        anchor_mode='spatial', attention_dim=4)
    restored.load_trainable(tmp_path)
    prompt, anchor = torch.randn(1, 5, 6), torch.randn(1, 4, 4, 4)
    assert torch.equal(artist.refinement_condition_adapter(prompt, anchor, 2),
                       restored.refinement_condition_adapter(prompt, anchor, 2))
    legacy = small_artist()
    legacy.refinement_condition_adapter = SRRefinementConditionAdapter(4, 6, anchor_tokens=4)
    with pytest.raises(RuntimeError, match='refinement|anchor'):
        legacy.load_trainable(tmp_path)


def test_old_refiner_cannot_silently_load_into_spatial_anchor(tmp_path):
    artist = small_artist()
    artist.refinement_condition_adapter = SRRefinementConditionAdapter(4, 6)
    torch.save({'refinement_condition_adapter.anchor_gate': torch.tensor(.5)}, tmp_path / 'condition_adapters.pt')
    artist.refinement_condition_adapter = SRRefinementConditionAdapter(4, 6, anchor_tokens=4,
        anchor_mode='spatial', attention_dim=4)
    with pytest.raises(RuntimeError, match='legacy|refinement'):
        artist.load_trainable(tmp_path)


def test_spatial_checkpoint_requires_complete_condition_file(tmp_path):
    artist = small_artist()
    artist.refinement_condition_adapter = SRRefinementConditionAdapter(4, 6, anchor_tokens=4,
        anchor_mode='spatial', attention_dim=4)
    artist.save_trainable(tmp_path)
    (tmp_path / 'condition_adapters.pt').unlink()
    with pytest.raises(FileNotFoundError, match='condition_adapters'):
        artist.load_trainable(tmp_path)
    artist.refinement_condition_adapter = SRRefinementConditionAdapter(4, 6)
    with pytest.raises(RuntimeError, match='Refinement|anchor'):
        artist.load_trainable(tmp_path)


def test_spatial_checkpoint_rejects_missing_query_weights(tmp_path):
    artist = small_artist()
    artist.refinement_condition_adapter = SRRefinementConditionAdapter(4, 6, anchor_tokens=4,
        anchor_mode='spatial', attention_dim=4)
    artist.save_trainable(tmp_path)
    path = tmp_path / 'condition_adapters.pt'
    state = torch.load(path, weights_only=True)
    del state['refinement_condition_adapter.anchor_query.weight']
    torch.save(state, path)
    with pytest.raises(RuntimeError, match='missing parameters'):
        artist.load_trainable(tmp_path)


def test_base_f0_without_refinement_weights_can_initialize_spatial_anchor(tmp_path):
    artist = small_artist()
    artist.refinement_condition_adapter = SRRefinementConditionAdapter(4, 6, anchor_tokens=4,
        anchor_mode='spatial', attention_dim=4)
    torch.save({'lora_adapter': torch.ones(1)}, tmp_path / 'condition_adapters.pt')
    artist.load_trainable(tmp_path)
    assert artist.lora_adapter.item() == 1
    assert artist.refinement_condition_adapter.anchor_gate.item() == 0


@pytest.fixture
def one_click_case(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    config = tmp_path / 'config.yaml'
    config.write_text('rl_sr: {}\n')
    source = tmp_path / 'train.jsonl'
    source.write_text('{}\n')
    checkpoint = tmp_path / 'checkpoint' / 'rg_flux_adapters'
    checkpoint.mkdir(parents=True)
    calls = tmp_path / 'calls.jsonl'
    context = dict(data_jsonl_path=str(source), evaluation_jsonl_path=str(source), train_max_samples=0,
        train_subset_seed=42, iterations=4, num_candidates=4, sft_max_steps=1, rl_max_steps=1,
        seed=42, metric_device='cpu', reward_device='cpu', dataset_id='toy', metrics='niqe', sft_bootstrap_rounds=3)
    shim = tmp_path / 'python_shim'
    shim.write_text('#!' + sys.executable + '\n' + '''
import os,json,sys
from pathlib import Path
args=sys.argv[1:]
with Path(os.environ['CAPTURE_CALLS']).open('a') as f: f.write(json.dumps(args)+'\\n')
def value(name): return args[args.index(name)+1]
if args[0].endswith('resolve_rl_sr_run_dir.py'):
    if '--field' in args: print(json.loads(os.environ['FAKE_CONTEXT'])[value('--field')])
    else: print(os.environ['RL_SR_RUN_DIR'])
else:
    for flag in ('--output','--output_jsonl'):
        if flag in args:
            p=Path(value(flag));p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}\\n')
    if '--output_dir' in args:
        p=Path(value('--output_dir'));p.mkdir(parents=True,exist_ok=True)
        (p/'sample_lineage.jsonl').write_text('{}\\n')
        (p/'round_01').mkdir(exist_ok=True)
        (p/'round_01'/'inference_manifest.json').write_text('{}')
''')
    shim.chmod(0o755)
    env = {**os.environ, 'PYTHON_BIN': str(shim), 'CONFIG': str(config),
           'F0_CHECKPOINT': str(checkpoint.parent), 'RL_SR_RUN_DIR': str(root),
           'CAPTURE_CALLS': str(calls), 'FAKE_CONTEXT': json.dumps(context)}
    for name in ('CONDA_ENV', 'F0_REUSE_DIR', 'F0_ADAPTER'):
        env.pop(name, None)
    return SimpleNamespace(env=env, calls=calls, root=root)


def test_one_click_c_bootstrap_collects_multiple_real_rounds(one_click_case):
    case = one_click_case
    completed = subprocess.run(['bash', 'tools/run_rl_sr_stage_ae.sh', 'f0'], env=case.env,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    commands = [json.loads(line) for line in case.calls.read_text().splitlines()]
    builder = next(args for args in commands if args[0].endswith('build_sr_refinement_states.py'))
    assert builder[builder.index('--max_round') + 1] == '4'
    bootstrap = next(args for args in commands if '--output_dir' in args
                     and '02_f0_bootstrap_trajectory' in args[args.index('--output_dir') + 1])
    assert bootstrap[bootstrap.index('--iterations') + 1] == '3'
    assert '--reuse_f0_dir' in bootstrap


def test_inference_round_count_does_not_block_cached_c_training(one_click_case):
    case = one_click_case
    (case.root / '02_states_for_c.jsonl').write_text('{}\n')
    case.env['ITERATIONS'] = '2'
    completed = subprocess.run(['bash', 'tools/run_rl_sr_stage_ae.sh', 'c'], env=case.env,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    commands = [json.loads(line) for line in case.calls.read_text().splitlines()]
    assert any(args[0].endswith('train_rg_flux_refiner_sft.py') for args in commands)
