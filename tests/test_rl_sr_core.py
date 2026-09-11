from pathlib import Path

import torch
from torch import nn

from losses.output_nft import group_optimality_probabilities, nft_velocity_loss, renoise_rectified_flow
from models.sr_refinement_condition import SRRefinementConditionAdapter
from rewards.sr_reward_v1 import AutoSRRewardV1
from rg_flux_fm import sample_multistep_fm
from rl_sr.schema import RuntimePaths, StateRecord
from rl_sr.snapshots import AdapterSnapshot, assert_rl_frozen_backbone, configure_rl_trainables


def test_portable_state_identity_excludes_runtime_paths():
    common = dict(
        dataset_id="toy",
        sample_key="sample",
        original_lr_key="lr/a.png",
        original_lr_sha256="lrhash",
        current_output_key="artifacts/r1/a.png",
        current_output_sha256="yhash",
        round_index=2,
        parent_state_id=None,
        geometry_sha256="geometry",
        producer_adapter_sha256="adapter",
        sampler_sha256="sampler",
    )
    first = StateRecord(**common, runtime=RuntimePaths(original_lr="D:/server-a/a.png"))
    second = StateRecord(**common, runtime=RuntimePaths(original_lr="/mnt/server-b/a.png"))
    assert first.state_id == second.state_id
    assert "server" not in first.identity_payload()["original_lr_key"]


def test_refinement_condition_starts_as_exact_noop():
    adapter = SRRefinementConditionAdapter(latent_channels=4, context_dim=6, anchor_tokens=2)
    prompt = torch.randn(2, 5, 6)
    anchor = torch.randn(2, 4, 3, 3)
    assert torch.equal(adapter(prompt, anchor, torch.tensor([2, 3])), prompt)
    with torch.no_grad():
        adapter.anchor_gate.fill_(1.0)
    assert not torch.equal(adapter(prompt, anchor, torch.tensor([2, 3])), prompt)


def test_nft_has_finite_gradient_and_group_relative_probabilities():
    current = torch.randn(3, 4, 2, 2, requires_grad=True)
    old = torch.randn_like(current)
    target = torch.randn_like(current)
    loss = nft_velocity_loss(current, old, target, torch.tensor([0.0, 0.5, 1.0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(current.grad).all()
    probability = group_optimality_probabilities(torch.tensor([0.1, 0.5, 0.9]))
    assert probability[2] > probability[1] > probability[0]
    z_t, velocity = renoise_rectified_flow(torch.randn(2, 4, 2, 2), tau=torch.tensor([0.2, 0.8]))
    assert z_t.shape == velocity.shape


def test_reward_v1_runs_without_pyiqa_in_diagnostic_mode():
    original = torch.rand(1, 3, 8, 8)
    current = torch.rand(1, 3, 16, 16)
    candidate = current.clone()
    result = AutoSRRewardV1(enable_quality=False).score(original, current, candidate)[0]
    assert result.version.startswith("sr_reward_v1")
    assert 0.0 <= result.confidence <= 1.0


def test_rl_guard_never_enables_backbone_parameters():
    class Artist(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Linear(2, 2)
            self.lora_adapter = nn.Parameter(torch.zeros(1))
            self.refinement_condition_adapter = nn.Linear(2, 2)

    artist = Artist()
    names = [name for name, _ in configure_rl_trainables(artist)]
    assert "lora_adapter" in names
    assert not artist.transformer.weight.requires_grad
    assert artist.refinement_condition_adapter.weight.requires_grad
    assert_rl_frozen_backbone(artist)
    snapshot = AdapterSnapshot.capture(artist, "policy", train_router=False)
    snapshot.apply(artist)


def test_legacy_sampler_artist_never_receives_refinement_kwargs():
    class LegacyArtist:
        def __call__(
            self,
            z_t,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids=None,
            degradation_vector=None,
            z_lr=None,
            dino_tokens=None,
            lr_cond_mode=None,
            router_condition=None,
            router_condition_mask=None,
            router_condition_confidence=None,
        ):
            return torch.zeros_like(z_t)

    prompt = torch.zeros(1, 2, 3)
    output = sample_multistep_fm(
        artist=LegacyArtist(),
        shape=(1, 4, 2, 2),
        prompt_embeds=prompt,
        pooled_prompt_embeds=torch.zeros(1, 0),
        num_steps=1,
        device=prompt.device,
        dtype=prompt.dtype,
    )
    assert output.shape == (1, 4, 2, 2)
