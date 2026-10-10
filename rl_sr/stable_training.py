"""SFT optimizer controls and finite, identifiable adapter checkpoints.

These controls do not change the paired supervision or flow-matching losses.
Existing condition adapters retain the refinement learning rate; only LoRA
parameters receive the optional lower learning rate or explicit freeze.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import torch

from rl_sr.artifact_store import write_json_atomic
from rl_sr.schema import content_sha256, content_tree_sha256
from rl_sr.snapshots import configure_rl_trainables, is_rl_trainable_name


SFT_LOSS_FIELDS = ("loss", "loss_fm", "loss_latent", "loss_charb", "loss_lpips")


@dataclass(frozen=True)
class SFTControls:
    lora_lr: float
    refinement_lr: float
    train_lora: bool
    checkpoint_every: int
    weight_decay: float


def resolve_sft_controls(config) -> SFTControls:
    settings = config.get("rl_sr") or {}
    training = config.get("training") or {}
    default_lr = float(settings.get("sft_lr", training.get("lr_adapter", 1.0e-4)))
    lora_lr = float(settings.get("sft_lora_lr", default_lr))
    refinement_lr = float(settings.get("sft_refinement_lr", default_lr))
    for name, value in (("sft_lora_lr", lora_lr), ("sft_refinement_lr", refinement_lr)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"rl_sr.{name} must be finite and positive")
    train_lora = settings.get("sft_train_lora", True)
    if not isinstance(train_lora, bool):
        raise ValueError("rl_sr.sft_train_lora must be a boolean")
    checkpoint_every = settings.get("sft_checkpoint_every", 0)
    if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every < 0:
        raise ValueError("rl_sr.sft_checkpoint_every must be a nonnegative integer")
    weight_decay = float(settings.get("weight_decay", training.get("weight_decay", 0.01)))
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("SFT weight_decay must be finite and nonnegative")
    return SFTControls(lora_lr, refinement_lr, train_lora, checkpoint_every, weight_decay)


def build_sft_parameter_groups(artist, config):
    controls = resolve_sft_controls(config)
    candidates = configure_rl_trainables(artist, train_router=False)
    trainable = []
    lora, refinement = [], []
    for name, parameter in candidates:
        is_lora = "lora_" in name.lower()
        if is_lora and not controls.train_lora:
            parameter.requires_grad_(False)
            continue
        trainable.append((name, parameter))
        (lora if is_lora else refinement).append(parameter)
    if not trainable:
        raise RuntimeError("SFT has no trainable condition adapters after freezing LoRA")
    groups = []
    for name, parameters, lr in (("lora", lora, controls.lora_lr),
                                 ("refinement", refinement, controls.refinement_lr)):
        if parameters:
            groups.append({"name": name, "params": parameters, "lr": lr,
                           "weight_decay": controls.weight_decay})
    return trainable, groups, controls


def _raise_if_any_rank_nonfinite(nonfinite, accelerator, description):
    if accelerator is not None:
        flag = torch.tensor(int(bool(nonfinite)), device=accelerator.device, dtype=torch.int32)
        nonfinite_on_any_rank = bool(accelerator.reduce(flag, reduction="sum").item())
    else:
        nonfinite_on_any_rank = bool(nonfinite)
    if nonfinite_on_any_rank:
        detail = ", ".join(nonfinite[:8]) if nonfinite else "another training rank"
        raise FloatingPointError(f"SFT {description} must be finite; invalid: {detail}")


def finite_sft_metrics(metrics, accelerator=None, required=()):
    missing = set(required) - set(metrics)
    if missing:
        raise ValueError(f"SFT checkpoint requires measured metrics: {sorted(missing)}")
    values = {}
    for name, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().float().item()
        values[name] = float(value)
    nonfinite = [name for name, value in values.items() if not math.isfinite(value)]
    _raise_if_any_rank_nonfinite(nonfinite, accelerator, "metrics")
    return values


def _weights_sha256(adapter_dir):
    files = sorted(path for path in Path(adapter_dir).rglob("*")
                   if path.is_file() and path.suffix in {".pt", ".bin", ".safetensors"})
    if not files:
        raise ValueError(f"SFT checkpoint contains no adapter weight files: {adapter_dir}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(adapter_dir).as_posix().encode("utf-8"))
        digest.update(content_sha256(path).encode("ascii"))
    return digest.hexdigest()


def save_sft_checkpoint(artist, checkpoint_dir, progress, metrics, accelerator=None):
    """Save a complete adapter policy, including frozen LoRA and router tensors.

    Every rank participates in adapter gathering; only the main rank writes.
    The sidecar is outside rg_flux_adapters so it cannot alter policy identity.
    This is an inference checkpoint, not an optimizer-resume checkpoint.
    """
    values = finite_sft_metrics(metrics, accelerator, required=(*SFT_LOSS_FIELDS, "grad_norm"))
    if int(progress.get("optimizer_updates", 0)) <= 0:
        raise ValueError("SFT checkpoints require at least one actual optimizer update")
    nonfinite = [name for name, parameter in artist.named_parameters()
                 if is_rl_trainable_name(name) and not torch.isfinite(parameter.detach()).all().item()]
    _raise_if_any_rank_nonfinite(nonfinite, accelerator, "adapter weights")
    checkpoint_dir = Path(checkpoint_dir)
    main_process = accelerator is None or accelerator.is_main_process
    if accelerator is not None:
        accelerator.wait_for_everyone()
    adapter_dir = checkpoint_dir / "rg_flux_adapters"
    artist.save_trainable(adapter_dir, save_files=main_process)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    if not main_process:
        return None
    summary = {**progress, "metrics": values, "adapter_sha256": content_tree_sha256(adapter_dir),
               "weights_sha256": _weights_sha256(adapter_dir), "checkpoint_kind": "complete_adapter_policy"}
    write_json_atomic(checkpoint_dir / "checkpoint_summary.json", summary)
    return summary
