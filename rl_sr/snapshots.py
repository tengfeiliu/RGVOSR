"""Adapter-only snapshots and strict frozen-backbone guards for RL-SR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch


DEFAULT_TRAINABLE_MARKERS = (
    "lora_",
    "shared_lora_",
    "routed_lora_",
    "moe_router.",
    "degradation_encoder.",
    "lr_condition_encoder.",
    "visual_condition_adapter.",
    "refinement_condition_adapter.",
)


def is_rl_trainable_name(name: str, markers: Iterable[str] = DEFAULT_TRAINABLE_MARKERS) -> bool:
    normalized = str(name).lower()
    return any(str(marker).lower() in normalized for marker in markers)


def configure_rl_trainables(artist, train_router: bool = False) -> list[tuple[str, torch.nn.Parameter]]:
    """Freeze every parameter except approved adapters, returning optimizer params.

    The FLUX transformer can contain LoRA tensors, so this checks parameter
    names rather than freezing the whole transformer module after LoRA setup.
    """

    allowed = list(DEFAULT_TRAINABLE_MARKERS)
    if not train_router:
        allowed = [marker for marker in allowed if marker != "moe_router."]
    trainable: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in artist.named_parameters():
        enabled = is_rl_trainable_name(name, allowed)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append((name, parameter))
    if not trainable:
        raise RuntimeError("RL-SR found no trainable LoRA or condition-adapter parameters")
    assert_rl_frozen_backbone(artist, train_router=train_router)
    return trainable


def assert_rl_frozen_backbone(artist, train_router: bool = False) -> None:
    allowed = list(DEFAULT_TRAINABLE_MARKERS)
    if not train_router:
        allowed = [marker for marker in allowed if marker != "moe_router."]
    unexpected = [
        name
        for name, parameter in artist.named_parameters()
        if parameter.requires_grad and not is_rl_trainable_name(name, allowed)
    ]
    if unexpected:
        preview = ", ".join(unexpected[:8])
        raise RuntimeError(
            "RL-SR requires the FLUX.2 backbone to remain frozen. "
            f"Unexpected trainable parameters: {preview}"
        )


def _tensor_digest(digest, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())


@dataclass(frozen=True)
class AdapterSnapshot:
    label: str
    state: dict[str, torch.Tensor]
    sha256: str
    train_router: bool = False

    @classmethod
    def capture(cls, artist, label: str, train_router: bool = False) -> "AdapterSnapshot":
        allowed = list(DEFAULT_TRAINABLE_MARKERS)
        if not train_router:
            allowed = [marker for marker in allowed if marker != "moe_router."]
        state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in artist.named_parameters()
            if is_rl_trainable_name(name, allowed)
        }
        if not state:
            raise RuntimeError("Cannot capture an empty RL-SR adapter snapshot")
        digest = hashlib.sha256()
        for name in sorted(state):
            digest.update(name.encode("utf-8"))
            _tensor_digest(digest, state[name])
        return cls(label=str(label), state=state, sha256=digest.hexdigest(), train_router=bool(train_router))

    def apply(self, artist, strict: bool = True) -> None:
        current = dict(artist.named_parameters())
        missing = [name for name in self.state if name not in current]
        if missing:
            raise RuntimeError(f"Snapshot '{self.label}' has unknown parameters: {missing[:4]}")
        if strict:
            allowed = list(DEFAULT_TRAINABLE_MARKERS)
            if not self.train_router:
                allowed = [marker for marker in allowed if marker != "moe_router."]
            current_allowed = {
                name for name in current if is_rl_trainable_name(name, allowed)
            }
            absent = current_allowed - set(self.state)
            if absent:
                raise RuntimeError(
                    f"Snapshot '{self.label}' is missing current adapter parameters: {sorted(absent)[:4]}"
                )
        with torch.no_grad():
            for name, tensor in self.state.items():
                parameter = current[name]
                if tuple(parameter.shape) != tuple(tensor.shape):
                    raise RuntimeError(
                        f"Snapshot shape mismatch for {name}: checkpoint={tuple(tensor.shape)} "
                        f"current={tuple(parameter.shape)}"
                    )
                parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
