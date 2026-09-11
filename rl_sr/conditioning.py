"""Recover frozen MoE routing inputs from portable refinement-state metadata."""

from __future__ import annotations

import torch

from models.router_condition import ROUTER_CONDITION_VERSION, ROUTER_CONDITION_KEYS


def _cfg(config, path, default=None):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _metadata(value):
    if hasattr(value, "metadata"):
        return value.metadata
    if isinstance(value, dict):
        if "values" in value and "valid_mask" in value:
            return {"router_condition": value}
        return value
    return None


def state_router_condition_tensors(states, config, device, dtype):
    """Return fixed condition8 router inputs or `(None, None, None)`.

    The router itself remains frozen; persisting its original condition avoids
    accidentally substituting a condition extracted from a generated y_k.
    """

    mode = str(_cfg(config, "model.lora_moe.router_input_mode", "prompt_lr"))
    if mode not in {"condition8", "condition8_timestep"}:
        return None, None, None
    expected_version = str(_cfg(config, "model.lora_moe.router_condition_version", ROUTER_CONDITION_VERSION))
    values, masks, confidences = [], [], []
    for state in states:
        metadata = _metadata(state)
        condition = metadata.get("router_condition") if isinstance(metadata, dict) else None
        if not isinstance(condition, dict):
            raise ValueError(
                "State metadata has no router_condition for condition8 routing. "
                "Rebuild it with tools/build_sr_refinement_states.py from the paired source JSONL."
            )
        version = str(condition.get("extractor_version") or "")
        if version != expected_version:
            raise ValueError(
                f"State router-condition version {version!r} does not match config {expected_version!r}"
            )
        vector = condition.get("values")
        mask = condition.get("valid_mask")
        if not isinstance(vector, (list, tuple)) or len(vector) != len(ROUTER_CONDITION_KEYS):
            raise ValueError("State router_condition.values must have eight entries")
        if not isinstance(mask, (list, tuple)) or len(mask) != len(ROUTER_CONDITION_KEYS):
            raise ValueError("State router_condition.valid_mask must have eight entries")
        values.append(vector)
        masks.append(mask)
        confidences.append(float(condition.get("confidence", 0.0)))
    values = torch.tensor(values, device=device, dtype=dtype)
    masks = torch.tensor(masks, device=device, dtype=dtype)
    confidences = torch.tensor(confidences, device=device, dtype=dtype)
    return values, masks, confidences
