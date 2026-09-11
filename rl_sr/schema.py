"""Portable identities and JSONL records for multi-round SR.

Artifact identities deliberately exclude absolute paths.  A record may contain
runtime paths so a job can load files on its current machine, but those paths
never participate in state, rollout, or policy fingerprints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "rl_sr_v1"


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        # JSON's native representation is deterministic for finite Python floats.
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("Canonical artifact payload cannot contain non-finite floats")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def content_tree_sha256(path: str | Path) -> str:
    """Hash a checkpoint file or directory using relative names and file bytes only.

    The root's absolute location is never included, so the same adapter copied
    to a different server retains its identity.
    """

    path = Path(path)
    if path.is_file():
        return content_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Cannot hash missing artifact: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Cannot hash empty artifact directory: {path}")
    for item in files:
        digest.update(normalize_relative_key(item.relative_to(path)).encode("utf-8"))
        digest.update(content_sha256(item).encode("ascii"))
    return digest.hexdigest()


def adapter_content_sha256(path: str | Path) -> str:
    """Hash only model adapters when a full training checkpoint is supplied.

    Training-state/optimizer files are intentionally excluded: they do not
    define G_ref, G_old, or G_phi and would make a policy identity depend on
    checkpoint-resume implementation details.
    """

    path = Path(path)
    nested_adapter_dir = path / "rg_flux_adapters"
    return content_tree_sha256(nested_adapter_dir if nested_adapter_dir.is_dir() else path)


def identity_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def normalize_relative_key(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Artifact identity requires a relative key, got absolute path: {value}")
    normalized = path.as_posix().lstrip("./")
    if not normalized or normalized == "." or ".." in Path(normalized).parts:
        raise ValueError(f"Invalid portable relative key: {value}")
    return normalized


def build_sample_key(dataset_id: str, relative_key: str | Path, content_hash: str) -> str:
    dataset_id = str(dataset_id).strip()
    if not dataset_id:
        raise ValueError("dataset_id must be non-empty")
    return identity_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "relative_key": normalize_relative_key(relative_key),
            "content_sha256": str(content_hash),
        }
    )


@dataclass(frozen=True)
class RuntimePaths:
    """Machine-local locations; intentionally excluded from portable identity."""

    original_lr: str | None = None
    current_sr: str | None = None
    hr: str | None = None
    sample_latent: str | None = None
    raw_sr: str | None = None
    executed_sr: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class StateRecord:
    dataset_id: str
    sample_key: str
    original_lr_key: str
    original_lr_sha256: str
    current_output_sha256: str
    round_index: int
    parent_state_id: str | None
    geometry_sha256: str
    producer_adapter_sha256: str
    sampler_sha256: str
    current_output_key: str | None = None
    hr_key: str | None = None
    hr_sha256: str | None = None
    prompt: str = ""
    prompt_sha256: str = ""
    router_condition_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime: RuntimePaths = field(default_factory=RuntimePaths, compare=False)
    state_id: str = ""

    def __post_init__(self):
        if int(self.round_index) < 2:
            raise ValueError("A refinement state must target round_index >= 2")
        if not self.prompt_sha256:
            object.__setattr__(self, "prompt_sha256", hashlib.sha256(self.prompt.encode("utf-8")).hexdigest())
        if not self.router_condition_sha256:
            condition = self.metadata.get("router_condition") if isinstance(self.metadata, Mapping) else None
            if condition is not None:
                object.__setattr__(self, "router_condition_sha256", identity_sha256(condition))
        if not self.state_id:
            identity = self.identity_payload()
            object.__setattr__(self, "state_id", identity_sha256(identity))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "sample_key": self.sample_key,
            "original_lr_key": normalize_relative_key(self.original_lr_key),
            "original_lr_sha256": self.original_lr_sha256,
            "current_output_key": None
            if self.current_output_key is None
            else normalize_relative_key(self.current_output_key),
            "current_output_sha256": self.current_output_sha256,
            "round_index": int(self.round_index),
            "parent_state_id": self.parent_state_id,
            "geometry_sha256": self.geometry_sha256,
            "producer_adapter_sha256": self.producer_adapter_sha256,
            "sampler_sha256": self.sampler_sha256,
            "hr_key": None if self.hr_key is None else normalize_relative_key(self.hr_key),
            "hr_sha256": self.hr_sha256,
            "prompt_sha256": self.prompt_sha256,
            "router_condition_sha256": self.router_condition_sha256,
        }

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runtime"] = self.runtime.as_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateRecord":
        values = dict(payload)
        values["runtime"] = RuntimePaths(**(values.get("runtime") or {}))
        return cls(**values)


@dataclass(frozen=True)
class RolloutRecord:
    state_id: str
    policy_adapter_sha256: str
    candidate_index: int
    sample_latent_sha256: str
    raw_sr_sha256: str
    executed_sr_sha256: str
    sampler_sha256: str
    reward_version: str | None = None
    reward_vector: dict[str, float] = field(default_factory=dict)
    total_reward: float | None = None
    advantage: float | None = None
    optimality_probability: float | None = None
    confidence: float | None = None
    hard_violation: bool = False
    runtime: RuntimePaths = field(default_factory=RuntimePaths, compare=False)
    rollout_id: str = ""

    def __post_init__(self):
        if int(self.candidate_index) < 0:
            raise ValueError("candidate_index must be non-negative")
        if not self.rollout_id:
            object.__setattr__(self, "rollout_id", identity_sha256(self.identity_payload()))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "state_id": self.state_id,
            "policy_adapter_sha256": self.policy_adapter_sha256,
            "candidate_index": int(self.candidate_index),
            "sample_latent_sha256": self.sample_latent_sha256,
            "raw_sr_sha256": self.raw_sr_sha256,
            "executed_sr_sha256": self.executed_sr_sha256,
            "sampler_sha256": self.sampler_sha256,
            "reward_version": self.reward_version,
        }

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runtime"] = self.runtime.as_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RolloutRecord":
        values = dict(payload)
        values["runtime"] = RuntimePaths(**(values.get("runtime") or {}))
        return cls(**values)
