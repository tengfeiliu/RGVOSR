"""Datasets for transitions (x, y_k) -> y_{k+1} used by RL-SR stages C/E."""

from __future__ import annotations

from pathlib import Path
import math
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms

from rl_sr.artifact_store import read_jsonl
from rl_sr.schema import StateRecord


class RoundBalancedBatchSampler(Sampler):
    """Mix distinct target rounds per batch; oversample scarce round buckets.

    A sampling epoch has floor(N / B) batches. It is not a once-only traversal:
    buckets wrap with a new shuffle to avoid majority rounds dominating.
    """

    def __init__(self, round_indices, batch_size, seed=42, min_rounds=2, drop_last=True):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.drop_last = bool(drop_last)
        self.buckets = {}
        for index, round_index in enumerate(round_indices):
            self.buckets.setdefault(int(round_index), []).append(index)
        self.rounds = sorted(self.buckets)
        if int(min_rounds) < 2:
            raise ValueError("Balanced batches require min_rounds >= 2")
        if len(self.rounds) < int(min_rounds) or self.batch_size < int(min_rounds):
            raise ValueError("Round-balanced batches need >=2 rounds and batch_size >= min_rounds; "
                             "rebuild single-round bootstrap states or disable round_balanced_batches")
        self.count = len(round_indices)
        if self.count < self.batch_size:
            raise ValueError("Dataset is smaller than one balanced batch")

    def __len__(self):
        return self.count // self.batch_size if self.drop_last else math.ceil(self.count / self.batch_size)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        buckets = {key: list(indices) for key, indices in self.buckets.items()}
        for indices in buckets.values():
            rng.shuffle(indices)
        cursors = {key: 0 for key in buckets}
        round_offset = self.epoch * len(self) * self.batch_size
        for batch_index in range(len(self)):
            batch = []
            for slot in range(self.batch_size):
                key = self.rounds[(round_offset + batch_index * self.batch_size + slot) % len(self.rounds)]
                if cursors[key] == len(buckets[key]):
                    rng.shuffle(buckets[key])
                    cursors[key] = 0
                batch.append(buckets[key][cursors[key]])
                cursors[key] += 1
            rng.shuffle(batch)
            yield batch


def refinement_round_indices(dataset):
    if hasattr(dataset, "states"):
        return [dataset.states[record.state_id].round_index for record in dataset.records]
    return [record.round_index for record in dataset.records]


def build_refinement_dataloader(dataset, batch_size, num_workers, collate_fn, config):
    settings = config.get("rl_sr") or {}
    kwargs = dict(num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
                  persistent_workers=num_workers > 0)
    if settings.get("round_balanced_batches", False):
        sampler = RoundBalancedBatchSampler(refinement_round_indices(dataset), batch_size,
                    seed=int(settings.get("seed", config.get("training", {}).get("seed", 42))),
                    min_rounds=int(settings.get("min_rounds_per_batch", 2)))
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)


def set_refinement_loader_epoch(loader, epoch):
    # Accelerator can wrap batch_sampler. Walk wrappers without discarding
    # the original per-batch mixing contract.
    sampler = loader.batch_sampler
    while sampler is not None:
        if isinstance(sampler, RoundBalancedBatchSampler):
            sampler.set_epoch(epoch)
            return
        sampler = getattr(sampler, "batch_sampler", None)


class SRRefinementStateDataset(Dataset):
    """Loads portable state identities and current-machine runtime paths.

    The JSONL's state identity deliberately has no absolute paths.  Runtime
    paths are only used here to open images on the current machine.  Rebuild
    the state manifest on a different server if its dataset mount changes.
    """

    def __init__(self, state_jsonl, require_hr=True, vae_align=16):
        self.state_jsonl = Path(state_jsonl)
        self.require_hr = bool(require_hr)
        self.vae_align = int(vae_align)
        self.to_tensor = transforms.ToTensor()
        if not self.state_jsonl.is_file():
            raise FileNotFoundError(f"Refinement state JSONL not found: {self.state_jsonl}")
        self.records = [StateRecord.from_dict(row) for row in read_jsonl(self.state_jsonl)]
        if not self.records:
            raise RuntimeError(f"No refinement states found in {self.state_jsonl}")
        self._validate_records()

    def _validate_records(self):
        missing = []
        for record in self.records:
            for label, value in (
                ("original_lr", record.runtime.original_lr),
                ("current_sr", record.runtime.current_sr),
                ("hr", record.runtime.hr if self.require_hr else None),
            ):
                if value and not Path(value).is_file():
                    missing.append(f"{record.state_id}:{label}={value}")
                if label in {"original_lr", "current_sr"} and not value:
                    missing.append(f"{record.state_id}:{label}=missing")
        if missing:
            preview = "; ".join(missing[:3])
            raise FileNotFoundError(
                "Refinement state runtime paths are unavailable on this machine. "
                "Rebuild the state manifest from the portable lineage on this server. "
                f"Examples: {preview}"
            )

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _load_rgb(path):
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")

    def _tensor(self, image):
        return self.to_tensor(image).mul(2.0).sub(1.0)

    def __getitem__(self, index):
        record = self.records[index]
        original_lr = self._load_rgb(record.runtime.original_lr)
        current_sr = self._load_rgb(record.runtime.current_sr)
        if current_sr.width % self.vae_align or current_sr.height % self.vae_align:
            raise ValueError(
                f"State {record.state_id} current SR size {current_sr.size} must be divisible by "
                f"vae_align={self.vae_align}"
            )
        original_lr_up = original_lr.resize(current_sr.size, Image.Resampling.BICUBIC)
        output = {
            "original_lr": self._tensor(original_lr),
            "original_lr_up": self._tensor(original_lr_up),
            "current_sr": self._tensor(current_sr),
            "prompt": record.prompt,
            "round_index": torch.tensor(record.round_index, dtype=torch.long),
            "state_id": record.state_id,
            "sample_key": record.sample_key,
            "router_condition": record.metadata.get("router_condition"),
        }
        if self.require_hr:
            hr = self._load_rgb(record.runtime.hr)
            if hr.size != current_sr.size:
                raise ValueError(
                    f"State {record.state_id} HR geometry {hr.size} differs from current SR {current_sr.size}. "
                    "Build states from geometry-matched paired crops."
                )
            output["hr"] = self._tensor(hr)
        return output


def sr_refinement_state_collate_fn(batch):
    if not batch:
        raise ValueError("Cannot collate an empty refinement batch")
    tensor_keys = ("original_lr", "original_lr_up", "current_sr", "round_index", "hr")
    output = {}
    for key in tensor_keys:
        if key in batch[0]:
            output[key] = torch.stack([item[key] for item in batch], dim=0)
    for key in ("prompt", "state_id", "sample_key"):
        output[key] = [item[key] for item in batch]
    output["router_condition"] = [item["router_condition"] for item in batch]
    return output
