"""Datasets for transitions (x, y_k) -> y_{k+1} used by RL-SR stages C/E."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from rl_sr.artifact_store import read_jsonl
from rl_sr.schema import StateRecord


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
