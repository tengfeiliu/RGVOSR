"""Paired, already-generated RGB proposals for offline acceptance training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SRUpdateDataset(Dataset):
    """Load real transitions without changing their current/proposal pairing.

    Identity examples must be generated with HR as the generator's current
    image and explicitly labelled ``identity: true`` in the manifest. This
    dataset never synthesizes identity examples from an ordinary proposal.
    Cropping and optional flips apply to all four images together.
    """

    image_fields = ("original_lr", "current_sr", "proposal", "hr")

    def __init__(self, manifest, crop_size=None, augment=False, seed=42):
        self.manifest = Path(manifest).resolve()
        self.crop_size = None if crop_size is None else int(crop_size)
        if self.crop_size is not None and self.crop_size < 1:
            raise ValueError("crop_size must be positive or None")
        self.augment = bool(augment)
        self.rng = random.Random(int(seed))
        manifest_bytes = self.manifest.read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.records = []
        for line_number, line in enumerate(manifest_bytes.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on manifest line {line_number}: {exc}") from exc
            self._validate_record(record, line_number)
            self.records.append(record)
        if not self.records:
            raise ValueError(f"No proposal rows in manifest {self.manifest}")

    def _validate_record(self, record, line_number):
        label = f"Manifest line {line_number}"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        for field in ("sample_key", "producer"):
            if not record.get(field):
                raise ValueError(f"{label} requires {field}")
        if not isinstance(record["sample_key"], str):
            raise ValueError(f"{label} sample_key must identify the original HR source")
        target_round = record.get("target_round")
        if isinstance(target_round, bool) or not isinstance(target_round, int) or target_round < 1:
            raise ValueError(f"{label} target_round must be a positive integer")
        if not isinstance(record.get("identity", False), bool):
            raise ValueError(f"{label} identity must be a boolean")
        sizes = {}
        for field in self.image_fields:
            value = record.get(field)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"{label} {field} must be an absolute image path")
            if not Path(value).is_file():
                raise FileNotFoundError(f"{label} {field} is unavailable: {value}")
            with Image.open(value) as image:
                sizes[field] = image.size
        aligned = {sizes[field] for field in ("current_sr", "proposal", "hr")}
        if len(aligned) != 1:
            raise ValueError(f"{label} geometry must align current_sr, proposal and hr: {sizes}")
        if self.crop_size is not None and min(sizes["current_sr"]) < self.crop_size:
            raise ValueError(f"{label} crop_size={self.crop_size} exceeds image geometry {sizes['current_sr']}")
        if record.get("identity", False):
            current = self._load_rgb(record["current_sr"])
            hr = self._load_rgb(record["hr"])
            if not torch.equal(current, hr):
                raise ValueError(f"{label} identity=true requires exact current_sr=HR and an HR-conditioned proposal")

    @staticmethod
    def _load_rgb(path):
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        return torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32).div_(127.5).sub_(1.0)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        images = {"current": self._load_rgb(record["current_sr"]),
                  "proposal": self._load_rgb(record["proposal"]),
                  "hr": self._load_rgb(record["hr"])}
        original_lr = self._load_rgb(record["original_lr"])
        height, width = images["current"].shape[-2:]
        images["original_lr_up"] = F.interpolate(
            original_lr.unsqueeze(0), size=(height, width), mode="bicubic", align_corners=False,
        ).squeeze(0).clamp_(-1.0, 1.0)
        if self.crop_size is not None:
            top = self.rng.randrange(height - self.crop_size + 1)
            left = self.rng.randrange(width - self.crop_size + 1)
            images = {key: value[:, top:top + self.crop_size, left:left + self.crop_size]
                      for key, value in images.items()}
        if self.augment:
            flip_dims = []
            if self.rng.random() < 0.5:
                flip_dims.append(-1)
            if self.rng.random() < 0.5:
                flip_dims.append(-2)
            if flip_dims:
                images = {key: torch.flip(value, flip_dims) for key, value in images.items()}
        return {
            **{key: value.contiguous() for key, value in images.items()},
            "target_round": torch.tensor(record["target_round"], dtype=torch.long),
            "sample_key": record["sample_key"],
            "identity_mask": torch.tensor(record.get("identity", False), dtype=torch.bool),
        }
