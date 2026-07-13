import json
import logging
import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as tv_functional

from dataloaders.degradation_meta import DEGRADATION_KEYS
from models.prompt_builder import build_sr_prompt


logger = logging.getLogger(__name__)


class RGFluxSRJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        crop_size=512,
        scale=4,
        mode="train",
        use_prompt=True,
        use_suggestions=True,
        prompt_variant=None,
        use_degradation_vector=True,
        vae_align=16,
        max_retry=100,
        return_profile=False,
        mixed_crop_enabled=False,
        full_frame_ratio=0.0,
        full_frame_max_long_side=768,
        full_frame_align=32,
        full_frame_pad_mode="reflect",
        full_frame_upscale_small=False,
    ):
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.crop_size = int(crop_size)
        self.scale = int(scale)
        self.mode = mode
        self.use_prompt = bool(use_prompt)
        self.use_suggestions = bool(use_suggestions)
        self.prompt_variant = prompt_variant
        self.use_degradation_vector = bool(use_degradation_vector)
        self.vae_align = int(vae_align)
        self.max_retry = int(max_retry)
        self.return_profile = bool(return_profile)
        self.mixed_crop_enabled = bool(mixed_crop_enabled)
        self.full_frame_ratio = float(full_frame_ratio)
        self.full_frame_max_long_side = int(full_frame_max_long_side)
        self.full_frame_align = int(full_frame_align)
        self.full_frame_pad_mode = str(full_frame_pad_mode).strip().lower()
        self.full_frame_upscale_small = bool(full_frame_upscale_small)
        self.to_tensor = transforms.ToTensor()

        if self.vae_align > 1:
            self.crop_size = self.crop_size - (self.crop_size % self.vae_align)
        if self.crop_size <= 0:
            raise ValueError("crop_size must be positive after VAE alignment")
        if not 0.0 <= self.full_frame_ratio <= 1.0:
            raise ValueError("full_frame_ratio must be between 0 and 1")
        if self.full_frame_max_long_side <= 0:
            raise ValueError("full_frame_max_long_side must be positive")
        if self.full_frame_align <= 0:
            raise ValueError("full_frame_align must be positive")
        if self.full_frame_pad_mode not in {"constant", "edge", "reflect", "symmetric"}:
            raise ValueError(
                "full_frame_pad_mode must be one of: constant, edge, reflect, symmetric"
            )
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        self.records = self._load_records()
        if not self.records:
            raise RuntimeError(f"No valid RG-FLUX-SR records found in {self.jsonl_path}")

    def _load_records(self):
        records = []
        skipped = 0
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skip invalid JSON at %s:%s: %s", self.jsonl_path, line_no, exc)
                    skipped += 1
                    continue

                hq_path = payload.get("hq_path")
                lq_path = payload.get("lq_path")
                result = payload.get("result")
                unipercept_raw = payload.get("unipercept_raw")
                unipercept_raw = unipercept_raw if isinstance(unipercept_raw, dict) else {}
                profile = unipercept_raw.get("profile")
                if not hq_path or not lq_path or not isinstance(result, dict) or not isinstance(profile, dict):
                    skipped += 1
                    continue
                if not Path(hq_path).exists() or not Path(lq_path).exists():
                    logger.warning("Skip missing pair at line %s: hq=%s lq=%s", line_no, hq_path, lq_path)
                    skipped += 1
                    continue

                records.append(
                    {
                        "hq_path": str(hq_path),
                        "lq_path": str(lq_path),
                        "profile": profile,
                        "result": result,
                    }
                )
        if skipped:
            logger.warning("Skipped %d invalid RG-FLUX-SR JSONL records.", skipped)
        return records

    def __len__(self):
        return len(self.records)

    def _load_rgb(self, path):
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")

    def _ensure_min_size(self, image, min_size):
        if image.width >= min_size and image.height >= min_size:
            return image
        scale = min_size / max(min(image.width, image.height), 1)
        size = (max(round(image.width * scale), min_size), max(round(image.height * scale), min_size))
        return image.resize(size, Image.Resampling.BICUBIC)

    def _crop_pair(self, hq, lq):
        hq = self._ensure_min_size(hq, self.crop_size)

        max_x = hq.width - self.crop_size
        max_y = hq.height - self.crop_size
        if self.mode == "train":
            crop_x = random.randint(0, max_x) if max_x > 0 else 0
            crop_y = random.randint(0, max_y) if max_y > 0 else 0
        else:
            crop_x = max_x // 2
            crop_y = max_y // 2

        hq_crop = hq.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))

        ratio_x = hq.width / max(lq.width, 1)
        ratio_y = hq.height / max(lq.height, 1)
        same_resolution = abs(ratio_x - 1.0) < 0.05 and abs(ratio_y - 1.0) < 0.05
        if same_resolution:
            lq_crop = lq.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
        else:
            lq_x = int(round(crop_x / ratio_x))
            lq_y = int(round(crop_y / ratio_y))
            lq_w = max(1, int(round(self.crop_size / ratio_x)))
            lq_h = max(1, int(round(self.crop_size / ratio_y)))
            lq_x = min(max(lq_x, 0), max(lq.width - lq_w, 0))
            lq_y = min(max(lq_y, 0), max(lq.height - lq_h, 0))
            lq_crop = lq.crop((lq_x, lq_y, lq_x + lq_w, lq_y + lq_h))

        lq_up = lq_crop.resize((self.crop_size, self.crop_size), Image.Resampling.BICUBIC)
        return hq_crop, lq_crop, lq_up

    def _align_up(self, value):
        return int(math.ceil(max(int(value), 1) / self.full_frame_align) * self.full_frame_align)

    def _pad_to_size(self, image, width, height):
        pad_w = max(int(width) - image.width, 0)
        pad_h = max(int(height) - image.height, 0)
        if pad_w == 0 and pad_h == 0:
            return image
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
        padding_mode = self.full_frame_pad_mode
        if padding_mode == "reflect" and (
            padding[0] >= image.width
            or padding[2] >= image.width
            or padding[1] >= image.height
            or padding[3] >= image.height
        ):
            padding_mode = "edge"
        return tv_functional.pad(
            image,
            padding,
            fill=0,
            padding_mode=padding_mode,
        )

    def _full_frame_pair(self, hq, lq):
        source_width, source_height = hq.size
        source_long_side = max(source_width, source_height)
        should_resize = source_long_side > self.full_frame_max_long_side
        should_resize = should_resize or (
            self.full_frame_upscale_small and source_long_side < self.full_frame_max_long_side
        )
        resize_scale = (
            self.full_frame_max_long_side / max(source_long_side, 1)
            if should_resize
            else 1.0
        )
        content_width = max(1, int(round(source_width * resize_scale)))
        content_height = max(1, int(round(source_height * resize_scale)))

        if (content_width, content_height) != hq.size:
            hq_content = hq.resize((content_width, content_height), Image.Resampling.BICUBIC)
        else:
            hq_content = hq

        ratio_x = source_width / max(lq.width, 1)
        ratio_y = source_height / max(lq.height, 1)
        lq_content_width = max(1, int(round(content_width / max(ratio_x, 1e-8))))
        lq_content_height = max(1, int(round(content_height / max(ratio_y, 1e-8))))
        if (lq_content_width, lq_content_height) != lq.size:
            lq_content = lq.resize(
                (lq_content_width, lq_content_height),
                Image.Resampling.BICUBIC,
            )
        else:
            lq_content = lq
        lq_up_content = lq_content.resize(
            (content_width, content_height),
            Image.Resampling.BICUBIC,
        )

        aligned_width = self._align_up(content_width)
        aligned_height = self._align_up(content_height)
        hq_frame = self._pad_to_size(hq_content, aligned_width, aligned_height)
        lq_up = self._pad_to_size(lq_up_content, aligned_width, aligned_height)

        # The raw LQ tensor is used only as a downsample-consistency reference in
        # the current training path. Returning the aligned LQ-up image keeps that
        # reference spatially consistent with the padded full-frame target.
        lq_frame = lq_up.copy()
        return hq_frame, lq_frame, lq_up

    def _sample_pair(self, hq, lq):
        use_full_frame = (
            self.mode == "train"
            and self.mixed_crop_enabled
            and self.full_frame_ratio > 0.0
            and random.random() < self.full_frame_ratio
        )
        if use_full_frame:
            return (*self._full_frame_pair(hq, lq), "full_frame")
        return (*self._crop_pair(hq, lq), "local_crop")

    def _normalize_m11(self, image):
        return self.to_tensor(image).mul(2.0).sub(1.0)

    def _degradation_vector(self, result):
        vector = result.get("degradation_vector")
        vector = vector if isinstance(vector, dict) else {}
        if not self.use_degradation_vector:
            return torch.zeros(len(DEGRADATION_KEYS), dtype=torch.float32)
        return torch.tensor([float(vector.get(key, 0.0) or 0.0) for key in DEGRADATION_KEYS], dtype=torch.float32)

    def __getitem__(self, index):
        for retry in range(self.max_retry):
            record = self.records[(index + retry) % len(self.records)]
            try:
                hq = self._load_rgb(record["hq_path"])
                lq = self._load_rgb(record["lq_path"])
                hq_crop, lq_crop, lq_up, spatial_mode = self._sample_pair(hq, lq)
                profile = record["profile"]
                result = record["result"]
                sample = {
                    "hq": self._normalize_m11(hq_crop),
                    "lq": self._normalize_m11(lq_crop),
                    "lq_up": self._normalize_m11(lq_up),
                    "prompt": build_sr_prompt(
                        profile,
                        use_prompt=self.use_prompt,
                        use_suggestions=self.use_suggestions,
                        prompt_variant=self.prompt_variant,
                    ),
                    "degradation_vector": self._degradation_vector(result),
                    "score": torch.tensor(float(result.get("score", 0.0) or 0.0), dtype=torch.float32),
                    "suggestions": list(result.get("suggestions") or []),
                    "hq_path": record["hq_path"],
                    "lq_path": record["lq_path"],
                    "spatial_mode": spatial_mode,
                }
                if self.return_profile:
                    sample["profile"] = profile
                return sample
            except Exception as exc:
                if retry == 0:
                    logger.warning("Failed to load RG-FLUX-SR sample %s: %s", record, exc)
        raise RuntimeError(f"Failed to load sample after {self.max_retry} retries from {self.jsonl_path}")


def rg_flux_collate_fn(batch):
    tensor_keys = ["hq", "lq_up", "degradation_vector", "score"]
    collated = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    try:
        collated["lq"] = torch.stack([item["lq"] for item in batch], dim=0)
    except RuntimeError:
        collated["lq"] = [item["lq"] for item in batch]
    collated["prompt"] = [item["prompt"] for item in batch]
    collated["suggestions"] = [item["suggestions"] for item in batch]
    collated["hq_path"] = [item["hq_path"] for item in batch]
    collated["lq_path"] = [item["lq_path"] for item in batch]
    collated["spatial_mode"] = [item.get("spatial_mode", "local_crop") for item in batch]
    if all("profile" in item for item in batch):
        collated["profile"] = [item["profile"] for item in batch]
    return collated
