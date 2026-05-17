import json
import logging
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
        use_degradation_vector=True,
        vae_align=16,
        max_retry=100,
    ):
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.crop_size = int(crop_size)
        self.scale = int(scale)
        self.mode = mode
        self.use_prompt = bool(use_prompt)
        self.use_suggestions = bool(use_suggestions)
        self.use_degradation_vector = bool(use_degradation_vector)
        self.vae_align = int(vae_align)
        self.max_retry = int(max_retry)
        self.to_tensor = transforms.ToTensor()

        if self.vae_align > 1:
            self.crop_size = self.crop_size - (self.crop_size % self.vae_align)
        if self.crop_size <= 0:
            raise ValueError("crop_size must be positive after VAE alignment")
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
                if not hq_path or not lq_path or not isinstance(result, dict):
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
                hq_crop, lq_crop, lq_up = self._crop_pair(hq, lq)
                result = record["result"]
                return {
                    "hq": self._normalize_m11(hq_crop),
                    "lq": self._normalize_m11(lq_crop),
                    "lq_up": self._normalize_m11(lq_up),
                    "prompt": build_sr_prompt(
                        result,
                        use_prompt=self.use_prompt,
                        use_suggestions=self.use_suggestions,
                    ),
                    "degradation_vector": self._degradation_vector(result),
                    "score": torch.tensor(float(result.get("score", 0.0) or 0.0), dtype=torch.float32),
                    "suggestions": list(result.get("suggestions") or []),
                    "hq_path": record["hq_path"],
                    "lq_path": record["lq_path"],
                }
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
    return collated
