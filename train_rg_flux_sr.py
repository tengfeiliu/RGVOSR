import argparse
import copy
import csv
import datetime
import inspect
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
try:
    from accelerate.utils import GradientAccumulationPlugin
except ImportError:
    try:
        from accelerate.utils.dataclasses import GradientAccumulationPlugin
    except ImportError:
        GradientAccumulationPlugin = None
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from dataloaders.degradation_meta import DEGRADATION_KEYS
from dataloaders.rg_flux_jsonl_dataset import RGFluxSRJsonlDataset, rg_flux_collate_fn
from metrics.rg_sr_metrics import DEFAULT_OMGSR_METRICS, evaluate_dataset_dirs
from models.rg_flux_artist_factory import build_rg_flux_artist
from models.prompt_builder import build_sr_prompt, normalize_prompt_variant
from models.text_embedding_cache import (
    get_text_embedding_cache,
    normalize_text_encoding_mode,
    resolve_prompt_embeddings,
)
from rg_flux_fm import build_flow_matching_inputs, sample_multistep_fm, sample_sigma


logger = get_logger(__name__)
_LPIPS_LOSS_MODEL = None


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def cfg(config, path, default=None):
    current = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def cfg_bool(config, path, default=False):
    value = cfg(config, path, default)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off", "none", "null", ""}:
            return False
    return bool(value)


def resolve_prompt_schedule(config):
    enabled = cfg_bool(config, "condition.prompt_schedule.enabled", False)
    switch_step = int(cfg(config, "condition.prompt_schedule.switch_step", 0) or 0)
    if switch_step < 0:
        raise ValueError("condition.prompt_schedule.switch_step must be non-negative")

    before_variant = normalize_prompt_variant(
        cfg(config, "condition.prompt_schedule.before_variant", "fixed")
    )
    after_value = cfg(
        config,
        "condition.prompt_schedule.after_variant",
        cfg(config, "condition.prompt_variant", "fixed"),
    )
    after_variant = normalize_prompt_variant(after_value or "fixed")
    return {
        "enabled": enabled,
        "switch_step": switch_step,
        "before_variant": before_variant or "fixed",
        "after_variant": after_variant or "fixed",
    }


def prompt_variant_for_step(prompt_schedule, global_step):
    if not prompt_schedule.get("enabled", False):
        return None
    if int(global_step) < int(prompt_schedule["switch_step"]):
        return prompt_schedule["before_variant"]
    return prompt_schedule["after_variant"]


def resolve_batch_prompts(batch, config, global_step, prompt_schedule=None):
    prompt_schedule = prompt_schedule or resolve_prompt_schedule(config)
    active_variant = prompt_variant_for_step(prompt_schedule, global_step)
    if active_variant is None:
        return list(batch["prompt"]), None

    profiles = batch.get("profile")
    if not isinstance(profiles, list) or len(profiles) != len(batch["prompt"]):
        raise RuntimeError(
            "Prompt curriculum requires one cleaned profile per batch sample. "
            "Enable return_profile on RGFluxSRJsonlDataset."
        )
    prompts = [
        build_sr_prompt(profile, prompt_variant=active_variant)
        for profile in profiles
    ]
    return prompts, active_variant


def resolve_mixed_crop_config(config):
    enabled = cfg_bool(config, "data.mixed_crop.enabled", False)
    ratio = float(cfg(config, "data.mixed_crop.full_frame_ratio", 0.0) or 0.0)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("data.mixed_crop.full_frame_ratio must be between 0 and 1")
    max_long_side = int(cfg(config, "data.mixed_crop.full_frame_max_long_side", 768) or 0)
    align = int(cfg(config, "data.mixed_crop.full_frame_align", 32) or 0)
    if max_long_side <= 0:
        raise ValueError("data.mixed_crop.full_frame_max_long_side must be positive")
    if align <= 0:
        raise ValueError("data.mixed_crop.full_frame_align must be positive")
    return {
        "enabled": enabled,
        "full_frame_ratio": ratio,
        "full_frame_max_long_side": max_long_side,
        "full_frame_align": align,
        "full_frame_pad_mode": str(
            cfg(config, "data.mixed_crop.full_frame_pad_mode", "reflect") or "reflect"
        ).strip().lower(),
        "full_frame_upscale_small": cfg_bool(
            config,
            "data.mixed_crop.upscale_small_images",
            False,
        ),
    }


def normalize_loss_record_formats(formats):
    if formats is None:
        return ["jsonl", "csv"]
    if isinstance(formats, str):
        value = formats.strip()
        if value.lower() in {"", "none", "null", "false", "off", "no"}:
            return []
        formats = [part.strip() for part in value.split(",")]
    normalized = []
    for item in formats:
        value = str(item).strip().lower()
        if value in {"jsonl", "csv"} and value not in normalized:
            normalized.append(value)
    return normalized


class LossHistoryRecorder:
    def __init__(self, logging_dir, formats=("jsonl", "csv")):
        self.logging_dir = Path(logging_dir)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.formats = normalize_loss_record_formats(formats)
        self.jsonl_path = self.logging_dir / "loss_history.jsonl"
        self.csv_path = self.logging_dir / "loss_history.csv"
        self.summary_path = self.logging_dir / "loss_summary.json"
        self.plot_path = self.logging_dir / "loss_curves.png"
        self.csv_fieldnames = None
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                self.csv_fieldnames = next(reader, None)

    @staticmethod
    def _metric_value(value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    def _record(self, global_step, logs):
        record = {
            "global_step": int(global_step),
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        for key, value in logs.items():
            if key in {"global_step", "time"}:
                continue
            record[key] = self._metric_value(value)
        return record

    def _append_jsonl(self, record):
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_csv(self, record):
        if self.csv_fieldnames is None:
            self.csv_fieldnames = ["global_step", "time"] + [
                key for key in record.keys() if key not in {"global_step", "time"}
            ]
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.csv_fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def _write_summary(self, record):
        summary = {}
        if self.summary_path.exists():
            try:
                summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}
        summary["last_step"] = int(record["global_step"])
        summary["last"] = record
        for key in ("loss_total", "loss_fm"):
            if key not in record:
                continue
            min_key = f"min_{key}"
            previous = summary.get(min_key)
            previous_value = previous.get(key) if isinstance(previous, dict) else None
            if previous_value is None or float(record[key]) <= float(previous_value):
                summary[min_key] = record
        self.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _as_float(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if value != value or value in {float("inf"), float("-inf")}:
            return None
        return value

    def _load_records_for_plot(self):
        records = []
        if self.jsonl_path.exists():
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(record)
        elif self.csv_path.exists():
            with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
                records.extend(csv.DictReader(handle))
        return records

    @staticmethod
    def _loss_keys_for_plot(records):
        preferred = [
            "loss_total",
            "loss_fm",
            "loss_latent",
            "loss_charb",
            "loss_lpips",
            "loss_down",
            "loss_div",
            "loss_entropy",
            "loss_balance",
        ]
        keys = []
        all_keys = []
        for record in records:
            all_keys.extend(record.keys())
        for key in preferred:
            if key in all_keys and key not in keys:
                keys.append(key)
        for key in all_keys:
            if key.startswith("loss_") and key != "loss_lpips_weight" and key not in keys:
                keys.append(key)
        if "loss_total" not in keys and "loss" in all_keys:
            keys.insert(0, "loss")
        return keys

    def _plot_snapshot_path(self, step):
        return self.logging_dir / f"loss_curves_step-{int(step):08d}.png"

    def write_plot(self, step=None):
        from PIL import Image, ImageDraw, ImageFont

        records = self._load_records_for_plot()
        width, height = 1200, 720
        left, top, right, bottom = 90, 55, 930, 625
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        title = "RG-FLUX-SR training loss curves"
        draw.text((left, 20), title, fill=(0, 0, 0), font=font)
        draw.text((width - 245, 20), "loss vs. step", fill=(80, 80, 80), font=font)

        series = {}
        steps = []
        for record in records:
            step_value = self._as_float(record.get("global_step"))
            if step_value is None:
                continue
            steps.append(step_value)
        keys = self._loss_keys_for_plot(records)
        for key in keys:
            values = []
            for record in records:
                step_value = self._as_float(record.get("global_step"))
                loss_value = self._as_float(record.get(key))
                if step_value is not None and loss_value is not None:
                    values.append((step_value, loss_value))
            if values:
                series[key] = values

        draw.rectangle((left, top, right, bottom), outline=(30, 30, 30), width=1)
        if not steps or not series:
            draw.text((left + 20, top + 20), "No numeric loss records yet.", fill=(90, 90, 90), font=font)
        else:
            x_min, x_max = min(steps), max(steps)
            all_values = [value for values in series.values() for _, value in values]
            y_min, y_max = min(all_values), max(all_values)
            if y_min >= 0:
                y_min = 0.0
            if x_min == x_max:
                x_max = x_min + 1.0
            if y_min == y_max:
                pad = abs(y_min) * 0.1 if y_min else 1.0
                y_min -= pad
                y_max += pad

            def x_pos(value):
                return left + (float(value) - x_min) / (x_max - x_min) * (right - left)

            def y_pos(value):
                return bottom - (float(value) - y_min) / (y_max - y_min) * (bottom - top)

            for i in range(6):
                ratio = i / 5.0
                y = top + ratio * (bottom - top)
                value = y_max - ratio * (y_max - y_min)
                draw.line((left, y, right, y), fill=(230, 230, 230), width=1)
                draw.text((10, y - 6), f"{value:.4g}", fill=(70, 70, 70), font=font)
            for i in range(6):
                ratio = i / 5.0
                x = left + ratio * (right - left)
                value = x_min + ratio * (x_max - x_min)
                draw.line((x, top, x, bottom), fill=(242, 242, 242), width=1)
                draw.text((x - 16, bottom + 12), f"{int(round(value))}", fill=(70, 70, 70), font=font)

            palette = [
                (31, 119, 180),
                (255, 127, 14),
                (44, 160, 44),
                (214, 39, 40),
                (148, 103, 189),
                (140, 86, 75),
                (227, 119, 194),
                (127, 127, 127),
                (188, 189, 34),
                (23, 190, 207),
            ]
            legend_x, legend_y = right + 25, top
            for index, (key, values) in enumerate(series.items()):
                color = palette[index % len(palette)]
                points = [(x_pos(step_value), y_pos(loss_value)) for step_value, loss_value in values]
                if len(points) == 1:
                    x, y = points[0]
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
                else:
                    draw.line(points, fill=color, width=2)
                draw.rectangle(
                    (legend_x, legend_y + index * 22 + 3, legend_x + 12, legend_y + index * 22 + 15),
                    fill=color,
                )
                draw.text((legend_x + 18, legend_y + index * 22), key, fill=(0, 0, 0), font=font)

            draw.text(((left + right) // 2 - 20, height - 45), "step", fill=(0, 0, 0), font=font)
            draw.text((12, top - 25), "loss", fill=(0, 0, 0), font=font)

        image.save(self.plot_path)
        if step is not None:
            image.save(self._plot_snapshot_path(step))
        return self.plot_path

    def append(self, global_step, logs):
        if not self.formats:
            return
        record = self._record(global_step, logs)
        if "jsonl" in self.formats:
            self._append_jsonl(record)
        if "csv" in self.formats:
            self._append_csv(record)
        self._write_summary(record)


def charbonnier_loss(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


def warmup_weight(global_step, start, end, max_weight):
    max_weight = float(max_weight)
    if max_weight <= 0:
        return 0.0
    start = int(start)
    end = int(end)
    if global_step < start:
        return 0.0
    if end <= start:
        return max_weight
    if global_step >= end:
        return max_weight
    return max_weight * float(global_step - start) / float(end - start)


def should_compute_every(global_step, every):
    every = int(every)
    if every <= 1:
        return True
    return int(global_step) % every == 0


def _interpolate_image(tensor, size=None, scale_factor=None, mode="area"):
    kwargs = {"mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor, size=size, scale_factor=scale_factor, **kwargs)


def _get_lpips_loss_model(device):
    global _LPIPS_LOSS_MODEL
    if _LPIPS_LOSS_MODEL is None:
        try:
            import lpips
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "LPIPS loss requested but lpips package is not installed. "
                "Set loss.lpips_weight=0 or install lpips."
            ) from exc
        model = lpips.LPIPS(net="alex")
        model.requires_grad_(False)
        model.eval()
        _LPIPS_LOSS_MODEL = model
    return _LPIPS_LOSS_MODEL.to(device)


def _resize_hq_for_loss(hq, sr_pred):
    hq = hq.to(device=sr_pred.device)
    if hq.shape[-2:] == sr_pred.shape[-2:]:
        return hq
    if not getattr(_resize_hq_for_loss, "_warned", False):
        logger.warning(
            "HQ tensor size %s does not match decoded SR size %s; resizing HQ for image-space loss.",
            tuple(hq.shape[-2:]),
            tuple(sr_pred.shape[-2:]),
        )
        setattr(_resize_hq_for_loss, "_warned", True)
    return _interpolate_image(hq.float(), size=sr_pred.shape[-2:], mode="bilinear")


def _downsample_reference(batch, lq_up, sr_down, scale, down_mode):
    lq = batch.get("lq") if isinstance(batch, dict) else None
    if torch.is_tensor(lq) and lq.ndim == 4:
        lr_ref = lq.to(device=sr_down.device, dtype=torch.float32)
        if lr_ref.shape[-2:] != sr_down.shape[-2:]:
            lr_ref = _interpolate_image(lr_ref, size=sr_down.shape[-2:], mode=down_mode)
        return lr_ref
    return _interpolate_image(
        lq_up.to(device=sr_down.device, dtype=torch.float32),
        scale_factor=1.0 / max(int(scale), 1),
        mode=down_mode,
    )


def crop_image_loss_inputs(z0_pred, hq, lq_up, config):
    crop_size = int(cfg(config, "loss.image_loss_crop_size", 0) or 0)
    if crop_size <= 0:
        return z0_pred, hq, lq_up, False
    image_height, image_width = hq.shape[-2:]
    if crop_size >= min(image_height, image_width):
        return z0_pred, hq, lq_up, False

    packed_height, packed_width = z0_pred.shape[-2:]
    token_scale_h = max(image_height // max(packed_height, 1), 1)
    token_scale_w = max(image_width // max(packed_width, 1), 1)
    token_crop_h = max(1, min(packed_height, crop_size // token_scale_h))
    token_crop_w = max(1, min(packed_width, crop_size // token_scale_w))
    max_top = packed_height - token_crop_h
    max_left = packed_width - token_crop_w
    if max_top <= 0 and max_left <= 0:
        return z0_pred, hq, lq_up, False

    if z0_pred.device.type == "cuda":
        top = int(torch.randint(max_top + 1, (1,), device=z0_pred.device).item()) if max_top > 0 else 0
        left = int(torch.randint(max_left + 1, (1,), device=z0_pred.device).item()) if max_left > 0 else 0
    else:
        top = int(torch.randint(max_top + 1, (1,)).item()) if max_top > 0 else 0
        left = int(torch.randint(max_left + 1, (1,)).item()) if max_left > 0 else 0

    image_top = top * token_scale_h
    image_left = left * token_scale_w
    image_crop_h = token_crop_h * token_scale_h
    image_crop_w = token_crop_w * token_scale_w
    z0_crop = z0_pred[..., top : top + token_crop_h, left : left + token_crop_w]
    hq_crop = hq[..., image_top : image_top + image_crop_h, image_left : image_left + image_crop_w]
    lq_up_crop = lq_up[..., image_top : image_top + image_crop_h, image_left : image_left + image_crop_w]
    return z0_crop, hq_crop, lq_up_crop, True


def compute_stage0b_supervised_losses(
    artist,
    config,
    global_step,
    loss_fm,
    z_t,
    v_pred,
    sigma,
    z_hr,
    hq,
    lq_up,
    batch=None,
):
    zero = loss_fm.new_zeros(())
    latent_weight = float(cfg(config, "loss.latent_weight", 0.0))
    charb_weight = float(cfg(config, "loss.charb_weight", 0.0))
    down_weight = float(cfg(config, "loss.down_weight", 0.0))
    lpips_max_weight = float(cfg(config, "loss.lpips_weight", 0.0))
    image_loss_every = int(cfg(config, "loss.image_loss_every", 1))
    lpips_every = int(cfg(config, "loss.lpips_every", 1))
    lpips_weight_now = warmup_weight(
        global_step,
        start=int(cfg(config, "loss.lpips_warmup_start", 2000)),
        end=int(cfg(config, "loss.lpips_warmup_end", 6000)),
        max_weight=lpips_max_weight,
    )
    if not should_compute_every(global_step, lpips_every):
        lpips_weight_now = 0.0

    sigma_view = sigma.reshape(-1, *([1] * (z_t.ndim - 1))).to(device=z_t.device, dtype=z_t.dtype)
    z0_pred = z_t - sigma_view * v_pred

    loss_latent = zero
    if latent_weight > 0:
        loss_latent = charbonnier_loss(z0_pred.float(), z_hr.float())

    image_loss_due = should_compute_every(global_step, image_loss_every)
    needs_image_loss = image_loss_due and (charb_weight > 0 or down_weight > 0)
    needs_lpips = lpips_weight_now > 0
    sr_pred = None
    loss_charb = zero
    loss_down = zero
    loss_lpips = zero
    if needs_image_loss or needs_lpips:
        z0_for_image, hq_for_image, lq_up_for_image, crop_lq_ref = crop_image_loss_inputs(
            z0_pred,
            hq,
            lq_up,
            config,
        )
        sr_pred = artist.decode_latents_for_loss(z0_for_image)
        hq_for_loss = _resize_hq_for_loss(hq_for_image, sr_pred)
        if image_loss_due and charb_weight > 0:
            loss_charb = charbonnier_loss(sr_pred.float(), hq_for_loss.float())
        if image_loss_due and down_weight > 0:
            scale = int(cfg(config, "data.scale", 4))
            down_mode = str(cfg(config, "loss.down_mode", "area"))
            sr_down = _interpolate_image(sr_pred.float(), scale_factor=1.0 / max(scale, 1), mode=down_mode)
            lr_ref_batch = {} if crop_lq_ref else (batch or {})
            lr_ref = _downsample_reference(lr_ref_batch, lq_up_for_image, sr_down, scale, down_mode)
            loss_down = charbonnier_loss(sr_down, lr_ref)
        if needs_lpips:
            lpips_resize = cfg(config, "loss.lpips_resize", 256)
            if lpips_resize is not None and int(lpips_resize) > 0:
                lpips_size = int(lpips_resize)
                sr_lpips = _interpolate_image(sr_pred.float(), size=(lpips_size, lpips_size), mode="bilinear")
                hq_lpips = _interpolate_image(hq_for_loss.float(), size=(lpips_size, lpips_size), mode="bilinear")
            else:
                sr_lpips = sr_pred.float()
                hq_lpips = hq_for_loss.float()
            lpips_model = _get_lpips_loss_model(sr_lpips.device)
            loss_lpips = lpips_model(sr_lpips, hq_lpips).mean()

    return {
        "z0_pred": z0_pred,
        "loss_latent": loss_latent.to(device=loss_fm.device),
        "loss_charb": loss_charb.to(device=loss_fm.device),
        "loss_down": loss_down.to(device=loss_fm.device),
        "loss_lpips": loss_lpips.to(device=loss_fm.device),
        "loss_lpips_weight": float(lpips_weight_now),
    }


def normalize_report_to(report_to):
    if report_to is None:
        return None
    if isinstance(report_to, str):
        value = report_to.strip()
        if value.lower() in {"", "none", "null", "false", "off", "no"}:
            return None
        return value
    return report_to


def create_logger(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(logging_dir, "log.txt"))],
    )
    return logging.getLogger(__name__)


def weight_dtype_from_accelerator(accelerator):
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def create_gradient_accumulation_plugin(num_steps):
    if GradientAccumulationPlugin is None:
        return None, False
    try:
        parameters = inspect.signature(GradientAccumulationPlugin).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "sync_each_batch" in parameters:
        return GradientAccumulationPlugin(num_steps=num_steps, sync_each_batch=True), True
    return GradientAccumulationPlugin(num_steps=num_steps), False


def deepspeed_zero_stage(ds_config):
    if not isinstance(ds_config, dict):
        return 0
    zero_optimization = ds_config.get("zero_optimization")
    if isinstance(zero_optimization, dict):
        value = zero_optimization.get("stage", 0)
    else:
        value = ds_config.get("zero_stage", zero_optimization or 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def get_deepspeed_config(accelerator):
    plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    if plugin is None:
        return None
    ds_config = getattr(plugin, "deepspeed_config", None)
    if hasattr(ds_config, "config"):
        ds_config = ds_config.config
    if not isinstance(ds_config, dict):
        return None
    return ds_config


def _deepspeed_auto_or_missing(value):
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "auto"})


def _deepspeed_int(value, default):
    if _deepspeed_auto_or_missing(value):
        return int(default)
    return int(value)


def resolve_hf_zero3_config(
    ds_config,
    per_device_batch,
    grad_accum_steps,
    num_processes,
    force_training_batch=False,
):
    resolved = copy.deepcopy(ds_config)
    micro_key = "train_micro_batch_size_per_gpu"
    accum_key = "gradient_accumulation_steps"
    train_key = "train_batch_size"

    if force_training_batch:
        micro = int(per_device_batch)
        accum = int(grad_accum_steps)
        train_batch = micro * accum * int(num_processes)
    else:
        micro = _deepspeed_int(resolved.get(micro_key), per_device_batch)
        accum = _deepspeed_int(resolved.get(accum_key), grad_accum_steps)
        train_batch = _deepspeed_int(resolved.get(train_key), micro * accum * int(num_processes))

    resolved[micro_key] = micro
    resolved[accum_key] = accum
    resolved[train_key] = train_batch
    return resolved


def _normalize_offload_device(device):
    if device is None:
        return None
    return str(device).strip().lower()


def get_deepspeed_optimizer_offload_device(ds_config):
    if not isinstance(ds_config, dict):
        return None
    zero_optimization = ds_config.get("zero_optimization")
    if isinstance(zero_optimization, dict):
        offload_optimizer = zero_optimization.get("offload_optimizer")
        if isinstance(offload_optimizer, dict) and "device" in offload_optimizer:
            return _normalize_offload_device(offload_optimizer.get("device"))
    return _normalize_offload_device(ds_config.get("offload_optimizer_device"))


def set_deepspeed_optimizer_offload_device(ds_config, device):
    if not isinstance(ds_config, dict):
        return ds_config
    normalized = _normalize_offload_device(device)
    if normalized is None:
        return ds_config
    disabled = normalized in {"", "none", "false", "no", "off"}
    ds_config["offload_optimizer_device"] = "none" if disabled else normalized
    zero_optimization = ds_config.get("zero_optimization")
    if isinstance(zero_optimization, dict):
        if disabled:
            zero_optimization.pop("offload_optimizer", None)
        else:
            offload_optimizer = zero_optimization.get("offload_optimizer")
            if not isinstance(offload_optimizer, dict):
                offload_optimizer = {}
                zero_optimization["offload_optimizer"] = offload_optimizer
            offload_optimizer["device"] = normalized
    return ds_config


def sync_deepspeed_config_for_training(
    ds_config,
    per_device_batch,
    grad_accum_steps,
    num_processes,
    optimizer_offload_device=None,
):
    if not isinstance(ds_config, dict):
        return None
    resolved = resolve_hf_zero3_config(
        ds_config,
        per_device_batch=per_device_batch,
        grad_accum_steps=grad_accum_steps,
        num_processes=num_processes,
        force_training_batch=True,
    )
    for key in ("train_micro_batch_size_per_gpu", "gradient_accumulation_steps", "train_batch_size"):
        ds_config[key] = resolved[key]
    if optimizer_offload_device is not None:
        set_deepspeed_optimizer_offload_device(ds_config, optimizer_offload_device)
    return ds_config


def sync_deepspeed_plugin_for_training(plugin, grad_accum_steps, optimizer_offload_device=None):
    if plugin is None:
        return
    for attr in ("gradient_accumulation_steps",):
        if hasattr(plugin, attr):
            try:
                setattr(plugin, attr, int(grad_accum_steps))
            except (AttributeError, TypeError, ValueError):
                pass
    if optimizer_offload_device is not None:
        normalized = _normalize_offload_device(optimizer_offload_device)
        for attr in ("offload_optimizer_device",):
            if hasattr(plugin, attr):
                try:
                    setattr(plugin, attr, normalized)
                except (AttributeError, TypeError):
                    pass


def find_latest_checkpoint(output_dir, resume_ckpt=None):
    if resume_ckpt:
        path = Path(resume_ckpt)
        if path.exists():
            return path
        print(f"Warning: resume checkpoint does not exist: {resume_ckpt}")
    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("checkpoint-*"))
    return candidates[-1] if candidates else None


def resolve_resume_checkpoint(output_dir, resume_ckpt=None, auto_resume=True):
    if resume_ckpt:
        return find_latest_checkpoint(output_dir, resume_ckpt)
    if not auto_resume:
        return None
    return find_latest_checkpoint(output_dir, None)


def training_state_rank_path(checkpoint_dir, rank):
    return Path(checkpoint_dir) / f"training_state_rank-{int(rank):05d}.pt"


def _optimizer_requires_rank_state_dict(optimizer):
    inner_optimizer = getattr(optimizer, "optimizer", optimizer)
    return hasattr(inner_optimizer, "dp_process_group") and hasattr(inner_optimizer, "_rigid_load_state_dict")


def _is_rank_keyed_optimizer_state(optimizer_state, rank):
    if isinstance(optimizer_state, dict):
        return int(rank) in optimizer_state
    if isinstance(optimizer_state, (list, tuple)):
        return int(rank) < len(optimizer_state)
    return False


def save_rg_checkpoint(accelerator, artist, optimizer, lr_scheduler, checkpoint_dir, global_step):
    accelerator.wait_for_everyone()
    checkpoint_dir = Path(checkpoint_dir)
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(artist)
    unwrapped.save_trainable(checkpoint_dir / "rg_flux_adapters", save_files=accelerator.is_main_process)
    accelerator.wait_for_everyone()
    rank = int(accelerator.process_index)
    optimizer_state = optimizer.state_dict()
    if _optimizer_requires_rank_state_dict(optimizer):
        optimizer_state = {rank: optimizer_state}
    torch.save(
        {
            "global_step": global_step,
            "optimizer": optimizer_state,
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        training_state_rank_path(checkpoint_dir, rank),
    )
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    torch.save(
        {
            "global_step": global_step,
            "num_processes": int(accelerator.num_processes),
            "rank_state_pattern": "training_state_rank-{rank:05d}.pt",
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        checkpoint_dir / "training_state.pt",
    )


def load_training_state(state_path):
    try:
        return torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(state_path, map_location="cpu")


def load_rg_checkpoint(accelerator, artist, optimizer, lr_scheduler, checkpoint_dir, resume_training_state=True):
    checkpoint_dir = Path(checkpoint_dir)
    adapter_dir = checkpoint_dir / "rg_flux_adapters"
    unwrapped = accelerator.unwrap_model(artist)
    unwrapped.load_trainable(adapter_dir if adapter_dir.exists() else checkpoint_dir, is_trainable=True)

    rank = int(accelerator.process_index)
    rank_state_path = training_state_rank_path(checkpoint_dir, rank)
    state_path = rank_state_path if rank_state_path.exists() else checkpoint_dir / "training_state.pt"
    global_step = 0
    if state_path.exists():
        state = load_training_state(state_path)
        global_step = int(state.get("global_step", 0))
        if resume_training_state:
            optimizer_state = state.get("optimizer", {})
            if _optimizer_requires_rank_state_dict(optimizer) and not _is_rank_keyed_optimizer_state(optimizer_state, rank):
                raise RuntimeError(
                    f"Checkpoint {checkpoint_dir} was saved without rank-local ZeRO-3 optimizer state for rank {rank}. "
                    "A full optimizer/scheduler resume is not reliable from this checkpoint. "
                    "Use a checkpoint saved with rank-local training_state_rank-*.pt files, or set "
                    "training.resume_training_state: false to explicitly load model/adapters only with a fresh optimizer."
                )
            optimizer.load_state_dict(optimizer_state)
            lr_scheduler.load_state_dict(state.get("lr_scheduler", {}))
    return global_step


def make_experiment_name(config):
    suffix = cfg(config, "training.suffix", "")
    lr_mode = cfg(config, "condition.lr_cond_mode", "latent_adapter")
    stage = cfg(config, "training.stage", "A")
    crop = cfg(config, "data.crop_size", 512)
    backend = str(cfg(config, "model.flux_backend", "flux1") or "flux1").lower()
    if backend in {"flux2_klein", "flux2-klein", "flux_2_klein"}:
        return f"rg_flux2_klein_sr_ms_stage{stage}_{lr_mode}_size{crop}{suffix}"
    return f"rg_flux_sr_ms_stage{stage}_{lr_mode}_size{crop}{suffix}"


def format_run_id(now=None):
    now = now or datetime.datetime.now()
    return now.strftime("%y%m%d%H")


def resolve_experiment_name(config, output_root=None, now=None):
    explicit_name = cfg(config, "training.exp_name", None)
    if explicit_name:
        return str(explicit_name), None

    base_name = make_experiment_name(config)
    if not cfg_bool(config, "training.add_datetime_suffix", True):
        return base_name, None

    run_id = str(cfg(config, "training.run_id", None) or format_run_id(now))
    exp_name = f"{base_name}_{run_id}"
    if output_root is None:
        return exp_name, run_id

    output_root = Path(output_root)
    candidate = exp_name
    retry_index = 2
    while (output_root / candidate).exists():
        candidate = f"{exp_name}_r{retry_index:02d}"
        retry_index += 1
    return candidate, run_id


def load_evaluation_records(jsonl_path, num_samples):
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Evaluation JSONL file not found: {jsonl_path}")
    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            lq_path = record.get("lq_path")
            result = record.get("result")
            unipercept_raw = record.get("unipercept_raw")
            unipercept_raw = unipercept_raw if isinstance(unipercept_raw, dict) else {}
            profile = unipercept_raw.get("profile")
            if not lq_path or not Path(lq_path).exists() or not isinstance(result, dict) or not isinstance(profile, dict):
                continue
            records.append(record)
            if len(records) >= num_samples:
                break
    if not records:
        raise RuntimeError(f"No valid evaluation records found in {jsonl_path}")
    return records


def prepare_eval_lq_up(image_path, upscale, align):
    image = Image.open(image_path).convert("RGB")
    if upscale > 1:
        image = image.resize((image.width * upscale, image.height * upscale), Image.Resampling.BICUBIC)
    width = max(align, image.width - image.width % align)
    height = max(align, image.height - image.height % align)
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return to_tensor(image).unsqueeze(0).mul(2.0).sub(1.0)


def degradation_tensor_from_result(result, device, dtype, use_degradation_vector=True):
    vector = result.get("degradation_vector") if isinstance(result, dict) else {}
    vector = vector if isinstance(vector, dict) and use_degradation_vector else {}
    values = [float(vector.get(key, 0.0) or 0.0) for key in DEGRADATION_KEYS]
    return torch.tensor(values, device=device, dtype=dtype).unsqueeze(0)


def fork_rng_for_device(device):
    if getattr(device, "type", None) == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return torch.random.fork_rng(devices=[index])
    return torch.random.fork_rng(devices=[])


def evaluation_logs_from_summary(summary_json):
    logs = {}
    for row in summary_json.get("summary", []):
        if row.get("dataset") != "eval":
            continue
        metric = row["metric"]
        logs[f"eval/{metric}"] = float(row["mean"])  # Logs use eval/<metric> keys.
    return logs


def run_rg_flux_evaluation(
    accelerator,
    artist,
    config,
    exp_name,
    global_step,
    weight_dtype,
    text_embedding_cache=None,
    local_logger=None,
):
    if not bool(cfg(config, "evaluation.enabled", False)):
        return None
    eval_every = int(cfg(config, "evaluation.eval_every", 500))
    if eval_every <= 0 or global_step <= 0 or global_step % eval_every != 0:
        return None

    eval_jsonl = cfg(config, "evaluation.jsonl_path", None) or cfg(config, "data.jsonl_path")
    records = load_evaluation_records(eval_jsonl, int(cfg(config, "evaluation.num_samples", 8)))
    eval_root = Path(cfg(config, "evaluation.output_dir", "eval")) / exp_name / f"step-{global_step:08d}"
    image_dir = eval_root / "images"
    metrics_dir = eval_root / "metrics"
    if accelerator.is_main_process:
        image_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if local_logger is not None:
            local_logger.info("Running RG-FLUX-SR evaluation at step %s on %s samples", global_step, len(records))

    unwrapped_artist = accelerator.unwrap_model(artist)
    was_training = artist.training
    artist.eval()
    to_pil = transforms.ToPILImage()
    lr_cond_mode = cfg(config, "condition.lr_cond_mode", "latent_adapter")
    use_degradation_vector = bool(cfg(config, "condition.use_degradation_vector", True))
    num_inference_steps = int(cfg(config, "evaluation.num_inference_steps", cfg(config, "flow_matching.num_inference_steps", 25)))
    eval_seed = int(cfg(config, "evaluation.seed", cfg(config, "training.seed", 42) or 42))

    try:
        with torch.no_grad():
            for sample_index, record in enumerate(records):
                result = record.get("result") if isinstance(record.get("result"), dict) else {}
                unipercept_raw = record.get("unipercept_raw")
                unipercept_raw = unipercept_raw if isinstance(unipercept_raw, dict) else {}
                profile = unipercept_raw.get("profile") if isinstance(unipercept_raw.get("profile"), dict) else {}
                prompt = build_sr_prompt(
                    profile,
                    use_prompt=bool(cfg(config, "condition.use_prompt", True)),
                    use_suggestions=bool(cfg(config, "condition.use_suggestions", True)),
                    prompt_variant=cfg(config, "condition.prompt_variant", None),
                )
                lq_up = prepare_eval_lq_up(
                    record["lq_path"],
                    upscale=int(cfg(config, "data.scale", 4)),
                    align=int(cfg(config, "data.vae_align", 16)),
                ).to(accelerator.device, dtype=weight_dtype)
                z_lr = unwrapped_artist.encode_images(
                    lq_up,
                    sample=lr_cond_mode != "flux2_image_concat",
                ).to(accelerator.device, dtype=weight_dtype)
                prompt_embeds, pooled_prompt_embeds, text_ids = resolve_prompt_embeddings(
                    artist=unwrapped_artist,
                    prompts=[prompt],
                    image_keys=[record["lq_path"]],
                    config=config,
                    device=accelerator.device,
                    dtype=weight_dtype,
                    cache=text_embedding_cache,
                )
                degradation_vector = degradation_tensor_from_result(
                    result,
                    accelerator.device,
                    weight_dtype,
                    use_degradation_vector=use_degradation_vector,
                )
                dino_tokens = unwrapped_artist.extract_visual_tokens(lq_up)
                with fork_rng_for_device(accelerator.device):
                    torch.manual_seed(eval_seed + sample_index)
                    if accelerator.device.type == "cuda":
                        torch.cuda.manual_seed_all(eval_seed + sample_index)
                    sr_latent = sample_multistep_fm(
                        artist=artist,
                        shape=tuple(z_lr.shape),
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        degradation_vector=degradation_vector,
                        z_lr=z_lr,
                        dino_tokens=dino_tokens,
                        lr_cond_mode=lr_cond_mode,
                        num_steps=num_inference_steps,
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                if accelerator.is_main_process:
                    sr = unwrapped_artist.decode_latents(sr_latent).clamp(-1, 1).add(1.0).mul(0.5).clamp(0, 1)
                    to_pil(sr[0].float().cpu()).save(image_dir / f"{sample_index:04d}_{Path(record['lq_path']).stem}.png")
                accelerator.wait_for_everyone()
    finally:
        if was_training:
            artist.train()

    summary_json = None
    if accelerator.is_main_process:
        metrics = cfg(config, "evaluation.metrics", DEFAULT_OMGSR_METRICS) or DEFAULT_OMGSR_METRICS
        metric_device = cfg(config, "evaluation.device", "cpu")
        summary_json = evaluate_dataset_dirs(
            {"eval": image_dir},
            output_dir=metrics_dir,
            metrics=metrics,
            device=metric_device,
        )
        if local_logger is not None:
            for row in summary_json["summary"]:
                metric = row["metric"]
                direction = summary_json["metric_directions"].get(metric, "")
                local_logger.info(
                    "[Eval @ step %s] %s (%s): %.6f",
                    global_step,
                    metric,
                    direction,
                    float(row["mean"]),
                )
    accelerator.wait_for_everyone()
    return summary_json


def main(config_path, dry_run=False):
    config = load_config(config_path)
    config.setdefault("training", {})
    config.setdefault("model", {})
    config.setdefault("data", {})
    config.setdefault("condition", {})
    config.setdefault("evaluation", {})
    config.setdefault("text_encoding", {})
    prompt_schedule = resolve_prompt_schedule(config)
    mixed_crop = resolve_mixed_crop_config(config)

    output_root = Path(cfg(config, "training.output_dir", "exp_rg_flux_sr"))
    report_to = normalize_report_to(cfg(config, "training.report_to", None))
    exp_name, resolved_run_id = resolve_experiment_name(config, output_root=output_root)
    config["training"]["resolved_exp_name"] = exp_name
    if resolved_run_id is None:
        config["training"].pop("resolved_run_id", None)
    else:
        config["training"]["resolved_run_id"] = resolved_run_id
    output_dir = output_root / exp_name
    logging_dir = output_dir / cfg(config, "training.logging_dir", "logs")
    per_device_batch = int(cfg(config, "data.batch_size", 1))
    if mixed_crop["enabled"] and mixed_crop["full_frame_ratio"] > 0.0 and per_device_batch != 1:
        raise ValueError(
            "Mixed local/full-frame training currently requires data.batch_size: 1 "
            "because full-frame samples have variable aligned resolutions."
        )
    grad_accum = int(cfg(config, "training.grad_accum_steps", 1))
    gradient_accumulation_plugin, supports_sync_each_batch = create_gradient_accumulation_plugin(grad_accum)

    accelerator_project_config = ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        mixed_precision=str(cfg(config, "model.dtype", "bf16")),
        log_with=report_to,
        project_config=accelerator_project_config,
    )

    local_logger = logger
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        with (output_dir / "args.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        local_logger = create_logger(logging_dir)
        local_logger.info("Experiment directory created at %s", output_dir)
        effective_batch = per_device_batch * accelerator.num_processes * grad_accum
        local_logger.info("===========> RG-FLUX-SR-MS Batch Size Debug Info:")
        local_logger.info("  accelerator.num_processes = %s", accelerator.num_processes)
        local_logger.info("  data.batch_size per device = %s", per_device_batch)
        local_logger.info("  training.grad_accum_steps = %s", grad_accum)
        local_logger.info("  effective global batch = %s", effective_batch)
        local_logger.info("  text_encoder_device = %s", cfg(config, "model.text_encoder_device", "cpu"))
        local_logger.info("  text_encoding.mode = %s", normalize_text_encoding_mode(config))
        local_logger.info("  text_encoding.cache_dir = %s", cfg(config, "text_encoding.cache_dir", None))
        local_logger.info("  vae_device = %s", cfg(config, "model.vae_device", "cpu"))
        local_logger.info("  max_prompt_sequence_length = %s", cfg(config, "model.max_prompt_sequence_length", 128))
        local_logger.info(
            "  prompt_schedule = enabled:%s switch_step:%s before:%s after:%s",
            prompt_schedule["enabled"],
            prompt_schedule["switch_step"],
            prompt_schedule["before_variant"],
            prompt_schedule["after_variant"],
        )
        local_logger.info(
            "  mixed_crop = enabled:%s local_ratio:%.3f full_frame_ratio:%.3f max_long_side:%s align:%s",
            mixed_crop["enabled"],
            1.0 - mixed_crop["full_frame_ratio"],
            mixed_crop["full_frame_ratio"],
            mixed_crop["full_frame_max_long_side"],
            mixed_crop["full_frame_align"],
        )
        crop_size = int(cfg(config, "data.crop_size", 512))
        vae_scale_factor = 8
        latent_size = crop_size // vae_scale_factor
        packed_image_tokens = (latent_size // 2) * (latent_size // 2)
        local_logger.info("===========> RG-FLUX-SR-MS Dry-run Token/Shape Debug Info:")
        local_logger.info("  data.crop_size = %s", crop_size)
        local_logger.info("  estimated latent size = %sx%s", latent_size, latent_size)
        local_logger.info("  packed image token count = %s", packed_image_tokens)
        local_logger.info("  model.max_prompt_sequence_length = %s", cfg(config, "model.max_prompt_sequence_length", 128))
        local_logger.info("  condition.lr_token_count = %s", cfg(config, "condition.lr_token_count", 64))
        local_logger.info("  condition.deg_token_count = %s", cfg(config, "condition.deg_token_count", 4))

    seed = cfg(config, "training.seed", 42)
    if seed is not None:
        set_seed(int(seed))

    ds_config = get_deepspeed_config(accelerator)
    if deepspeed_zero_stage(ds_config) == 3:
        if not supports_sync_each_batch:
            raise RuntimeError(
                "DeepSpeed ZeRO-3 is incompatible with Accelerate no_sync gradient accumulation. "
                "Upgrade accelerate to a version with GradientAccumulationPlugin(sync_each_batch=True), "
                "or set training.grad_accum_steps=1 for smoke testing."
            )
        requested_optimizer_offload = cfg(config, "training.deepspeed_optimizer_offload_device", None)
        if requested_optimizer_offload is None and cfg(config, "model.flux_backend", "flux1") == "flux2_klein":
            requested_optimizer_offload = "none"
        original_grad_accum = ds_config.get("gradient_accumulation_steps") if isinstance(ds_config, dict) else None
        original_optimizer_offload = get_deepspeed_optimizer_offload_device(ds_config)
        resolved_ds_config = sync_deepspeed_config_for_training(
            ds_config,
            per_device_batch=per_device_batch,
            grad_accum_steps=grad_accum,
            num_processes=accelerator.num_processes,
            optimizer_offload_device=requested_optimizer_offload,
        )
        sync_deepspeed_plugin_for_training(
            getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None),
            grad_accum_steps=grad_accum,
            optimizer_offload_device=requested_optimizer_offload,
        )
        runtime_config = config.setdefault("_runtime", {})
        runtime_config["deepspeed_zero_stage"] = 3
        runtime_config["disable_transformer_gradient_checkpointing"] = True
        runtime_config["hf_zero3_config"] = resolved_ds_config
        if accelerator.is_main_process:
            if original_grad_accum != resolved_ds_config.get("gradient_accumulation_steps"):
                local_logger.info(
                    "Synchronized DeepSpeed gradient_accumulation_steps from %s to %s.",
                    original_grad_accum,
                    resolved_ds_config.get("gradient_accumulation_steps"),
                )
            current_optimizer_offload = get_deepspeed_optimizer_offload_device(resolved_ds_config)
            if original_optimizer_offload != current_optimizer_offload:
                local_logger.info(
                    "Synchronized DeepSpeed optimizer offload device from %s to %s.",
                    original_optimizer_offload,
                    current_optimizer_offload,
                )
            local_logger.info(
                "Prepared HfDeepSpeedConfig for Flux transformer construction "
                "(train_batch_size=%s, micro_batch=%s, grad_accum=%s).",
                resolved_ds_config.get("train_batch_size"),
                resolved_ds_config.get("train_micro_batch_size_per_gpu"),
                resolved_ds_config.get("gradient_accumulation_steps"),
            )
            local_logger.info("Disabled Flux transformer gradient checkpointing for DeepSpeed ZeRO-3 compatibility.")

    dataset = RGFluxSRJsonlDataset(
        jsonl_path=cfg(config, "data.jsonl_path"),
        crop_size=int(cfg(config, "data.crop_size", 512)),
        scale=int(cfg(config, "data.scale", 4)),
        mode="train",
        use_prompt=bool(cfg(config, "condition.use_prompt", True)),
        use_suggestions=bool(cfg(config, "condition.use_suggestions", True)),
        prompt_variant=cfg(config, "condition.prompt_variant", None),
        use_degradation_vector=bool(cfg(config, "condition.use_degradation_vector", True)),
        vae_align=int(cfg(config, "data.vae_align", 16)),
        return_profile=prompt_schedule["enabled"],
        mixed_crop_enabled=mixed_crop["enabled"],
        full_frame_ratio=mixed_crop["full_frame_ratio"],
        full_frame_max_long_side=mixed_crop["full_frame_max_long_side"],
        full_frame_align=mixed_crop["full_frame_align"],
        full_frame_pad_mode=mixed_crop["full_frame_pad_mode"],
        full_frame_upscale_small=mixed_crop["full_frame_upscale_small"],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch,
        shuffle=True,
        num_workers=int(cfg(config, "data.num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=bool(cfg(config, "data.num_workers", 4) > 0),
        collate_fn=rg_flux_collate_fn,
    )

    artist = build_rg_flux_artist(config)
    init_single_lora = cfg(config, "model.lora_moe.init_from_single_lora", None)
    if init_single_lora and not cfg(config, "training.resume_ckpt", None) and hasattr(artist, "initialize_moe_from_single_lora"):
        loaded_lora_tensors = artist.initialize_moe_from_single_lora(init_single_lora)
        if accelerator.is_main_process:
            local_logger.info(
                "Initialized LoRA-MoE from single-LoRA checkpoint %s (%s tensors).",
                init_single_lora,
                loaded_lora_tensors,
            )
    trainable_named_params = [(name, param) for name, param in artist.named_parameters() if param.requires_grad]
    trainable_params = [param for _, param in trainable_named_params]
    if not trainable_named_params:
        raise RuntimeError("No trainable parameters found for RG-FLUX-SR-MS Stage A/B.")
    lora_params = [
        param
        for name, param in trainable_named_params
        if "lora" in name.lower() or name.startswith("transformer.")
    ]
    lora_param_ids = {id(param) for param in lora_params}
    adapter_params = [param for _, param in trainable_named_params if id(param) not in lora_param_ids]
    param_groups = []
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": float(cfg(config, "training.lr_adapter", 1e-4))})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": float(cfg(config, "training.lr_lora", 5e-5))})

    optimizer_class = torch.optim.AdamW
    if bool(cfg(config, "training.use_8bit_adam", False)):
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.AdamW8bit

    optimizer = optimizer_class(
        param_groups,
        betas=(float(cfg(config, "training.adam_beta1", 0.9)), float(cfg(config, "training.adam_beta2", 0.95))),
        weight_decay=float(cfg(config, "training.weight_decay", 0.01)),
        eps=float(cfg(config, "training.adam_epsilon", 1e-8)),
    )

    max_steps = 1 if dry_run else int(cfg(config, "training.max_steps", 100000))
    lr_scheduler = get_scheduler(
        cfg(config, "training.lr_scheduler", "constant_with_warmup"),
        optimizer=optimizer,
        num_warmup_steps=int(cfg(config, "training.lr_warmup_steps", 0)) * accelerator.num_processes,
        num_training_steps=max_steps * accelerator.num_processes,
        num_cycles=int(cfg(config, "training.lr_num_cycles", 1)),
    )

    artist, optimizer, dataloader, lr_scheduler = accelerator.prepare(artist, optimizer, dataloader, lr_scheduler)
    weight_dtype = weight_dtype_from_accelerator(accelerator)
    text_embedding_cache = get_text_embedding_cache(
        config,
        dtype=cfg(config, "text_encoding.dtype", cfg(config, "model.dtype", "bf16")),
    )

    global_step = 0
    resume_path = resolve_resume_checkpoint(
        output_dir,
        resume_ckpt=cfg(config, "training.resume_ckpt", None),
        auto_resume=cfg_bool(config, "training.auto_resume", True),
    )
    if resume_path:
        if accelerator.is_main_process:
            logger.info("Loading RG-FLUX-SR-MS state from %s", resume_path)
        resume_training_state = cfg_bool(config, "training.resume_training_state", True)
        if accelerator.is_main_process and not resume_training_state:
            logger.warning("training.resume_training_state is false; optimizer and scheduler state will not be restored.")
        global_step = load_rg_checkpoint(
            accelerator,
            artist,
            optimizer,
            lr_scheduler,
            resume_path,
            resume_training_state=resume_training_state,
        )

    if accelerator.is_main_process and report_to is not None:
        accelerator.init_trackers(
            project_name=cfg(config, "training.tracker_project_name", "rg_flux_sr"),
            config=copy.deepcopy(config),
        )

    progress_bar = tqdm(
        range(global_step, max_steps),
        initial=global_step,
        total=max_steps,
        desc="RG-FLUX-SR-MS",
        disable=not accelerator.is_local_main_process,
    )
    fm_weight = float(cfg(config, "loss.fm_weight", 1.0))
    latent_weight = float(cfg(config, "loss.latent_weight", 0.0))
    charb_weight = float(cfg(config, "loss.charb_weight", 0.0))
    down_weight = float(cfg(config, "loss.down_weight", 0.0))
    router_div_weight = float(cfg(config, "loss.router_div_weight", 0.0))
    router_entropy_weight = float(cfg(config, "loss.router_entropy_weight", 0.0))
    router_balance_weight = float(cfg(config, "loss.router_balance_weight", 0.0))
    checkpoint_dir = output_dir / "checkpoints"
    save_every = int(cfg(config, "training.save_every", 5000))
    log_every = int(cfg(config, "training.log_every", 100))
    loss_record_every = int(cfg(config, "training.loss_record_every", 1))
    loss_plot_every = int(cfg(config, "training.loss_plot_every", save_every))
    loss_record_formats = normalize_loss_record_formats(
        cfg(config, "training.loss_record_formats", ["jsonl", "csv"])
    )
    loss_recorder = (
        LossHistoryRecorder(logging_dir, formats=loss_record_formats)
        if accelerator.is_main_process and loss_record_every > 0 and loss_record_formats
        else None
    )
    sigma_sampling = cfg(config, "flow_matching.sigma_sampling", "uniform")
    lr_cond_mode = cfg(config, "condition.lr_cond_mode", "latent_adapter")
    text_encoding_mode = normalize_text_encoding_mode(config)
    fixed_prompt_embedding_cache = {}

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break
            hq = batch["hq"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
            lq_up = batch["lq_up"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
            degradation_vector = batch["degradation_vector"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
            prompts, active_prompt_variant = resolve_batch_prompts(
                batch,
                config,
                global_step,
                prompt_schedule=prompt_schedule,
            )

            unwrapped_artist = accelerator.unwrap_model(artist)
            moe_schedule = (
                unwrapped_artist.set_moe_training_schedule(global_step, max_steps)
                if hasattr(unwrapped_artist, "set_moe_training_schedule")
                else {"enabled": False}
            )
            with torch.no_grad():
                z_hr = unwrapped_artist.encode_images(hq).to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                z_lr = unwrapped_artist.encode_images(
                    lq_up,
                    sample=lr_cond_mode != "flux2_image_concat",
                ).to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                fixed_cache_key = None
                if active_prompt_variant == "fixed" and text_encoding_mode == "online":
                    fixed_cache_key = (len(prompts), tuple(prompts))
                cached_fixed_state = (
                    fixed_prompt_embedding_cache.get(fixed_cache_key)
                    if fixed_cache_key is not None
                    else None
                )
                if cached_fixed_state is None:
                    prompt_embeds, pooled_prompt_embeds, text_ids = resolve_prompt_embeddings(
                        artist=unwrapped_artist,
                        prompts=prompts,
                        image_keys=batch["lq_path"],
                        config=config,
                        device=accelerator.device,
                        dtype=weight_dtype,
                        cache=text_embedding_cache,
                    )
                    if fixed_cache_key is not None:
                        fixed_prompt_embedding_cache[fixed_cache_key] = (
                            prompt_embeds.detach(),
                            pooled_prompt_embeds.detach(),
                            text_ids.detach(),
                        )
                else:
                    prompt_embeds, pooled_prompt_embeds, text_ids = cached_fixed_state
                dino_tokens = unwrapped_artist.extract_visual_tokens(lq_up)
                sigma = sample_sigma(z_hr.shape[0], z_hr.device, sampling=sigma_sampling).to(dtype=weight_dtype)
                eps = torch.randn_like(z_hr)
                z_t, v_target = build_flow_matching_inputs(z_hr, eps=eps, sigma=sigma)

            with accelerator.accumulate(artist):
                with accelerator.autocast():
                    v_pred = artist(
                        z_t=z_t,
                        timestep=sigma,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        degradation_vector=degradation_vector,
                        z_lr=z_lr,
                        dino_tokens=dino_tokens,
                        lr_cond_mode=lr_cond_mode,
                    )
                    if v_pred.shape != v_target.shape:
                        raise RuntimeError(f"v_pred shape {tuple(v_pred.shape)} != target {tuple(v_target.shape)}")
                    loss_fm = torch.nn.functional.mse_loss(v_pred.float(), v_target.float())
                    supervised_losses = compute_stage0b_supervised_losses(
                        artist=unwrapped_artist,
                        config=config,
                        global_step=global_step,
                        loss_fm=loss_fm,
                        z_t=z_t,
                        v_pred=v_pred,
                        sigma=sigma,
                        z_hr=z_hr,
                        hq=hq,
                        lq_up=lq_up,
                        batch=batch,
                    )
                    loss_latent = supervised_losses["loss_latent"]
                    loss_charb = supervised_losses["loss_charb"]
                    loss_lpips = supervised_losses["loss_lpips"]
                    loss_lpips_weight = supervised_losses["loss_lpips_weight"]
                    loss_down = supervised_losses["loss_down"]
                    loss = (
                        fm_weight * loss_fm
                        + latent_weight * loss_latent
                        + charb_weight * loss_charb
                        + loss_lpips_weight * loss_lpips
                        + down_weight * loss_down
                    )
                    moe_aux = (
                        unwrapped_artist.moe_auxiliary_losses()
                        if hasattr(unwrapped_artist, "moe_auxiliary_losses")
                        else {}
                    )
                    loss_div = moe_aux.get("div")
                    loss_entropy = moe_aux.get("entropy")
                    loss_balance = moe_aux.get("balance")
                    if loss_div is not None and router_div_weight:
                        loss = loss + router_div_weight * loss_div
                    if loss_entropy is not None and router_entropy_weight:
                        loss = loss + router_entropy_weight * loss_entropy
                    if loss_balance is not None and router_balance_weight:
                        loss = loss + router_balance_weight * loss_balance

                accelerator.backward(loss)
                if accelerator.sync_gradients and float(cfg(config, "training.max_grad_norm", 1.0)) > 0:
                    accelerator.clip_grad_norm_(trainable_params, float(cfg(config, "training.max_grad_norm", 1.0)))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                step_logs = {
                    "loss": loss.detach().item(),
                    "loss_total": loss.detach().item(),
                    "loss_fm": loss_fm.detach().item(),
                    "loss_latent": loss_latent.detach().item(),
                    "loss_charb": loss_charb.detach().item(),
                    "loss_lpips": loss_lpips.detach().item(),
                    "loss_lpips_weight": float(loss_lpips_weight),
                    "loss_down": loss_down.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if active_prompt_variant is not None:
                    step_logs["prompt/variant_id"] = float(
                        {"fixed": 0, "suggestion": 1, "iqa": 2, "iqa_suggestion": 3}[
                            active_prompt_variant
                        ]
                    )
                spatial_modes = batch.get("spatial_mode", [])
                if mixed_crop["enabled"] and spatial_modes:
                    step_logs["data/full_frame_fraction"] = sum(
                        mode == "full_frame" for mode in spatial_modes
                    ) / len(spatial_modes)
                if loss_div is not None:
                    step_logs["loss_div"] = loss_div.detach().item()
                if loss_entropy is not None:
                    step_logs["loss_entropy"] = loss_entropy.detach().item()
                if loss_balance is not None:
                    step_logs["loss_balance"] = loss_balance.detach().item()
                if moe_schedule.get("enabled"):
                    step_logs["router/temperature"] = moe_schedule.get("temperature", 0.0)
                    moe_logs = (
                        unwrapped_artist.moe_log_stats()
                        if hasattr(unwrapped_artist, "moe_log_stats")
                        else {}
                    )
                    step_logs.update(moe_logs)
                if accelerator.is_main_process:
                    if loss_recorder is not None and global_step % loss_record_every == 0:
                        loss_recorder.append(global_step, step_logs)
                    if log_every > 0 and global_step % log_every == 0:
                        progress_bar.set_postfix(**step_logs)
                        accelerator.log(step_logs, step=global_step)
                if save_every > 0 and global_step % save_every == 0:
                    save_rg_checkpoint(
                        accelerator,
                        artist,
                        optimizer,
                        lr_scheduler,
                        checkpoint_dir / f"checkpoint-{global_step:08d}",
                        global_step,
                    )
                if (
                    accelerator.is_main_process
                    and loss_recorder is not None
                    and loss_plot_every > 0
                    and global_step % loss_plot_every == 0
                ):
                    loss_recorder.write_plot(step=global_step)
                eval_summary = run_rg_flux_evaluation(
                    accelerator,
                    artist,
                    config,
                    exp_name,
                    global_step,
                    weight_dtype,
                    text_embedding_cache=text_embedding_cache,
                    local_logger=local_logger if accelerator.is_main_process else None,
                )
                if accelerator.is_main_process and eval_summary is not None:
                    eval_logs = evaluation_logs_from_summary(eval_summary)
                    if eval_logs:
                        accelerator.log(eval_logs, step=global_step)

    save_rg_checkpoint(
        accelerator,
        artist,
        optimizer,
        lr_scheduler,
        checkpoint_dir / f"checkpoint-{global_step:08d}",
        global_step,
    )
    if accelerator.is_main_process and loss_recorder is not None:
        loss_recorder.write_plot(step=global_step)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_rg_flux_sr_ms.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Run exactly one optimization step.")
    args = parser.parse_args()
    main(args.config, dry_run=args.dry_run)
