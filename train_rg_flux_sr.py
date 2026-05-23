import argparse
import copy
import inspect
import json
import logging
import os
from pathlib import Path

import torch
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
from models.prompt_builder import build_sr_prompt
from rg_flux_fm import build_flow_matching_inputs, sample_multistep_fm, sample_sigma


logger = get_logger(__name__)


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


def save_rg_checkpoint(accelerator, artist, optimizer, lr_scheduler, checkpoint_dir, global_step):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(artist)
    unwrapped.save_trainable(checkpoint_dir / "rg_flux_adapters")
    torch.save(
        {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        checkpoint_dir / "training_state.pt",
    )


def load_rg_checkpoint(accelerator, artist, optimizer, lr_scheduler, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    adapter_dir = checkpoint_dir / "rg_flux_adapters"
    unwrapped = accelerator.unwrap_model(artist)
    unwrapped.load_trainable(adapter_dir if adapter_dir.exists() else checkpoint_dir, is_trainable=True)

    state_path = checkpoint_dir / "training_state.pt"
    global_step = 0
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state.get("optimizer", {}))
        lr_scheduler.load_state_dict(state.get("lr_scheduler", {}))
        global_step = int(state.get("global_step", 0))
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
            if not lq_path or not Path(lq_path).exists() or not isinstance(result, dict):
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


def run_rg_flux_evaluation(accelerator, artist, config, exp_name, global_step, weight_dtype, local_logger=None):
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
                prompt = build_sr_prompt(
                    result,
                    use_prompt=bool(cfg(config, "condition.use_prompt", True)),
                    use_suggestions=bool(cfg(config, "condition.use_suggestions", True)),
                )
                lq_up = prepare_eval_lq_up(
                    record["lq_path"],
                    upscale=int(cfg(config, "data.scale", 4)),
                    align=int(cfg(config, "data.vae_align", 16)),
                ).to(accelerator.device, dtype=weight_dtype)
                z_lr = unwrapped_artist.encode_images(lq_up).to(accelerator.device, dtype=weight_dtype)
                prompt_embeds, pooled_prompt_embeds, text_ids = unwrapped_artist.encode_prompts(
                    [prompt],
                    device=accelerator.device,
                    dtype=weight_dtype,
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

    report_to = normalize_report_to(cfg(config, "training.report_to", None))
    exp_name = cfg(config, "training.exp_name", None) or make_experiment_name(config)
    output_root = Path(cfg(config, "training.output_dir", "exp_rg_flux_sr"))
    output_dir = output_root / exp_name
    logging_dir = output_dir / cfg(config, "training.logging_dir", "logs")
    per_device_batch = int(cfg(config, "data.batch_size", 1))
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
        local_logger.info("  vae_device = %s", cfg(config, "model.vae_device", "cpu"))
        local_logger.info("  max_prompt_sequence_length = %s", cfg(config, "model.max_prompt_sequence_length", 128))
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
        use_degradation_vector=bool(cfg(config, "condition.use_degradation_vector", True)),
        vae_align=int(cfg(config, "data.vae_align", 16)),
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

    global_step = 0
    resume_path = resolve_resume_checkpoint(
        output_dir,
        resume_ckpt=cfg(config, "training.resume_ckpt", None),
        auto_resume=cfg_bool(config, "training.auto_resume", True),
    )
    if resume_path:
        if accelerator.is_main_process:
            logger.info("Loading RG-FLUX-SR-MS state from %s", resume_path)
        global_step = load_rg_checkpoint(accelerator, artist, optimizer, lr_scheduler, resume_path)

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
    checkpoint_dir = output_dir / "checkpoints"
    save_every = int(cfg(config, "training.save_every", 5000))
    log_every = int(cfg(config, "training.log_every", 100))
    sigma_sampling = cfg(config, "flow_matching.sigma_sampling", "uniform")
    lr_cond_mode = cfg(config, "condition.lr_cond_mode", "latent_adapter")

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break
            hq = batch["hq"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
            lq_up = batch["lq_up"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
            degradation_vector = batch["degradation_vector"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
            prompts = batch["prompt"]

            unwrapped_artist = accelerator.unwrap_model(artist)
            with torch.no_grad():
                z_hr = unwrapped_artist.encode_images(hq).to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                z_lr = unwrapped_artist.encode_images(lq_up).to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                prompt_embeds, pooled_prompt_embeds, text_ids = unwrapped_artist.encode_prompts(
                    prompts,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
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
                    loss = fm_weight * loss_fm

                accelerator.backward(loss)
                if accelerator.sync_gradients and float(cfg(config, "training.max_grad_norm", 1.0)) > 0:
                    accelerator.clip_grad_norm_(trainable_params, float(cfg(config, "training.max_grad_norm", 1.0)))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if accelerator.is_main_process and global_step % log_every == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "loss_fm": loss_fm.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                if save_every > 0 and global_step % save_every == 0:
                    save_rg_checkpoint(
                        accelerator,
                        artist,
                        optimizer,
                        lr_scheduler,
                        checkpoint_dir / f"checkpoint-{global_step:08d}",
                        global_step,
                    )
                eval_summary = run_rg_flux_evaluation(
                    accelerator,
                    artist,
                    config,
                    exp_name,
                    global_step,
                    weight_dtype,
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
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_rg_flux_sr_ms.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Run exactly one optimization step.")
    args = parser.parse_args()
    main(args.config, dry_run=args.dry_run)
