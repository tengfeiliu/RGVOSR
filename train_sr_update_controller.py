"""Train only the RGB acceptance controller on immutable offline proposals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from dataloaders.sr_update_dataset import SRUpdateDataset
from models.sr_update_controller import SRUpdateController
from rl_sr.conservative import conservative_update_losses, fuse_rgb


METRIC_FIELDS = (
    "completed_update", "elapsed_seconds", "batch_size", "loss", "loss_hr", "loss_regress",
    "loss_identity", "loss_smooth", "alpha_mean", "mask_mean", "weight_mean", "gradient_norm",
    "error_current_mean", "error_out_mean", "nondegradation_fraction", "identity_count",
    "max_memory_allocated_bytes",
)
IMMUTABLE_CONFIG_FIELDS = (
    "lr", "crop_size", "seed", "channels", "regress_weight", "identity_weight", "smooth_weight", "lpips_weight",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL of real current/proposal/HR RGB image paths")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=200, help="Total target optimizer updates, including resumed updates")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regress_weight", type=float, default=1.0)
    parser.add_argument("--identity_weight", type=float, default=1.0)
    parser.add_argument("--smooth_weight", type=float, default=0.01)
    parser.add_argument("--lpips_weight", type=float, default=0.05)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--resume", nargs="?", const="auto", help="Controller checkpoint path, or use output_dir when omitted")
    args = parser.parse_args(argv)
    for name in ("max_steps", "batch_size", "crop_size", "channels"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    for name in ("regress_weight", "identity_weight", "smooth_weight", "lpips_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name} must be finite and nonnegative")
    return args


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def train(args):
    started = time.monotonic()
    output_dir = Path(args.output_dir).resolve()
    manifest = Path(args.manifest).resolve()
    config = {**vars(args), "manifest": str(manifest), "output_dir": str(output_dir)}
    dataset = SRUpdateDataset(manifest, crop_size=args.crop_size, augment=True, seed=args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for CPU smoke tests")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sampler_rng = torch.Generator().manual_seed(args.seed)
    checkpoint_path = output_dir / "controller.pt"
    state_path = output_dir / "trainer_state.pt"
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.csv"
    resumed_state = None
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = checkpoint_path if args.resume == "auto" else Path(args.resume).resolve()
        resume_state_path = resume_checkpoint.parent / "trainer_state.pt"
        if not resume_checkpoint.is_file() or not resume_state_path.is_file():
            raise FileNotFoundError(f"Resume requires controller.pt and trainer_state.pt in {resume_checkpoint.parent}")
        resumed_state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        if resumed_state.get("manifest_sha256") != dataset.manifest_sha256:
            raise ValueError("Resume manifest SHA256 differs from the recorded training manifest")
        for name in IMMUTABLE_CONFIG_FIELDS:
            if resumed_state["config"].get(name) != config[name]:
                raise ValueError(f"Resume config differs for {name}; start a separate output directory")
        model = SRUpdateController.load(resume_checkpoint, device=device)
        # trainer_state is the authoritative completed optimizer update. An
        # interruption between the two atomic writes cannot mix model/Adam states.
        model.load_state_dict(resumed_state["model_state_dict"])
    else:
        if checkpoint_path.exists() or state_path.exists() or metrics_path.exists():
            raise FileExistsError(f"Existing training artifacts in {output_dir}; use --resume or a new output_dir")
        model = SRUpdateController(channels=args.channels).to(device=device, dtype=torch.float32)

    image_paths = sorted({record[field] for record in dataset.records for field in dataset.image_fields})
    image_sha256 = {path: _sha256(path) for path in image_paths}
    if resumed_state is not None and resumed_state["provenance"]["image_sha256"] != image_sha256:
        raise ValueError("Resume proposal/input image contents changed despite an unchanged manifest")
    code_root = Path(__file__).resolve().parent
    code_files = ("train_sr_update_controller.py", "dataloaders/sr_update_dataset.py",
                  "models/sr_update_controller.py", "rl_sr/conservative.py")
    provenance = {
        "manifest": str(manifest), "manifest_sha256": dataset.manifest_sha256,
        "image_sha256": image_sha256,
        "source_code_sha256": {name: _sha256(code_root / name) for name in code_files},
        "proposal_rows": len(dataset), "identity_rows": sum(record.get("identity", False) for record in dataset.records),
        "sample_keys": sorted({record["sample_key"] for record in dataset.records}),
        "producers": [json.loads(value) for value in sorted({json.dumps(record["producer"], sort_keys=True) for record in dataset.records})],
        "identity_policy": "explicit current=HR rows with recorded HR-conditioned proposals; no current replacement",
        "seed": args.seed, "torch_version": str(torch.__version__),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    completed_updates = 0
    images_seen = 0
    elapsed_before = 0.0
    final_loss = None
    sample_order = []
    sample_cursor = 0
    previous_peak_allocated = 0
    previous_peak_reserved = 0
    resume_history = []
    if resumed_state is not None:
        optimizer.load_state_dict(resumed_state["optimizer_state_dict"])
        completed_updates = int(resumed_state["completed_updates"])
        images_seen = int(resumed_state["images_seen"])
        elapsed_before = float(resumed_state["elapsed_seconds"])
        final_loss = resumed_state["final_loss"]
        sample_order = resumed_state["sample_order"]
        sample_cursor = int(resumed_state["sample_cursor"])
        sampler_rng.set_state(resumed_state["sampler_rng_state"])
        dataset.rng.setstate(resumed_state["dataset_rng_state"])
        random.setstate(resumed_state["python_rng_state"])
        np.random.set_state(resumed_state["numpy_rng_state"])
        torch.set_rng_state(resumed_state["torch_rng_state"])
        if device.type == "cuda" and resumed_state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(resumed_state["cuda_rng_state"], device)
        previous_peak_allocated = int(resumed_state["max_memory_allocated_bytes"])
        previous_peak_reserved = int(resumed_state["max_memory_reserved_bytes"])
        resume_history = list(resumed_state.get("resume_history", []))
        resume_history.append({"checkpoint": str(resume_checkpoint), "from_update": completed_updates,
                               "batch_size": args.batch_size, "device": args.device})
    if args.max_steps < completed_updates:
        raise ValueError("--max_steps is below the number of already completed updates")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", config)
    prior_metrics = []
    if resume_checkpoint is not None:
        prior_metrics_path = resume_checkpoint.parent / "metrics.csv"
        if prior_metrics_path.is_file():
            with prior_metrics_path.open(newline="", encoding="utf-8") as handle:
                prior_metrics = [row for row in csv.DictReader(handle) if int(row["completed_update"]) <= completed_updates]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(prior_metrics)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss_weights = {"pixel": 1.0, "lpips": args.lpips_weight, "regress": args.regress_weight,
                    "identity": args.identity_weight, "smooth": args.smooth_weight}

    def report(status, failure=None):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = {
            "status": status, "completed_updates": completed_updates, "requested_updates": args.max_steps,
            "images_seen": images_seen, "final_loss": final_loss,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "max_memory_allocated_bytes": max(previous_peak_allocated, torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
            "max_memory_reserved_bytes": max(previous_peak_reserved, torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0),
            "max_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "config": config, "provenance": provenance, "resume_history": resume_history,
            "controller_checkpoint": str(checkpoint_path),
        }
        if failure is not None:
            result["failure"] = {"type": type(failure).__name__, "message": str(failure)}
        _write_json(summary_path, result)
        return result

    def save_update():
        progress = report("running")
        metadata = {"completed_updates": completed_updates, "manifest_sha256": dataset.manifest_sha256,
                    "config": config, "provenance": provenance}
        state = {
            **progress, "manifest_sha256": dataset.manifest_sha256,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "sample_order": sample_order, "sample_cursor": sample_cursor,
            "sampler_rng_state": sampler_rng.get_state(), "dataset_rng_state": dataset.rng.getstate(),
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        temporary_state = state_path.with_name(state_path.name + ".tmp")
        torch.save(state, temporary_state)
        temporary_state.replace(state_path)
        temporary_controller = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        model.save(temporary_controller, metadata=metadata)
        temporary_controller.replace(checkpoint_path)

    try:
        lpips_model = None
        if args.lpips_weight > 0:
            from train_rg_flux_sr import _get_lpips_loss_model
            lpips_model = _get_lpips_loss_model(device)
            lpips_model.requires_grad_(False)
            lpips_model.eval()
        model.train()
        with torch.enable_grad():
            while completed_updates < args.max_steps:
                if sample_cursor >= len(sample_order):
                    sample_order = torch.randperm(len(dataset), generator=sampler_rng).tolist()
                    sample_cursor = 0
                indices = sample_order[sample_cursor:sample_cursor + args.batch_size]
                sample_cursor += len(indices)
                batch = default_collate([dataset[index] for index in indices])
                batch = {name: value.to(device) if torch.is_tensor(value) else value for name, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                gates = model(batch["current"], batch["original_lr_up"], batch["proposal"], batch["target_round"])
                output = fuse_rgb(batch["current"], batch["proposal"], gates["weight"])
                losses = conservative_update_losses(output, batch["current"], batch["hr"], gates["mask"],
                    identity_mask=batch["identity_mask"], lpips_model=lpips_model, weights=loss_weights)
                loss = losses["loss"]
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Controller loss is nonfinite; no optimizer update performed")
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                if not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
                    raise FloatingPointError("Controller gradients are absent or nonfinite; no optimizer update performed")
                gradient_norm = sum(gradient.detach().float().square().sum().item() for gradient in gradients) ** 0.5
                optimizer.step()
                completed_updates += 1
                images_seen += len(indices)
                final_loss = float(loss.detach().item())
                row = {
                    "completed_update": completed_updates, "elapsed_seconds": elapsed_before + time.monotonic() - started,
                    "batch_size": len(indices), "gradient_norm": gradient_norm,
                    "alpha_mean": gates["alpha"].detach().mean().item(), "mask_mean": gates["mask"].detach().mean().item(),
                    "weight_mean": gates["weight"].detach().mean().item(),
                    "error_current_mean": losses["error_current"].detach().mean().item(),
                    "error_out_mean": losses["error_out"].detach().mean().item(),
                    "nondegradation_fraction": (losses["error_out"] <= losses["error_current"]).float().mean().item(),
                    "identity_count": int(batch["identity_mask"].sum().item()),
                    "max_memory_allocated_bytes": max(previous_peak_allocated, torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
                }
                for name in ("loss", "loss_hr", "loss_regress", "loss_identity", "loss_smooth"):
                    row[name] = losses[name].detach().item()
                with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=METRIC_FIELDS).writerow(row)
                save_update()
                print(f"controller update={completed_updates}/{args.max_steps} loss={final_loss:.6f} "
                      f"weight={row['weight_mean']:.6f} batch={len(indices)}", flush=True)
        if not checkpoint_path.is_file():
            save_update()
        return report("complete")
    except Exception as exc:
        report("failed", failure=exc)
        raise


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
