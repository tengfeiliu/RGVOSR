"""Isolated, resumable 4x real-LR diagnostics with an old-C negative control."""

from __future__ import annotations

import copy
import gc
import json
import random
import time
from pathlib import Path

from PIL import Image

from rl_sr.artifact_store import read_jsonl, write_json_atomic
from rl_sr.schema import adapter_content_sha256, build_sample_key, content_sha256, identity_sha256
from tools.run_sr_self_refine_experiment import case_seed

DATASETS = ("reallq250", "reallr200")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_real_split(experiment):
    """Select eight real inputs per dataset without modifying paired splits."""
    from models.prompt_builder import build_sr_prompt
    from models.router_condition import ROUTER_CONDITION_VERSION, extract_router_condition

    source = Path(experiment.config["real_source_jsonl"]).expanduser().resolve()
    source_hash = content_sha256(source)
    seed = int(experiment.config.get("seed", 42))
    groups = {name: {} for name in DATASETS}
    for original in read_jsonl(source):
        name = str(original.get("dataset_name", "")).strip().casefold()
        if name not in groups:
            continue
        path = Path(original["lq_path"]).expanduser()
        if not path.is_absolute():
            path = Path(getattr(experiment, "base_repo", source.parent)) / path
        path = path.resolve()
        row = copy.deepcopy(original)
        row["dataset_name"], row["lq_path"] = name, str(path)
        groups[name].setdefault(str(path), row)
    rng = random.Random(seed)
    selected = []
    for name in DATASETS:
        candidates = [groups[name][path] for path in sorted(groups[name])]
        if len(candidates) < 8:
            raise ValueError(f"Real diagnostic requires eight unique {name} inputs; found {len(candidates)}")
        rows = sorted(rng.sample(candidates, 8), key=lambda row: row["lq_path"])
        for row in rows:
            lr_hash = content_sha256(row["lq_path"])
            profile = (row.get("unipercept_raw") or {}).get("profile") or {}
            row.update(
                hq_path=row["lq_path"], has_gt=False, _split="real_diagnostic",
                _sample_key=build_sample_key(name, f"original_lr/sha256/{lr_hash}", lr_hash),
                _prompt=build_sr_prompt(profile, use_prompt=True, use_suggestions=True,
                                        prompt_variant="iqa_suggestion", include_caption=True),
                _router_condition=extract_router_condition(profile, version=ROUTER_CONDITION_VERSION).as_dict(),
                _lr_sha256=lr_hash, _real_source_sha256=source_hash, _real_split_seed=seed,
            )
            selected.append(row)
    saved = Path(experiment.run) / "real_split.json"
    if saved.exists():
        if _read(saved) != selected:
            raise ValueError("Real diagnostic split or source content changed; use a new run directory")
    else:
        write_json_atomic(saved, selected)
    return selected


def _output_path(root, row, round_number):
    arm = "F0" if round_number == 1 else "oldC_negative"
    return root / arm / f"round_{round_number:02d}" / row["dataset_name"] / f"{row['_sample_key']}.png"


def _expected(root, row, round_number, adapters, policy_hash, base_seed):
    path = _output_path(root, row, round_number)
    parent = Path(row["lq_path"]) if round_number == 1 else _output_path(root, row, round_number - 1)
    with Image.open(row["lq_path"]) as image:
        original_size = image.size
    target = tuple(value * 4 for value in original_size)
    prepared = tuple(max(16, value - value % 16) for value in target)
    adapter = adapters[0 if round_number == 1 else 1]
    return dict(
        path=str(path), sample_key=row["_sample_key"], dataset_name=row["dataset_name"],
        round=round_number, has_gt=False, policy_sha256=policy_hash,
        adapter_path=adapter["path"], adapter_sha256=adapter["sha256"],
        input_path=str(parent), input_sha256=content_sha256(parent) if parent.is_file() else None,
        lr_path=row["lq_path"], lr_sha256=content_sha256(row["lq_path"]),
        prompt_sha256=identity_sha256({"prompt": row["_prompt"]}),
        router_condition_sha256=identity_sha256(row["_router_condition"]),
        seed=case_seed(row["_sample_key"], round_number, base_seed), nfe=25,
        input_upscale=4 if round_number == 1 else 1,
        original_size=list(original_size), prepared_size=list(prepared), output_size=list(target),
        uses_refinement_adapter=round_number > 1,
    )


def _cached_output(expected, root):
    path = Path(expected["path"])
    metadata = path.with_suffix(".json")
    if not path.exists() and not metadata.exists():
        return None
    cached = _read(metadata) if metadata.is_file() else None
    if cached is not None:
        if any(cached.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Real4x output provenance changed: {path}")
        prepared_path = Path(cached.get("prepared_input_path", ""))
        if not prepared_path.is_file() or content_sha256(prepared_path) != cached.get("prepared_input_sha256"):
            raise ValueError(f"Real4x prepared input content hash changed: {prepared_path}")
    if path.exists() != metadata.exists():
        # A crash can interrupt publication between the PNG and its sidecar.
        # Preserve the unmatched artifact; regenerate only under the already
        # verified policy and parent. Complete pairs never take this path.
        if expected["input_sha256"] is None:
            raise ValueError(f"Incomplete real4x output lost its verified parent: {path}")
        remaining = path if path.exists() else metadata
        if not remaining.is_file():
            raise ValueError(f"Real4x incomplete artifact is not a file: {remaining}")
        quarantine = root / "quarantine" / f"{expected['sample_key']}_r{expected['round']}_{time.time_ns()}"
        quarantine.mkdir(parents=True)
        remaining.rename(quarantine / remaining.name)
        write_json_atomic(quarantine / "reason.json", dict(
            reason="incomplete output/metadata publication", original_path=str(remaining),
            policy_sha256=expected["policy_sha256"], parent_sha256=expected["input_sha256"],
        ))
        return None
    if not path.is_file() or not metadata.is_file():
        raise ValueError(f"Invalid real4x output/metadata pair: {path}")
    if cached.get("output_sha256") != content_sha256(path):
        raise ValueError(f"Real4x output content hash changed: {path}")
    with Image.open(path) as image:
        if list(image.size) != expected["output_size"] or image.mode != "RGB":
            raise ValueError(f"Real4x output dimensions or RGB mode changed: {path}")
    return cached


def run_real_diagnostic(experiment):
    """Generate fresh F0 4x and old-C rounds 2..4, with no GT or NR scoring."""
    from inference_rg_flux_sr import prepare_lq_up

    rows = prepare_real_split(experiment)
    root = Path(experiment.run) / "real4x"
    negative = Path(experiment.config["negative_adapter"]).expanduser()
    if not negative.is_absolute():
        raise ValueError("negative_adapter must be an absolute old-C adapter path")
    paths = [Path(experiment.f0_path()).resolve(), negative.resolve()]
    adapters = [dict(path=str(path), sha256=adapter_content_sha256(path)) for path in paths]
    seed = int(experiment.config.get("seed", 42))
    policy = dict(
        version="real4x_v1", has_gt=False, diagnostic_only=True, upscale=4, vae_align=16,
        seed=seed, adapters=adapters, split_sha256=identity_sha256({"rows": rows}),
        sampler=dict(nfe=25, schedule="linear", init_mode="pure_noise", sigma_start=1.0),
        source_code_sha256=content_sha256(__file__),
        proposal_source_sha256=content_sha256(Path(__file__).with_name("run_sr_self_refine_experiment.py")),
        preparation_source_sha256=content_sha256(Path(__file__).resolve().parents[1] / "inference_rg_flux_sr.py"),
    )
    signature = identity_sha256(policy)
    policy_path = root / "policy.json"
    if policy_path.exists():
        if _read(policy_path) != {**policy, "policy_sha256": signature}:
            raise ValueError("Real4x policy, adapter, split, seed or source code changed; use a new run directory")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Unrecorded real4x artifacts cannot be reused")
    else:
        write_json_atomic(policy_path, {**policy, "policy_sha256": signature})

    # Check every existing artifact before loading either model. Missing future
    # rounds are allowed; a recorded round must retain its complete parent.
    for round_number in range(1, 5):
        for row in rows:
            expected = _expected(root, row, round_number, adapters, signature, seed)
            if expected["input_sha256"] is None and Path(expected["path"]).exists():
                raise ValueError(f"Real4x output lost its parent input: {expected['path']}")
            _cached_output(expected, root)

    outputs = []
    for adapter_index, rounds in ((0, (1,)), (1, (2, 3, 4))):
        model = None
        try:
            for row in rows:
                for round_number in rounds:
                    expected = _expected(root, row, round_number, adapters, signature, seed)
                    cached = _cached_output(expected, root)
                    if cached is not None:
                        outputs.append(cached)
                        continue
                    if expected["input_sha256"] is None:
                        raise ValueError(f"Missing real4x parent input for round {round_number}")
                    if model is None:
                        model = experiment.artist(paths[adapter_index])
                    prepared_path = root / "inputs" / f"{row['_sample_key']}_r{round_number}.png"
                    prepared_path.parent.mkdir(parents=True, exist_ok=True)
                    prepared, _, tensor = prepare_lq_up(
                        expected["input_path"], upscale=expected["input_upscale"], align=16,
                    )
                    del tensor
                    if list(prepared.size) != expected["prepared_size"]:
                        raise ValueError("Real4x prepared current-image dimensions differ from policy")
                    prepared.save(prepared_path)
                    proposal, actual_seed = experiment.proposal(
                        model, row, prepared_path, round_number, f0=round_number == 1,
                    )
                    if actual_seed != expected["seed"]:
                        raise ValueError("Real4x proposal seed differs from recorded seed")
                    temporary_output = root / "work" / f"{row['_sample_key']}_r{round_number}.png"
                    experiment.save_image(proposal, temporary_output)
                    del proposal
                    output_path = Path(expected["path"])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_final = output_path.with_suffix(".tmp.png")
                    with Image.open(temporary_output) as image:
                        image.convert("RGB").resize(
                            tuple(expected["output_size"]), Image.Resampling.LANCZOS,
                        ).save(temporary_final)
                    temporary_final.replace(output_path)
                    cached = {**expected, "output_sha256": content_sha256(output_path),
                              "prepared_input_path": str(prepared_path),
                              "prepared_input_sha256": content_sha256(prepared_path)}
                    write_json_atomic(output_path.with_suffix(".json"), cached)
                    outputs.append(cached)
                    experiment.journal(f"Real4x {row['dataset_name']} {row['_sample_key'][:12]} round {round_number}")
        finally:
            del model
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    result = dict(status="completed", has_gt=False, diagnostic_only=True, upscale=4,
                  sample_count=len(rows), policy_sha256=signature, outputs=outputs,
                  interpretation="Qualitative F0 4x and old-C negative control; excluded from training and selection")
    write_json_atomic(root / "diagnostic.json", result)
    return result
