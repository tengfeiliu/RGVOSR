"""F0 reuse correctness without loading FLUX, CUDA, or IQA networks."""

import copy
import importlib
import json
import shutil
import sys
import types
from pathlib import Path

import pytest
import yaml
from PIL import Image

from tools.reuse_sr_f0_outputs import reuse_f0_outputs


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def cached_run(tmp_path, monkeypatch):
    # Hashes and states use their real implementations, independently of the
    # unrelated torch utilities eagerly imported by rl_sr.__init__.
    package = types.ModuleType("rl_sr")
    package.__path__ = [str(Path("rl_sr").resolve())]
    monkeypatch.setitem(sys.modules, "rl_sr", package)
    checkpoint = tmp_path / "checkpoint" / "rg_flux_adapters"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter.pt").write_bytes(b"frozen F0 test adapter")
    old = tmp_path / "old_run" / "01_f0_round1_train_state"
    images = old / "round_01" / "default"
    images.mkdir(parents=True)
    dataset = tmp_path / "data"
    dataset.mkdir()
    rows, sources = [], []
    for index in range(4):
        source = dataset / f"input{index}.png"
        hr = dataset / f"hr{index}.png"
        generated = images / source.name
        Image.new("RGB", (8, 8), (index, 0, 0)).save(source)
        Image.new("RGB", (8, 8), (index, 10, 0)).save(hr)
        Image.new("RGB", (8, 8), (index, 20, 0)).save(generated)
        rows.append({"dataset": "default", "sample_id": source.stem,
                     "source_path": str(source), "source_input_root": str(dataset),
                     "rounds": [{"round": 1, "path": str(generated), "exists": True}]})
        sources.append({"lq_path": str(source), "hq_path": str(hr)})
    source_jsonl = dataset / "source.jsonl"
    source_jsonl.write_text("".join(json.dumps(r) + "\n" for r in sources), encoding="utf-8")
    (old / "sample_lineage.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write_json(old / "iterative_manifest.json", {
        "status": "running", "checkpoint_path": str(checkpoint), "output_dir": str(old), "rounds": [],
    })
    write_json(old / "round_01" / "inference_manifest.json", {
        "checkpoint_path": str(checkpoint), "output_dir": str(old / "round_01"),
        "sampling": {"num_inference_steps": 25, "schedule": "empirical_shift", "init_mode": "pure_noise",
                     "sigma_start": 1.0, "seed": 42, "input_upscale": 1, "uses_refinement_adapter": False},
        "suggestion_pairing": "matched",
        "datasets": [{"name": "default", "prompt_variant": "fixed", "include_caption": False,
                      "pre_cropped_input": True, "restore_input_size": False, "valid_image_count": 4,
                      "suggestion_pairing_manifest": "old-full-dataset-pairing.csv"}],
    })
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "condition": {"prompt_variant": "fixed", "include_caption": False, "lr_cond_mode": "flux2_image_concat",
                      "use_degradation_vector": False, "refinement": {"enabled": True}},
        "data": {"pre_cropped": True}, "text_encoding": {"mode": "online"},
        "flow_matching": {"inference_schedule": "empirical_shift"},
    }), encoding="utf-8")
    inputs = tmp_path / "selected.txt"
    inputs.write_text("\n".join(r["source_path"] for r in rows[1:3]) + "\n", encoding="utf-8")
    args = types.SimpleNamespace(source_dir=old, input=inputs, checkpoint=checkpoint,
                                 config=config, seed=42, output_dir=tmp_path / "new_01")
    return types.SimpleNamespace(args=args, rows=rows, source_jsonl=source_jsonl, tmp=tmp_path)


def test_reuse_subset_while_old_iqa_running_and_build_actual_sampler_states(cached_run, monkeypatch):
    case = cached_run
    args = case.args
    before = {str(p): p.read_bytes() for p in args.source_dir.rglob("*") if p.is_file()}
    # Accept the old run root as well as its 01 directory.
    direct_args = copy.copy(args)
    direct_args.source_dir = args.source_dir.parent
    result = reuse_f0_outputs(direct_args)
    assert result["sample_count"] == 2
    assert result["source_status"] == "running"
    assert result["adapter_verification"] == "legacy_checkpoint_current_content"
    assert not result["metrics_reused"]
    saved = json.loads((args.output_dir / "iterative_manifest.json").read_text())
    assert saved["metrics_status"] == "skipped_for_f0_reuse"
    records = [json.loads(line) for line in (args.output_dir / "sample_lineage.jsonl").read_text().splitlines()]
    assert {r["source_path"] for r in records} == {r["source_path"] for r in case.rows[1:3]}
    for row in records:
        image = Path(row["rounds"][0]["path"])
        assert image.read_bytes() == (args.source_dir / "round_01" / "default" / image.name).read_bytes()
        assert image.is_relative_to(args.output_dir)
    assert before == {str(p): p.read_bytes() for p in args.source_dir.rglob("*") if p.is_file()}

    builder = importlib.import_module("tools.build_sr_refinement_states")
    output = case.tmp / "02_states_for_c.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "build_sr_refinement_states.py", "--lineage_jsonl", str(args.output_dir / "sample_lineage.jsonl"),
        "--source_jsonl", str(case.source_jsonl), "--dataset_id", "toy", "--artifact_root", str(args.output_dir),
        "--producer_adapter", str(args.checkpoint), "--output_jsonl", str(output), "--max_round", "2",
        "--sampler_manifest", str(args.output_dir / "round_01" / "inference_manifest.json"),
        "--prompt_variant", "fixed", "--no-include_caption",
    ])
    summary = builder.build_states(builder.parse_args())
    assert summary["state_count"] == 2
    assert summary["sampler"]["schedule"] == "empirical_shift"  # Not silently relabelled linear.


@pytest.mark.parametrize("failure", ["missing_image", "missing_sample", "corrupt_image", "duplicate", "checkpoint", "seed", "prompt", "shuffled"])
def test_rejects_incompatible_or_incomplete_cache_before_copying(cached_run, failure):
    args = cached_run.args
    image = args.source_dir / "round_01" / "default" / "input1.png"
    lineage_file = args.source_dir / "sample_lineage.jsonl"
    manifest_file = args.source_dir / "round_01" / "inference_manifest.json"
    if failure == "missing_image":
        image.unlink()
    elif failure == "corrupt_image":
        image.write_bytes(b"unfinished PNG")
    elif failure == "missing_sample":
        lineage_file.write_text("\n".join(lineage_file.read_text().splitlines()[2:]), encoding="utf-8")
    elif failure == "duplicate":
        with lineage_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(cached_run.rows[1]) + "\n")
    elif failure == "checkpoint":
        other = cached_run.tmp / "other_adapter"
        other.mkdir()
        (other / "adapter.pt").write_bytes(b"different policy")
        args.checkpoint = other
    elif failure == "seed":
        args.seed = 7
    else:
        manifest = json.loads(manifest_file.read_text())
        if failure == "prompt":
            manifest["datasets"][0]["include_caption"] = True
        else:
            manifest["suggestion_pairing"] = "shuffled"
        write_json(manifest_file, manifest)
    with pytest.raises((ValueError, OSError)):
        reuse_f0_outputs(args)
    assert not args.output_dir.exists()


def test_identical_adapter_in_another_mount_and_moved_output_tree(cached_run):
    args = cached_run.args
    copied_checkpoint = cached_run.tmp / "new_mount" / "rg_flux_adapters"
    shutil.copytree(args.checkpoint, copied_checkpoint)
    args.checkpoint = copied_checkpoint
    moved = cached_run.tmp / "relocated_01"
    args.source_dir.rename(moved)
    args.source_dir = moved
    assert reuse_f0_outputs(args)["sample_count"] == 2
    with pytest.raises(FileExistsError):
        reuse_f0_outputs(args)


def test_missing_legacy_checkpoint_is_not_silently_trusted(cached_run):
    args = cached_run.args
    path = args.source_dir / "round_01" / "inference_manifest.json"
    metadata = json.loads(path.read_text())
    metadata["checkpoint_path"] = str(cached_run.tmp / "unavailable-checkpoint")
    write_json(path, metadata)
    with pytest.raises(FileNotFoundError, match="no saved adapter hash"):
        reuse_f0_outputs(args)


def test_multiround_skips_f0_forward_but_retains_anchors_conditions_and_iqa(cached_run, monkeypatch):
    from test_rg_flux_iterative_inference import import_iterative_module

    case = cached_run
    reuse_f0_outputs(case.args)
    iterative = import_iterative_module()
    original_list_images = iterative.list_images

    def list_images(path):
        if Path(path).suffix == ".txt":
            return [Path(line) for line in Path(path).read_text().splitlines()]
        return original_list_images(path)

    monkeypatch.setattr(iterative, "list_images", list_images)
    conditions = {r["source_path"].replace("\\", "/"): {"profile": {"caption": r["sample_id"]}}
                  for r in case.rows[1:3]}
    monkeypatch.setattr(iterative, "load_jsonl_conditions", lambda _: conditions)
    inferences, evaluations = [], []

    def infer(**kwargs):
        inferences.append(kwargs["refinement_round"])
        assert kwargs["anchor_path_resolver"] is not None
        output = kwargs["output_dir"]
        output.mkdir(parents=True)
        for image in list_images(kwargs["input_path"]):
            anchor = kwargs["anchor_path_resolver"](image, "default", None)
            assert anchor != image
            assert anchor.name == image.name
            actual = iterative.condition_for_image(kwargs["condition_index"], image, dataset_name="default")
            assert actual == conditions[str(anchor).replace("\\", "/")]
            shutil.copyfile(image, output / image.name)
        return {}

    def evaluate(**kwargs):
        images = list(kwargs["dataset_dirs"]["default"].glob("*.png"))
        evaluations.append(len(images))
        return {"summary": []}

    monkeypatch.setattr(iterative, "run_inference_dataset", infer)
    monkeypatch.setattr(iterative, "evaluate_dataset_dirs", evaluate)
    args = iterative.build_arg_parser().parse_args([
        "--input", str(case.args.input), "--checkpoint", str(case.args.checkpoint),
        "--config", str(case.args.config), "--output_dir", str(case.tmp / "04_multiround"),
        "--refiner_checkpoint", str(case.args.checkpoint), "--reuse_f0_dir", str(case.args.output_dir),
        "--upscale", "1", "--iterations", "4", "--metrics", "clipiqa", "--dtype", "fp32",
    ])
    manifest = iterative.run_iterative_inference(args)
    assert inferences == [2, 3, 4]
    assert evaluations == [2, 2, 2, 2]
    assert manifest["rounds"][0]["reused_f0"] is True
    assert not any(r["reused_f0"] for r in manifest["rounds"][1:])
    for selected in case.rows[1:3]:
        name = Path(selected["source_path"]).name
        assert (case.tmp / "04_multiround" / "round_01" / "default" / name).read_bytes() == (
            case.args.output_dir / "round_01" / "default" / name).read_bytes()
