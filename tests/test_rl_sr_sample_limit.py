"""CPU-only coverage for A-E training subset selection and run continuity."""

import json
import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml
from PIL import Image

from tools.create_sr_input_manifest import build_manifest
from tools.resolve_rl_sr_run_dir import load_context
from tools.create_sr_evaluation_manifests import build_manifests


def make_dataset(root, count=40):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        image = root / f"image_{index:04d}.png"
        image.write_bytes(b"test-image")
        rows.append({"lq_path": str(image), "hq_path": str(root / f"hr_{index}.png")})
    path = root / "train.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path, rows


def make_config(root, **rl_settings):
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump({
        "data": {"jsonl_path": "train.jsonl"},
        "rl_sr": rl_settings,
    }), encoding="utf-8")
    return path


def selected_names(path):
    return [Path(line).name for line in path.read_text(encoding="utf-8").splitlines()]


def test_limit_is_deterministic_portable_and_does_not_edit_source(tmp_path):
    selections = []
    for server in ("server_a", "different_mount_b"):
        source, _ = make_dataset(tmp_path / server)
        original = source.read_bytes()
        output = source.parent / "inputs.txt"
        summary = build_manifest(source, output, "training", max_samples=7, subset_seed=42)
        assert summary["available_input_count"] == 40
        assert summary["input_count"] == 7
        names = selected_names(output)
        assert len(set(names)) == 7
        assert names == sorted(names)
        selections.append(names)
        assert source.read_bytes() == original
        saved = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert saved["max_samples"] == 7
        assert saved["subset_seed"] == 42
    assert selections[0] == selections[1]


def test_default_keeps_all_and_limit_above_available_is_capped(tmp_path):
    source, rows = make_dataset(tmp_path)
    for maximum in (0, 4000):
        output = tmp_path / f"inputs_{maximum}.txt"
        result = build_manifest(source, output, "training", max_samples=maximum)
        assert result["input_count"] == len(rows)
        assert output.read_text(encoding="utf-8").splitlines() == [row["lq_path"] for row in rows]


def test_sampling_deduplicates_and_seed_changes_the_subset(tmp_path):
    source, rows = make_dataset(tmp_path)
    with source.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rows[0]) + "\n")
    outputs = []
    for seed in (42, 93):
        output = tmp_path / f"inputs_{seed}.txt"
        result = build_manifest(source, output, "training", max_samples=8, subset_seed=seed)
        assert result["available_input_count"] == 40
        outputs.append(selected_names(output))
    assert outputs[0] != outputs[1]


@pytest.mark.parametrize("bad_limit", [-1, 2.5, True, "4000"])
def test_bad_limits_are_rejected_before_any_output(tmp_path, bad_limit):
    source, _ = make_dataset(tmp_path, 2)
    output = tmp_path / "inputs.txt"
    with pytest.raises(ValueError, match="non-negative integer"):
        build_manifest(source, output, "training", max_samples=bad_limit)
    assert not output.exists()


def test_reusing_manifest_preserves_it_and_rejects_new_selection(tmp_path):
    source, _ = make_dataset(tmp_path)
    output = tmp_path / "inputs.txt"
    build_manifest(source, output, "training", max_samples=8)
    before = output.read_bytes(), output.stat().st_mtime_ns
    build_manifest(source, output, "training", max_samples=8, reuse_existing=True)
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before
    for settings in ({"max_samples": 9}, {"max_samples": 8, "subset_seed": 93}):
        with pytest.raises(ValueError, match="Start a new run"):
            build_manifest(source, output, "training", reuse_existing=True, **settings)
        assert output.read_bytes() == before[0]


def test_old_full_manifest_remains_usable(tmp_path):
    source, rows = make_dataset(tmp_path, 3)
    output = tmp_path / "inputs.txt"
    output.write_text("".join(row["lq_path"] + "\n" for row in rows), encoding="utf-8")
    output.with_suffix(".manifest.json").write_text(json.dumps({"input_count": 3}), encoding="utf-8")
    assert build_manifest(source, output, "training", reuse_existing=True)["input_count"] == 3


def test_yaml_and_cli_overrides_drive_context_and_run_name(tmp_path):
    config = make_config(tmp_path, train_max_samples=4000, train_subset_seed=42)
    from_yaml = load_context(config, "checkpoint-00036000", tmp_path, None)
    assert from_yaml["train_max_samples"] == 4000
    assert "_n4000-ss42_" in from_yaml["run_name"]
    override = load_context(config, "checkpoint-00036000", tmp_path, None,
                            train_max_samples_override=100, train_subset_seed_override=7)
    assert override["train_max_samples"] == 100
    assert override["train_subset_seed"] == 7
    assert "_n100-ss7_" in override["run_name"]
    unlimited = load_context(config, "checkpoint-00036000", tmp_path, None, train_max_samples_override=0)
    assert unlimited["train_max_samples"] == 0
    assert "-ss" not in unlimited["run_name"]


def test_old_yaml_defaults_to_full_dataset(tmp_path):
    config = make_config(tmp_path)
    context = load_context(config, "checkpoint-00036000", tmp_path, None)
    assert context["train_max_samples"] == 0
    assert context["train_subset_seed"] == 42


def test_resume_inherits_subset_and_rejects_size_or_seed_change(tmp_path):
    config = make_config(tmp_path)
    run = tmp_path / "existing"
    run.mkdir()
    (run / "run_context.json").write_text(json.dumps({
        "train_max_samples": 4000, "train_subset_seed": 123,
    }), encoding="utf-8")
    context = load_context(config, "checkpoint-00036000", tmp_path, None, existing_run_dir=run)
    assert context["train_max_samples"] == 4000
    assert context["train_subset_seed"] == 123
    for overrides in ({"train_max_samples_override": 0}, {"train_subset_seed_override": 42}):
        with pytest.raises(ValueError, match="Cannot change the training subset"):
            load_context(config, "checkpoint-00036000", tmp_path, None, existing_run_dir=run, **overrides)


def test_cli_creates_context_with_effective_override(tmp_path):
    config = make_config(tmp_path)
    result = subprocess.run([
        sys.executable, "tools/resolve_rl_sr_run_dir.py", "--config", str(config),
        "--f0_checkpoint", "checkpoint-00036000", "--repo_root", str(tmp_path),
        "--train_max_samples", "4000", "--train_subset_seed", "7", "--create",
    ], check=True, capture_output=True, text=True)
    run = Path(result.stdout.strip())
    saved = json.loads((run / "run_context.json").read_text(encoding="utf-8"))
    assert saved["train_max_samples"] == 4000
    assert saved["train_subset_seed"] == 7
    assert "_n4000-ss7_" in run.name


def test_training_cap_does_not_limit_or_split_evaluation_jsonl(tmp_path):
    rows = []
    for name, count in (("RealLQ250", 3), ("RealLR200", 2)):
        for index in range(count):
            path = tmp_path / f"{name}_{index}.png"
            path.write_bytes(b"image")
            rows.append({"lq_path": str(path), "dataset": name})
    source = tmp_path / "evaluation.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = source.read_bytes()
    config = make_config(tmp_path, train_max_samples=1, evaluation={
        "jsonl_path": str(source),
        "datasets": [
            {"name": "RealLQ250", "dataset_filter": "RealLQ250", "expected_count": 3},
            {"name": "RealLR200", "dataset_filter": "RealLR200", "expected_count": 2},
        ],
    })
    output = tmp_path / "evaluation"
    build_manifests(config, tmp_path, output)
    assert len((output / "RealLQ250_lq_inputs.txt").read_text(encoding="utf-8").splitlines()) == 3
    assert len((output / "RealLR200_lq_inputs.txt").read_text(encoding="utf-8").splitlines()) == 2
    assert source.read_bytes() == before


def test_selected_inputs_propagate_to_c_and_every_rl_round(tmp_path, monkeypatch):
    # State construction is CPU-only. Avoid the package initializer, which
    # imports unrelated torch snapshot utilities, while using the real schema,
    # artifact store, prompt builder and state builder below.
    package = types.ModuleType("rl_sr")
    package.__path__ = [str(Path("rl_sr").resolve())]
    monkeypatch.setitem(sys.modules, "rl_sr", package)
    builder = importlib.import_module("tools.build_sr_refinement_states")
    source, rows = make_dataset(tmp_path / "dataset", 12)
    for index, row in enumerate(rows):
        Image.new("RGB", (4, 4), (index, 0, 0)).save(row["lq_path"])
        Image.new("RGB", (8, 8), (index, 0, 0)).save(row["hq_path"])
    input_list = tmp_path / "inputs.txt"
    build_manifest(source, input_list, "training", max_samples=5)
    selected = input_list.read_text(encoding="utf-8").splitlines()
    generated = tmp_path / "generated"
    generated.mkdir()
    lineage = []
    for source_path in selected:
        rounds = []
        for round_number in range(1, 5):
            image = generated / f"{Path(source_path).stem}_r{round_number}.png"
            Image.new("RGB", (8, 8), (round_number, 1, 0)).save(image)
            rounds.append({"round": round_number, "path": str(image), "exists": True})
        lineage.append({"dataset": "default", "source_path": source_path, "rounds": rounds})
    lineage_path = generated / "sample_lineage.jsonl"
    lineage_path.write_text("".join(json.dumps(row) + "\n" for row in lineage), encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter.pt").write_bytes(b"test adapter bytes for content hashing")
    for max_round, expected_count in ((2, 5), (4, 15)):
        output = tmp_path / f"states_{max_round}.jsonl"
        monkeypatch.setattr(sys, "argv", [
            "build_sr_refinement_states.py", "--lineage_jsonl", str(lineage_path),
            "--source_jsonl", str(source), "--dataset_id", "toy",
            "--artifact_root", str(generated), "--producer_adapter", str(adapter),
            "--output_jsonl", str(output), "--max_round", str(max_round),
            "--prompt_variant", "fixed", "--no-include_caption",
        ])
        summary = builder.build_states(builder.parse_args())
        assert summary["state_count"] == expected_count
        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        for round_number in range(2, max_round + 1):
            assert summary["round_counts"][str(round_number)] == 5
            paths = {Path(row["runtime"]["original_lr"]).resolve() for row in records
                     if row["round_index"] == round_number}
            assert paths == {Path(value).resolve() for value in selected}
