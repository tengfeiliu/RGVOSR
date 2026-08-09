import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


class RouterAblationUtilityTests(unittest.TestCase):
    def test_history_weights_deduplicate_resume_and_stop_at_checkpoint(self):
        from tools.moe_router_ablation import mean_router_weights_from_history

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loss_history.csv"
            fields = [
                "global_step",
                "router/expert_0_usage",
                "router/expert_1_usage",
                "router/expert_0_used",
                "router/expert_1_used",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {"global_step": 1, "router/expert_0_usage": 0.8, "router/expert_1_usage": 0.2},
                        {"global_step": 2, "router/expert_0_usage": 0.7, "router/expert_1_usage": 0.3},
                        # Latest duplicate must replace the earlier step 2 record.
                        {"global_step": 2, "router/expert_0_usage": 0.4, "router/expert_1_usage": 0.6},
                        {"global_step": 3, "router/expert_0_usage": 0.0, "router/expert_1_usage": 1.0},
                    ]
                )
            weights = mean_router_weights_from_history(path, checkpoint_step=2, last_n=2)
            self.assertAlmostEqual(weights[0], 0.6)
            self.assertAlmostEqual(weights[1], 0.4)

    def test_history_weights_reject_invalid_rows_instead_of_silently_skipping(self):
        from tools.moe_router_ablation import mean_router_weights_from_history

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loss_history.csv"
            path.write_text(
                "global_step,router/expert_0_usage,router/expert_1_usage\n"
                "10,nan,1.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "global_step=10"):
                mean_router_weights_from_history(path, checkpoint_step=10)

    def test_derangement_is_deterministic_one_to_one_and_has_no_self_pair(self):
        from tools.moe_router_ablation import deterministic_derangement

        first = deterministic_derangement(20, 3407)
        self.assertEqual(first, deterministic_derangement(20, 3407))
        self.assertEqual(sorted(first), list(range(20)))
        self.assertTrue(all(index != donor for index, donor in enumerate(first)))
        with self.assertRaises(ValueError):
            deterministic_derangement(1, 3407)

    def test_shuffle_trace_keeps_triplet_together_and_stays_inside_dataset(self):
        from tools.moe_router_ablation import load_shuffled_condition_records

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            rows = []
            for index, dataset in enumerate(("a", "a", "b", "b")):
                rows.append(
                    {
                        "sample_index": index,
                        "step_index": 0,
                        "dataset": dataset,
                        "source_image_path": f"{dataset}/{index}.png",
                        "router_condition": [index / 10.0, 0.0],
                        "router_condition_mask": [1.0, float(index % 2)],
                        "router_condition_confidence": [index / 4.0],
                    }
                )
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            shuffled = load_shuffled_condition_records(trace, seed=7)
            for source_index, donor in enumerate(shuffled):
                self.assertNotEqual(source_index, donor["donor_sample_index"])
                self.assertEqual(rows[source_index]["dataset"], donor["donor_dataset"])
                donor_row = rows[donor["donor_sample_index"]]
                self.assertEqual(donor["router_condition"], donor_row["router_condition"])
                self.assertEqual(donor["router_condition_mask"], donor_row["router_condition_mask"])
                self.assertEqual(
                    donor["router_condition_confidence"],
                    donor_row["router_condition_confidence"],
                )


FakeRouterBase = nn.Module if nn is not None else object


@unittest.skipIf(torch is None, "torch is not installed")
class RouterAblationControllerTests(unittest.TestCase):
    class FakeRouter(FakeRouterBase):
        def __init__(self):
            super().__init__()
            self.num_experts = 4

        def forward(
            self,
            router_condition=None,
            router_condition_mask=None,
            router_condition_confidence=None,
            timestep=None,
            routing_mode="topk",
            top_k=2,
            temperature=0.7,
            return_details=False,
        ):
            dense = torch.tensor(
                [[0.1, 0.2, 0.6, 0.1]],
                dtype=router_condition.dtype,
                device=router_condition.device,
            )
            alpha = torch.tensor(
                [[0.0, 0.25, 0.75, 0.0]],
                dtype=router_condition.dtype,
                device=router_condition.device,
            )
            return {
                "alpha": alpha,
                "clean_dense_alpha": dense,
                "dispatch_dense_alpha": dense,
            }

    def run_mode(self, mode, **kwargs):
        from tools.moe_router_ablation import RouterAblationController

        router = self.FakeRouter()
        artist = SimpleNamespace(moe_router=router)
        original_forward = router.forward
        controller = RouterAblationController(mode=mode, num_steps=1, **kwargs)
        controller.install(artist)
        result = router(
            router_condition=torch.tensor([[0.1] * 8]),
            router_condition_mask=torch.ones(1, 8),
            router_condition_confidence=torch.ones(1),
            timestep=torch.tensor([0.75]),
            routing_mode="topk",
            top_k=2,
            temperature=0.7,
            return_details=True,
        )
        controller.uninstall()
        self.assertEqual(router.forward, original_forward)
        self.assertAlmostEqual(sum(controller.records[0]["used_alpha"]), 1.0)
        self.assertEqual(controller.records[0]["timestep"], 0.75)
        return result

    def test_learned_mode_is_identity_and_does_not_mutate_dense_details(self):
        result = self.run_mode("learned_top2")
        self.assertTrue(torch.allclose(result["alpha"], torch.tensor([[0.0, 0.25, 0.75, 0.0]])))
        self.assertTrue(
            torch.allclose(result["clean_dense_alpha"], torch.tensor([[0.1, 0.2, 0.6, 0.1]]))
        )

    def test_fixed_uniform_dense_and_onehot_modes(self):
        fixed = self.run_mode("fixed_mean", fixed_weights=[0.0, 2.0, 8.0, 0.0])
        self.assertTrue(torch.allclose(fixed["alpha"], torch.tensor([[0.0, 0.2, 0.8, 0.0]])))
        uniform = self.run_mode("uniform")
        self.assertTrue(torch.allclose(uniform["alpha"], torch.full((1, 4), 0.25)))
        dense = self.run_mode("dense_soft")
        self.assertTrue(torch.allclose(dense["alpha"], torch.tensor([[0.1, 0.2, 0.6, 0.1]])))
        onehot = self.run_mode("onehot", onehot_expert=3)
        self.assertTrue(torch.equal(onehot["alpha"], torch.tensor([[0.0, 0.0, 0.0, 1.0]])))


class RouterAblationCommandTests(unittest.TestCase):
    def test_command_preserves_full_frame_prompt_and_condition_flags(self):
        from tools.run_rg_flux_moe_router_ablation import build_ablation_inference_command

        args = SimpleNamespace(
            dataset_dirs=["RealLQ250=/data/lq"],
            run_dir="exp/run",
            num_inference_steps=25,
            upscale=4,
            dtype="bf16",
            seed=42,
            jsonl_path="datasets/inference.jsonl",
            text_encoding_mode="online",
            text_embedding_cache=None,
            device="cuda",
            lr_cond_mode="flux2_image_concat",
            min_size=None,
            prompt_variant="condition8_text",
            use_prompt=True,
            use_suggestions=True,
            include_caption=True,
            use_degradation_vector=False,
            full_frame_inference=True,
            restore_input_size=True,
            condition_shuffle_seed=3407,
        )
        command = build_ablation_inference_command(
            args,
            checkpoint_dir=Path("exp/run/checkpoints/checkpoint-00024000"),
            output_dir=Path("out/learned_top2"),
            mode="learned_top2",
        )
        for flag in (
            "--full_frame_inference",
            "--restore_input_size",
            "--include_caption",
            "--use_prompt",
            "--use_suggestions",
            "--no-use_degradation_vector",
        ):
            self.assertIn(flag, command)
        self.assertNotIn("--fixed_weights", command)
        self.assertNotIn("--shuffle_reference_trace", command)

    def test_dry_run_expands_onehot_and_keeps_outputs_outside_run(self):
        from tools.run_rg_flux_moe_router_ablation import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "exp" / "moe_run"
            adapter = run_dir / "checkpoints" / "checkpoint-00024000" / "rg_flux_adapters"
            adapter.mkdir(parents=True)
            (run_dir / "args.json").write_text(
                json.dumps({"model": {"lora_moe": {"num_routed_experts": 2}}}),
                encoding="utf-8",
            )
            logs = run_dir / "logs"
            logs.mkdir()
            with (logs / "loss_history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "global_step",
                        "router/expert_0_usage",
                        "router/expert_1_usage",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "global_step": 24000,
                        "router/expert_0_usage": 0.3,
                        "router/expert_1_usage": 0.7,
                    }
                )
            output_root = root / "router_outputs"
            result = main(
                [
                    "--run_dir",
                    str(run_dir),
                    "--checkpoint_steps",
                    "24000",
                    "--ablation_modes",
                    "learned_top2",
                    "fixed_mean",
                    "onehot",
                    "--dataset_dirs",
                    "test=/data/test",
                    "--ablation_output_root",
                    str(output_root),
                    "--skip_eval",
                    "--dry_run",
                ]
            )
            self.assertEqual(result, 0)
            manifest = json.loads(
                (output_root / run_dir.name / "router_ablation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [record["mode"] for record in manifest["records"]],
                ["learned_top2", "fixed_mean", "onehot_e0", "onehot_e1"],
            )
            self.assertTrue(
                all(str(output_root) in record["inference_output_dir"] for record in manifest["records"])
            )


class RouterAblationAnalysisTests(unittest.TestCase):
    def test_comparison_normalizes_higher_and_lower_better_metrics(self):
        from tools.analyze_rg_flux_moe_router_ablation import compare_mode_to_baseline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            for directory, musiq, niqe in (
                (baseline, [0.8, 0.7], [3.0, 4.0]),
                (candidate, [0.6, 0.5], [5.0, 6.0]),
            ):
                directory.mkdir()
                with (directory / "per_image_scores.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["dataset", "filename", "musiq", "niqe"],
                    )
                    writer.writeheader()
                    for index in range(2):
                        writer.writerow(
                            {
                                "dataset": "test",
                                "filename": f"{index}.png",
                                "musiq": musiq[index],
                                "niqe": niqe[index],
                            }
                        )
                (directory / "summary_scores.json").write_text(
                    json.dumps(
                        {
                            "metric_directions": {
                                "musiq": "higher_better",
                                "niqe": "lower_better",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            rows = compare_mode_to_baseline(
                baseline,
                candidate,
                "learned_top2",
                "uniform",
                bootstrap_samples=50,
            )
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["baseline_advantage_mean"] > 0 for row in rows))
            self.assertTrue(all(row["baseline_win_rate"] == 1.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
