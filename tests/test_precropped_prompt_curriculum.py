import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image


def complete_profile():
    return {
        "caption": "A person stands beside a bicycle on a street.",
        "iaa": {},
        "iqa": {
            "distortion_location": "Blur is visible across fine edges.",
            "distortion_severity": "The degradation is moderate.",
            "distortion_type": "Defocus blur and mild noise.",
            "overall_quality": "Technical fidelity and fine detail are reduced.",
        },
        "ista": {},
        "suggestion": "Moderately reduce blur while preserving source structure.",
    }


class PrecroppedPromptCurriculumTests(unittest.TestCase):
    def test_crop_coordinates_and_sample_ids_are_deterministic(self):
        from tools.generate_precropped_unipercept_cache import (
            deterministic_crop_positions,
            stable_sample_id,
        )

        kwargs = {
            "width": 1400,
            "height": 900,
            "source_key": "/data/source.png",
            "crop_size": 512,
            "crops_per_image": 2,
            "crop_seed": 42,
            "max_crop_iou": 0.25,
            "crop_search_attempts": 32,
        }
        first = deterministic_crop_positions(**kwargs)
        second = deterministic_crop_positions(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertLessEqual(first[1]["iou_with_first"], 0.25)
        self.assertNotEqual(
            stable_sample_id("/data/source.png", 0, 512, 42),
            stable_sample_id("/data/source.png", 1, 512, 42),
        )

    def test_single_legal_crop_position_still_returns_two_records(self):
        from tools.generate_precropped_unipercept_cache import (
            deterministic_crop_positions,
        )

        positions = deterministic_crop_positions(
            width=512,
            height=512,
            source_key="small.png",
            crop_size=512,
            crops_per_image=2,
        )

        self.assertEqual(len(positions), 2)
        self.assertEqual(
            [(item["x"], item["y"]) for item in positions],
            [(0, 0), (0, 0)],
        )
        self.assertFalse(positions[1]["overlap_constraint_met"])

    def test_short_side_is_upscaled_bicubic_without_padding(self):
        from tools.generate_precropped_unipercept_cache import resize_short_side

        image = Image.new("RGB", (300, 200))
        resized, resized_size, scale = resize_short_side(image, 512)

        self.assertEqual(resized.size, (768, 512))
        self.assertEqual(resized_size, (768, 512))
        self.assertAlmostEqual(scale, 2.56)

    def test_schedule_boundaries_and_reachable_stages(self):
        try:
            from train_rg_flux_sr import (
                prompt_variant_for_step,
                reachable_prompt_conditions,
                resolve_prompt_schedule,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"training dependencies are unavailable: {exc}")

        base = {
            "training": {"max_steps": 20000},
            "condition": {
                "prompt_variant": "iqa_suggestion",
                "include_caption": True,
                "prompt_schedule": {
                    "enabled": True,
                    "switch_step": 10000,
                    "before_variant": "fixed",
                    "before_include_caption": False,
                    "after_variant": "iqa_suggestion",
                    "after_include_caption": True,
                },
            },
        }
        schedule = resolve_prompt_schedule(base)
        self.assertEqual(
            prompt_variant_for_step(schedule, 9999),
            {"variant": "fixed", "include_caption": False},
        )
        self.assertEqual(
            prompt_variant_for_step(schedule, 10000),
            {"variant": "iqa_suggestion", "include_caption": True},
        )
        self.assertEqual(len(reachable_prompt_conditions(base, schedule)), 2)

        direct = json.loads(json.dumps(base))
        direct["condition"]["prompt_schedule"]["switch_step"] = 0
        direct_schedule = resolve_prompt_schedule(direct)
        self.assertEqual(
            prompt_variant_for_step(direct_schedule, 0),
            {"variant": "iqa_suggestion", "include_caption": True},
        )
        self.assertEqual(
            reachable_prompt_conditions(direct, direct_schedule),
            [{"variant": "iqa_suggestion", "include_caption": True}],
        )

        fixed_only = json.loads(json.dumps(base))
        fixed_only["condition"]["prompt_schedule"]["switch_step"] = 20000
        fixed_schedule = resolve_prompt_schedule(fixed_only)
        self.assertEqual(
            prompt_variant_for_step(fixed_schedule, 19999),
            {"variant": "fixed", "include_caption": False},
        )
        self.assertEqual(
            reachable_prompt_conditions(fixed_only, fixed_schedule),
            [{"variant": "fixed", "include_caption": False}],
        )

        invalid = json.loads(json.dumps(base))
        invalid["condition"]["prompt_schedule"]["switch_step"] = 20001
        with self.assertRaisesRegex(ValueError, "training.max_steps"):
            resolve_prompt_schedule(invalid)

    def test_token_preflight_rejects_untruncated_overflow(self):
        try:
            from train_rg_flux_sr import (
                resolve_prompt_schedule,
                validate_dataset_prompt_token_lengths,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"training dependencies are unavailable: {exc}")

        class FakeArtist:
            def prompt_token_lengths(self, prompts):
                return [513 for _ in prompts]

        config = {
            "model": {
                "flux_backend": "flux2_klein",
                "max_prompt_sequence_length": 512,
            },
            "text_encoding": {"mode": "online"},
            "training": {"max_steps": 1},
            "condition": {
                "prompt_variant": "iqa_suggestion",
                "include_caption": True,
                "prompt_schedule": {
                    "enabled": True,
                    "switch_step": 0,
                    "before_variant": "fixed",
                    "before_include_caption": False,
                    "after_variant": "iqa_suggestion",
                    "after_include_caption": True,
                },
            },
        }
        records = [
            {
                "sample_id": "sample-1",
                "lq_path": "/data/lq.png",
                "profile": complete_profile(),
            }
        ]
        with self.assertRaisesRegex(ValueError, "tokens=513"):
            validate_dataset_prompt_token_lengths(
                FakeArtist(),
                records,
                config,
                prompt_schedule=resolve_prompt_schedule(config),
            )

    def test_dataset_defaults_to_strict_precropped_mode(self):
        try:
            from dataloaders.rg_flux_jsonl_dataset import RGFluxSRJsonlDataset
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"dataset dependencies are unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hq_path = root / "hq.png"
            lq_path = root / "lq.png"
            Image.new("RGB", (512, 512)).save(hq_path)
            Image.new("RGB", (512, 512)).save(lq_path)
            jsonl_path = root / "train.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "sample_id": "sample-1",
                        "hq_path": str(hq_path),
                        "lq_path": str(lq_path),
                        "crop": {"crop_index": 0},
                        "unipercept_raw": {"profile": complete_profile()},
                        "result": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dataset = RGFluxSRJsonlDataset(
                jsonl_path,
                prompt_variant="iqa_suggestion",
                include_caption=True,
            )
            sample = dataset[0]

        self.assertEqual(sample["spatial_mode"], "pre_cropped")
        self.assertEqual(tuple(sample["hq"].shape), (3, 512, 512))
        self.assertEqual(tuple(sample["lq_up"].shape), (3, 512, 512))
        self.assertEqual(sample["sample_id"], "sample-1")

    def test_new_single_and_moe_configs_share_precropped_curriculum(self):
        config_paths = [
            Path(
                "configs/train_rg_flux2_klein_sr_stage0b_512_prompt_curriculum_precropped.yaml"
            ),
            Path(
                "configs/train_rg_flux2_klein_sr_moe_stage0b_512_prompt_curriculum_precropped.yaml"
            ),
        ]
        configs = [
            yaml.safe_load(path.read_text(encoding="utf-8"))
            for path in config_paths
        ]
        for config in configs:
            self.assertTrue(config["data"]["pre_cropped"])
            self.assertEqual(config["data"]["crop_size"], 512)
            self.assertEqual(
                config["condition"]["prompt_schedule"]["switch_step"],
                0,
            )
            self.assertEqual(
                config["condition"]["prompt_schedule"]["after_variant"],
                "iqa_suggestion",
            )
            self.assertTrue(
                config["condition"]["prompt_schedule"][
                    "after_include_caption"
                ]
            )
            self.assertFalse(config["evaluation"]["enabled"])
        self.assertNotIn("lora_backend", configs[0]["model"])
        self.assertEqual(configs[1]["model"]["lora_backend"], "moe")
        self.assertEqual(configs[1]["model"]["lora_moe"]["top_k"], 2)
        self.assertEqual(configs[1]["loss"]["image_loss_crop_size"], 256)


if __name__ == "__main__":
    unittest.main()
