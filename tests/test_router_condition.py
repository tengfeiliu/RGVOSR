import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


@unittest.skipIf(torch is None, "torch is not installed")
class RouterConditionTests(unittest.TestCase):
    def test_key_specific_severity_and_negation_do_not_leak(self):
        from models.router_condition import ROUTER_CONDITION_KEYS, extract_router_condition

        condition = extract_router_condition(
            {
                "iqa": {
                    "distortion_type": "blur, noise",
                    "distortion_severity": "mild blur, severe noise",
                }
            }
        )
        values = dict(zip(ROUTER_CONDITION_KEYS, condition.values))
        self.assertAlmostEqual(values["blur"], 0.25)
        self.assertAlmostEqual(values["noise"], 0.75)

        negated = extract_router_condition(
            {"iqa": {"distortion_type": "no noise and severe blur"}}
        )
        values = dict(zip(ROUTER_CONDITION_KEYS, negated.values))
        self.assertEqual(values["noise"], 0.0)
        self.assertAlmostEqual(values["blur"], 0.75)

        generic = extract_router_condition(
            {
                "iqa": {
                    "distortion_type": "blur and noise",
                    "distortion_severity": "severe",
                }
            }
        )
        values = dict(zip(ROUTER_CONDITION_KEYS, generic.values))
        self.assertAlmostEqual(values["blur"], 0.75)
        self.assertAlmostEqual(values["noise"], 0.75)

    def test_fixed_suggestion_boilerplate_is_not_a_condition(self):
        from models.router_condition import ROUTER_CONDITION_KEYS, extract_router_condition

        fixed = (
            "Preserve the original exposure, color relationships, geometry, and semantic content. "
            "Do not invent unsupported details."
        )
        condition = extract_router_condition({"suggestion": fixed})
        self.assertEqual(condition.valid_mask, (0.0,) * len(ROUTER_CONDITION_KEYS))
        self.assertEqual(condition.confidence, 0.0)

        specific = extract_router_condition(
            {"suggestion": "Preserve text readability. " + fixed}
        )
        values = dict(zip(ROUTER_CONDITION_KEYS, specific.valid_mask))
        self.assertEqual(values["structure_risk"], 1.0)

    def test_production_suggestion_phrases_are_covered(self):
        from models.router_condition import ROUTER_CONDITION_KEYS, extract_router_condition

        cases = (
            ("Mildly reduce visible blur while preserving stable boundaries.", "blur", 0.25),
            ("Moderately suppress visible noise while preserving natural texture.", "noise", 0.5),
            (
                "Mildly suppress visible compression artifacts without changing source structure.",
                "compression",
                0.25,
            ),
            ("Moderately reduce visible ringing and halo artifacts.", "ringing_aliasing", 0.5),
            (
                "Mildly reduce visible aliasing and pixelation on source-supported edges.",
                "ringing_aliasing",
                0.25,
            ),
            (
                "Conservatively recover source-supported edge and texture detail.",
                "texture_loss",
                0.5,
            ),
        )
        for text, key, expected in cases:
            with self.subTest(text=text):
                condition = extract_router_condition({"suggestion": text})
                values = dict(zip(ROUTER_CONDITION_KEYS, condition.values))
                valid = dict(zip(ROUTER_CONDITION_KEYS, condition.valid_mask))
                self.assertEqual(valid[key], 1.0)
                # Suggestion-only physical values are confidence-scaled by 0.6.
                self.assertAlmostEqual(values[key], expected * 0.6)

    def test_legacy_degradation_vector_is_never_read(self):
        from models.router_condition import extract_router_condition, router_condition_tensors

        a = extract_router_condition({})
        b = extract_router_condition(
            {"degradation_vector": {"blur": 1.0, "noise": 1.0}}
        )
        self.assertEqual(a.values, b.values)
        self.assertEqual(a.valid_mask, b.valid_mask)

    def test_condition_and_timestep_router_never_require_lr(self):
        from models.lora_moe import ProfileLatentRouter

        condition = torch.tensor([[0.75, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0]])
        mask = (condition > 0).float()
        confidence = torch.tensor([0.25])
        router = ProfileLatentRouter(
            prompt_dim=16,
            latent_channels=8,
            num_experts=4,
            hidden_dim=32,
            input_mode="condition8_timestep",
        )
        low = router(
            router_condition=condition,
            router_condition_mask=mask,
            router_condition_confidence=confidence,
            timestep=torch.tensor([0.0]),
            return_details=True,
        )
        high = router(
            router_condition=condition,
            router_condition_mask=mask,
            router_condition_confidence=confidence,
            timestep=torch.tensor([1.0]),
            return_details=True,
        )
        self.assertEqual(low["alpha"].shape, (1, 4))
        self.assertFalse(torch.allclose(low["features"], high["features"]))

    def test_condition_validation_fails_fast(self):
        from models.lora_moe import ProfileLatentRouter

        router = ProfileLatentRouter(
            prompt_dim=16,
            latent_channels=8,
            hidden_dim=32,
            input_mode="condition8_timestep",
        )
        with self.assertRaisesRegex(ValueError, "finite values"):
            router(
                router_condition=torch.tensor([[float("nan")] + [0.0] * 7]),
                timestep=torch.tensor([0.5]),
            )
        with self.assertRaisesRegex(ValueError, "raw flow-matching sigma"):
            router(
                router_condition=torch.zeros(1, 8),
                timestep=torch.tensor([500.0]),
            )

    def test_clean_dense_is_separate_from_noisy_dispatch(self):
        from models.lora_moe import ProfileLatentRouter

        torch.manual_seed(7)
        router = ProfileLatentRouter(
            prompt_dim=16,
            latent_channels=8,
            hidden_dim=32,
            input_mode="condition8",
        ).train()
        details = router(
            router_condition=torch.zeros(2, 8),
            routing_mode="topk",
            noise_std=0.5,
            return_details=True,
        )
        self.assertFalse(torch.equal(details["logits"], details["routed_logits"]))
        self.assertFalse(
            torch.allclose(details["clean_dense_alpha"], details["dispatch_dense_alpha"])
        )

    def test_seeded_expert_perturbation_is_reproducible(self):
        from models.lora_moe import SharedRoutedMoELoRALinear

        def build(seed):
            torch.manual_seed(seed)
            layer = SharedRoutedMoELoRALinear(
                nn.Linear(8, 6),
                rank=3,
                alpha=3,
                num_routed_experts=4,
            )
            layer.initialize_routed_residuals(perturb_scale=0.3)
            return layer

        first = build(42)
        second = build(42)
        third = build(43)
        self.assertTrue(torch.equal(first.routed_lora_A, second.routed_lora_A))
        self.assertFalse(torch.equal(first.routed_lora_A, third.routed_lora_A))

    def test_ema_balance_has_batch_one_router_gradient(self):
        from models.lora_moe import routing_balance_loss

        logits = torch.tensor([[0.2, 0.1, -0.1, -0.2]], requires_grad=True)
        probabilities = torch.softmax(logits, dim=-1)
        usage = torch.tensor([0.7, 0.1, 0.1, 0.1])
        loss = routing_balance_loss(probabilities, ema_dispatch_usage=usage)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[0, 0].item(), 0.0)
        self.assertTrue(torch.all(logits.grad[0, 1:] < 0.0))

    def test_router_ema_state_round_trip(self):
        from models.lora_moe import ProfileLatentRouter, add_moe_persistent_buffers

        class Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe_router = ProfileLatentRouter(
                    16,
                    8,
                    hidden_dim=32,
                    input_mode="condition8",
                )

        first = Container()
        first.moe_router.update_ema_usage(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        # This mirrors save_trainable's parameter-only gather plus explicit buffer append.
        saved = add_moe_persistent_buffers(first, {})
        self.assertIn("moe_router.ema_dispatch_usage", saved)
        self.assertIn("moe_router.ema_update_count", saved)
        second = Container()
        second.load_state_dict(saved, strict=False)
        self.assertTrue(
            torch.equal(
                first.moe_router.ema_dispatch_usage,
                second.moe_router.ema_dispatch_usage,
            )
        )
        self.assertEqual(
            first.moe_router.ema_update_count.item(),
            second.moe_router.ema_update_count.item(),
        )

    def test_inference_paired_profile_uses_the_same_extractor(self):
        from inference_rg_flux_sr import profile_with_donor_iqa, profile_with_donor_suggestion
        from models.router_condition import extract_router_condition, router_condition_tensors

        source = {
            "iqa": {"distortion_type": "mild blur"},
            "suggestion": "Remove mild blur.",
        }
        donor = {
            "iqa": {"distortion_type": "severe noise"},
            "suggestion": "Suppress severe noise.",
        }
        paired_iqa = profile_with_donor_iqa(source, donor)
        paired_suggestion = profile_with_donor_suggestion(source, donor)
        paired_extracted = extract_router_condition(paired_iqa)
        values, valid_mask, confidence = router_condition_tensors(paired_iqa)
        self.assertTrue(torch.equal(values, torch.tensor(paired_extracted.values)))
        self.assertTrue(torch.equal(valid_mask, torch.tensor(paired_extracted.valid_mask)))
        self.assertAlmostEqual(confidence.item(), paired_extracted.confidence)
        self.assertNotEqual(
            extract_router_condition(source).values,
            extract_router_condition(paired_iqa).values,
        )
        self.assertNotEqual(
            extract_router_condition(source).values,
            extract_router_condition(paired_suggestion).values,
        )

    def test_teacher_schedule_boundaries(self):
        from models.lora_moe import blend_teacher_routing, teacher_router_mix

        self.assertEqual(teacher_router_mix(0.0, 0.15, 0.35, True), 0.0)
        self.assertEqual(teacher_router_mix(0.15, 0.15, 0.35, True), 0.0)
        self.assertAlmostEqual(teacher_router_mix(0.25, 0.15, 0.35, True), 0.5)
        self.assertEqual(teacher_router_mix(0.35, 0.15, 0.35, True), 1.0)
        self.assertEqual(teacher_router_mix(0.0, 0.15, 0.35, False), 1.0)

        router = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]])
        teacher = torch.tensor([[0.1, 0.7, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])
        valid_mask = torch.tensor([[1.0] + [0.0] * 7, [0.0] * 8])
        used = blend_teacher_routing(router, teacher, router_mix=0.0, valid_mask=valid_mask)
        self.assertTrue(torch.equal(used[0], teacher[0]))
        self.assertTrue(torch.equal(used[1], router[1]))

    def test_timestep_prototypes_keep_requested_source_coverage(self):
        from tools.init_flux2_lora_moe import expected_prototype_feature_count

        self.assertEqual(expected_prototype_feature_count(128, "condition8"), 128)
        self.assertEqual(
            expected_prototype_feature_count(128, "condition8_timestep"),
            640,
        )

    def test_functional_diversity_is_reparameterization_invariant(self):
        from models.lora_moe import SharedRoutedMoELoRALinear, moe_diversity_loss

        layer = SharedRoutedMoELoRALinear(
            nn.Linear(12, 7),
            rank=3,
            alpha=3,
            num_routed_experts=3,
        )
        with torch.no_grad():
            layer.routed_lora_A.normal_()
            layer.routed_lora_B.normal_()
        before = moe_diversity_loss([layer], probe_dim=12)
        scale = torch.tensor([2.0, 0.5, 1.5])
        with torch.no_grad():
            layer.routed_lora_B.mul_(scale[None, None, :])
            layer.routed_lora_A.div_(scale[None, :, None])
        after = moe_diversity_loss([layer], probe_dim=12)
        self.assertTrue(torch.allclose(before, after, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
