import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


@unittest.skipIf(torch is None, "torch is not installed")
class LoRAMoETests(unittest.TestCase):
    def test_shared_lora_is_active_without_routed_gate(self):
        from models.lora_moe import SharedRoutedMoELoRALinear

        base = nn.Linear(4, 3, bias=False)
        layer = SharedRoutedMoELoRALinear(base, rank=2, alpha=2, num_routed_experts=2)
        with torch.no_grad():
            layer.shared_lora_A.fill_(0.5)
            layer.shared_lora_B.fill_(0.25)
            layer.routed_lora_A.zero_()
            layer.routed_lora_B.zero_()

        x = torch.randn(2, 5, 4)
        layer.set_routing(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

        base_out = layer.base_layer(x)
        moe_out = layer(x)

        self.assertFalse(torch.allclose(base_out, moe_out))
        self.assertTrue(torch.allclose(moe_out - base_out, moe_out - base_out))
        self.assertFalse(layer.base_layer.weight.requires_grad)

    def test_route_logits_full_softmax_and_topk(self):
        from models.lora_moe import route_logits

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        full = route_logits(logits, mode="soft", top_k=2, temperature=2.0)
        top2 = route_logits(logits, mode="topk", top_k=2, temperature=1.0)

        self.assertEqual(full.shape, logits.shape)
        self.assertTrue(torch.all(full > 0))
        self.assertTrue(torch.allclose(full.sum(dim=-1), torch.ones(2)))
        self.assertTrue(torch.allclose(top2.sum(dim=-1), torch.ones(2)))
        self.assertTrue(torch.equal((top2 > 0).sum(dim=-1), torch.tensor([2, 2])))

    def test_profile_latent_router_modes_return_expected_shapes(self):
        from models.lora_moe import ProfileLatentRouter

        prompt_embeds = torch.randn(3, 7, 16)
        z_lr = torch.randn(3, 8, 4, 4)

        for mode in ("stat_only", "conv_only", "stat_conv"):
            router = ProfileLatentRouter(
                prompt_dim=16,
                latent_channels=8,
                num_experts=4,
                hidden_dim=32,
                latent_branch=mode,
            )
            logits, alpha, features = router(prompt_embeds, z_lr, routing_mode="soft", top_k=2, temperature=1.5)
            self.assertEqual(logits.shape, (3, 4))
            self.assertEqual(alpha.shape, (3, 4))
            self.assertEqual(features.shape, (3, 32))
            self.assertTrue(torch.allclose(alpha.sum(dim=-1), torch.ones(3), atol=1e-6))

    def test_moe_auxiliary_losses_are_differentiable(self):
        from models.lora_moe import (
            SharedRoutedMoELoRALinear,
            moe_diversity_loss,
            routing_balance_loss,
            routing_entropy_loss,
        )

        layer = SharedRoutedMoELoRALinear(nn.Linear(4, 3), rank=2, alpha=2, num_routed_experts=3)
        alpha = torch.softmax(torch.randn(5, 3, requires_grad=True), dim=-1)

        loss = (
            moe_diversity_loss([layer])
            + routing_entropy_loss(alpha, encourage_high_entropy=True)
            + routing_balance_loss(alpha)
        )
        loss.backward()

        self.assertIsNotNone(layer.routed_lora_A.grad)
        self.assertIsNotNone(layer.routed_lora_B.grad)
        self.assertIsNone(layer.base_layer.weight.grad)


if __name__ == "__main__":
    unittest.main()
