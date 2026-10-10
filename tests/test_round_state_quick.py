import pytest
import torch
from models.sr_round_state_adapter import RoundStateAdapter, hold_moe_routing, choose_candidate, gradient_consistency
from models.lora_moe import SharedRoutedMoELoRALinear


def test_zero_gate_identity_and_gradient():
    module = RoundStateAdapter(12, max_round=4)
    baseline = torch.randn(3, 12)
    rounds = torch.tensor([2, 3, 4])
    result = module.inject(baseline, rounds)
    assert torch.equal(result, baseline)
    result.square().sum().backward()
    assert module.round_gate.grad.abs().item() > 0


def test_mixed_rounds_and_bounds():
    module = RoundStateAdapter(12, max_round=4)
    module.round_gate.data.fill_(0.8)
    value = module.inject(torch.zeros(3, 12), torch.tensor([2, 3, 4]))
    assert not torch.equal(value[0], value[1])
    with pytest.raises(ValueError): module.inject(torch.zeros(1,12), torch.tensor([5]))
    module.state_gate.data.fill_(5)
    current = torch.randn(3,128,8,8)
    delta = module.state_delta(current, current * .5, torch.tensor([2,3,4]))
    assert delta.abs().max() <= module.max_state_delta
    zero = RoundStateAdapter(12).state_delta(current,current,torch.tensor([2,3,4]))
    assert torch.count_nonzero(zero) == 0


def test_checkpoint_routing_is_held_and_cleared():
    from torch.utils.checkpoint import checkpoint
    layer = SharedRoutedMoELoRALinear(torch.nn.Linear(4,4), rank=2, alpha=2, num_routed_experts=2)
    layer.routed_lora_B.data.normal_()
    alpha = torch.tensor([[.2,.8]], requires_grad=True)
    x = torch.randn(1,2,4, requires_grad=True)
    with hold_moe_routing(layer,alpha):
        checkpoint(layer,x,use_reentrant=False).sum().backward()
        assert layer._routing_alpha is alpha
    assert layer._routing_alpha is None
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all()
    checkpoint_grad=alpha.grad.clone()
    alpha.grad=None; layer.zero_grad(); x.grad=None
    with hold_moe_routing(layer,alpha): layer(x).sum().backward()
    torch.testing.assert_close(alpha.grad,checkpoint_grad)


def test_batched_sampler_matches_existing_per_case_noise():
    from tools.run_sr_round_state_quick import sample_batch
    from rg_flux_fm import sample_multistep_fm
    class Toy:
        def __call__(self,z_t,**kwargs): return z_t*.1
        def decode_latents(self,z): return z
    current=torch.zeros(2,4,2,2)
    embeds=dict(prompt_embeds=torch.zeros(1,2,3),pooled_prompt_embeds=torch.zeros(1,0),text_ids=torch.zeros(1,2,4))
    cond=torch.zeros(2,8); seeds=[13,27]
    result=sample_batch(Toy(),current,current,embeds,cond,cond,torch.zeros(2),2,seeds,steps=3)
    expected=[]
    for seed in seeds:
        torch.manual_seed(seed)
        expected.append(sample_multistep_fm(Toy(),(1,4,2,2),embeds['prompt_embeds'],embeds['pooled_prompt_embeds'],
            z_lr=current[:1],num_steps=3,schedule='linear',device='cpu',dtype=torch.float32).clamp(-1,1))
    torch.testing.assert_close(result,torch.cat(expected),rtol=0,atol=0)


def test_selection_never_uses_test_or_hides_rejected_candidate():
    f0 = dict(psnr=20.,ssim=.6,lpips=.2,dists=.1)
    bad = dict(psnr=18.,ssim=.5,lpips=.3,dists=.2)
    good = dict(psnr=20.,ssim=.6,lpips=.19,dists=.09)
    picked = choose_candidate(f0, [('bad',bad),('good',good)])
    assert picked['selected']=='good' and not picked['candidates'][0]['eligible']
    picked = choose_candidate(f0,[('bad',bad)])
    assert picked['selected'] is None and picked['candidates'][0]['metrics']==bad


def test_gradient_loss_matches_hr_edges_and_backpropagates():
    target=torch.zeros(1,3,8,8); target[:,:,:,4:]=1
    assert gradient_consistency(target,target).item()==0
    assert gradient_consistency(target+2,target).item()==0 # Charbonnier constrains absolute color.
    pred=torch.zeros_like(target,requires_grad=True)
    loss=gradient_consistency(pred,target)
    assert loss.item()>0
    loss.backward(); assert pred.grad.abs().sum()>0
