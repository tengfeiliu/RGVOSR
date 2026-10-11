import json

import pytest
import torch

from models.sr_observation_loop import ObservationLoop
from tools.run_rl_sr_observation_loop import load_split, rollout


def test_c3_recursion_uses_original_lr_and_hidden_state():
    torch.manual_seed(11)
    model = ObservationLoop(width=16, depth=1, heads=4)
    model.residual_head.weight.data.normal_(std=.01)
    f0 = torch.rand(2, 3, 32, 32) * 2 - 1
    lr = torch.rand_like(f0) * 2 - 1
    outputs = rollout(model, f0, lr)
    assert len(outputs) == 3
    assert all(value.shape == f0.shape for value in outputs)
    assert not torch.allclose(outputs[-1], rollout(model, f0, lr.flip(0))[-1])
    assert not torch.allclose(outputs[0], outputs[-1])


def test_split_rejects_cross_split_source_leakage(tmp_path):
    for name in ("lr.png", "f0.png", "hr.png"):
        (tmp_path / name).touch()
    row = dict(lq_path=str(tmp_path / "lr.png"), _f0_path=str(tmp_path / "f0.png"),
               hq_path=str(tmp_path / "hr.png"), source_hq_path=str(tmp_path / "source.png"))
    path = tmp_path / "split.json"
    path.write_text(json.dumps(dict(train=[row], validation=[row])))
    with pytest.raises(ValueError, match="Source leakage"):
        load_split(path)
