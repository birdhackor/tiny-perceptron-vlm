"""自己的 optimizer／全使用中 RNG 精確續算，及毀損拒絕。"""

import random

import pytest
import torch
from torch import nn

from tiny_perceptron.omni_products.training_runtime import (
    deterministic,
    reset_codebook_moments,
    restore_checkpoint,
    save_checkpoint,
)


def make_state():
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.2), nn.Linear(8, 1))
    return model, torch.optim.AdamW(model.parameters(), lr=0.01), torch.Generator().manual_seed(11)


def updates(model, optimizer, generator, count):
    for _ in range(count):
        inputs = torch.randn(8, 4) * random.random()
        target = torch.randint(0, 2, (8, 1), generator=generator).float()
        optimizer.zero_grad()
        (model(inputs) - target).square().mean().backward()
        optimizer.step()


def test_exact_resume_restores_optimizer_sampler_python_and_torch_rng_after_initialization(tmp_path):
    deterministic(17)
    direct, direct_optimizer, direct_generator = make_state()
    updates(direct, direct_optimizer, direct_generator, 4)
    deterministic(17)
    interrupted, optimizer, generator = make_state()
    updates(interrupted, optimizer, generator, 2)
    path = tmp_path / "own-continuation.pt"
    sha = save_checkpoint(path, interrupted, optimizer, generator, 2, {"scope": "CPU continuation test"})
    deterministic(999)
    resumed, resumed_optimizer, resumed_generator = make_state()
    step, metadata = restore_checkpoint(path, sha, resumed, resumed_optimizer, resumed_generator)
    assert step == 2 and metadata["scope"] == "CPU continuation test"
    updates(resumed, resumed_optimizer, resumed_generator, 2)
    for name, value in resumed.state_dict().items():
        assert torch.equal(value, direct.state_dict()[name])
    for key, state in direct_optimizer.state_dict()["state"].items():
        for field, value in state.items():
            actual = resumed_optimizer.state_dict()["state"][key][field]
            assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual
    with path.open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="SHA"):
        restore_checkpoint(path, sha, resumed, resumed_optimizer, resumed_generator)


def test_training_code_restart_clears_only_replaced_adam_rows_and_keeps_continuation_usage(tmp_path):
    model = nn.Embedding(3, 4)
    optimizer = torch.optim.AdamW(model.parameters())
    model.weight.sum().backward()
    optimizer.step()
    before = optimizer.state[model.weight]["exp_avg"].clone()
    reset_codebook_moments(optimizer, model.weight, torch.tensor([1]))
    assert not optimizer.state[model.weight]["exp_avg"][1].any()
    assert torch.equal(optimizer.state[model.weight]["exp_avg"][[0, 2]], before[[0, 2]])
    generator = torch.Generator().manual_seed(9)
    auxiliary = {"code_usage": torch.tensor([2, 0, 7]), "logs": [{"update": 17}], "best": {"dev_score": 4.5}}
    path = tmp_path / "codebook.pt"
    sha = save_checkpoint(path, model, optimizer, generator, 17, {"scope": "codec training state"}, auxiliary)
    _, metadata = restore_checkpoint(path, sha, model, optimizer, generator)
    assert torch.equal(metadata["_continuation"]["code_usage"], auxiliary["code_usage"])
    assert metadata["_continuation"]["best"] == auxiliary["best"]
