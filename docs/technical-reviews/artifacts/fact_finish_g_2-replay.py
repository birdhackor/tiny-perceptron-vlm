"""Small CPU checks supporting the fresh G.2 review; no retained binary weights."""

import hashlib
import json
import math
import platform
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.alignment import LoRALinear, dpo_loss  # noqa: E402
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss  # noqa: E402
from tiny_perceptron.training import load_checkpoint, save_checkpoint  # noqa: E402


def tensor_record(tensor):
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
    }


def update(model, optimizer, x, y):
    optimizer.zero_grad()
    masked_loss(model(x)["logits"], y).backward()
    optimizer.step()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    scores = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64, requires_grad=True)
    weights = [math.exp(2), math.exp(1), math.exp(0)]
    manual = [weight / sum(weights) for weight in weights]
    probabilities = scores.softmax(0)
    assert torch.allclose(probabilities, torch.tensor(manual, dtype=torch.float64), atol=1e-15, rtol=0)
    assert [round(value, 3) for value in manual] == [0.665, 0.245, 0.090]
    assert abs(sum(manual) - 1) < 1e-15
    loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([2]))
    assert abs(loss.item() + math.log(manual[2])) < 1e-15
    loss.backward()
    expected_gradient = probabilities.detach().clone()
    expected_gradient[2] -= 1
    assert torch.allclose(scores.grad, expected_gradient, atol=1e-15, rtol=0)
    loss_values = [-math.log(value) for value in [0.090, 0.5, 0.9]]
    assert loss_values[0] > loss_values[1] > loss_values[2]
    w = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    (w - 3).square().backward()
    assert w.grad.item() == -4.0
    finite_difference = (((1.0 + 0.001) - 3) ** 2 - ((1.0 - 0.001) - 3) ** 2) / 0.002
    assert abs(finite_difference + 4.0) < 1e-11
    confidence_cases = []
    for scale in [1.0, 10.0]:
        p = (scores.detach() * scale).softmax(0)
        confidence_cases.append({"scale": scale, "prediction": p.argmax().item(), "confidence": p.max().item()})
        assert p.argmax().item() != 2
    assert confidence_cases[1]["confidence"] > 0.9999
    batches = [batch.tolist() for batch in DataLoader(list(range(5)), batch_size=2, shuffle=False)]
    assert batches == [[0, 1], [2, 3], [4]]

    model = TinyLM(ModelConfig(vocab_size=5, width=8, max_length=8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
    update(model, optimizer, x, y)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "own-checkpoint.pt"
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            step=1,
            metadata={"chars": ["。", "狗", "看", "貓", "，"]},
            training_state={"total_steps": 2, "data_position": 1},
        )
        loaded, payload = load_checkpoint(checkpoint, restore_rng=True)
        weights_only, _ = load_checkpoint(checkpoint)
        restored_optimizer = torch.optim.AdamW(loaded.parameters(), lr=0.01)
        restored_optimizer.load_state_dict(payload["optimizer"])
        fresh_optimizer = torch.optim.AdamW(weights_only.parameters(), lr=0.01)
        assert torch.equal(model(x)["logits"], loaded(x)["logits"])
        assert torch.equal(model(x)["logits"], weights_only(x)["logits"])
        assert payload["step"] == 1
        assert payload["training_state"] == {"total_steps": 2, "data_position": 1}
        assert len(payload["optimizer"]["state"]) > 0
        saved_tensors = {name: tensor_record(value) for name, value in payload["model"].items()}
        for candidate, opt in [(model, optimizer), (loaded, restored_optimizer), (weights_only, fresh_optimizer)]:
            update(candidate, opt, x, y)
        comparison = {}
        for name, value in model.state_dict().items():
            restored_value = loaded.state_dict()[name]
            fresh_value = weights_only.state_dict()[name]
            assert torch.equal(value, restored_value)
            comparison[name] = {
                "saved": saved_tensors[name],
                "after_uninterrupted": tensor_record(value),
                "after_restored": tensor_record(restored_value),
                "restored_max_abs_difference": (value - restored_value).abs().max().item(),
                "fresh_optimizer_max_abs_difference": (value - fresh_value).abs().max().item(),
            }
        fresh_difference = max(record["fresh_optimizer_max_abs_difference"] for record in comparison.values())
        assert fresh_difference > 0
        checkpoint_result = {
            "binary_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "retained_binary": False,
            "saved_step": payload["step"],
            "metadata": payload["metadata"],
            "training_state": payload["training_state"],
            "optimizer_state_entries": len(payload["optimizer"]["state"]),
            "payload_keys": sorted(payload),
            "tensor_comparison": comparison,
            "restored_all_tensors_equal": True,
            "fresh_optimizer_max_abs_difference": fresh_difference,
        }
    chosen = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    rejected = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    reference_chosen = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    reference_rejected = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    preference_loss = dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.1)
    preference_loss.backward()
    assert abs(preference_loss.item() - math.log(2)) < 1e-15
    assert chosen.grad.item() == -0.05 and rejected.grad.item() == 0.05
    assert reference_chosen.grad is None and reference_rejected.grad is None
    layer = LoRALinear(nn.Linear(16, 12), rank=2, alpha=2)
    original_base = {name: value.detach().clone() for name, value in layer.base.state_dict().items()}
    lora_optimizer = torch.optim.SGD(
        [parameter for parameter in layer.parameters() if parameter.requires_grad], lr=0.01
    )
    inputs = torch.randn(3, 16)
    assert torch.equal(layer(inputs), layer.base(inputs))
    trainable = sum(parameter.numel() for parameter in layer.parameters() if parameter.requires_grad)
    assert trainable == 56
    layer(inputs).sum().backward()
    assert layer.base.weight.grad is None and layer.base.bias.grad is None
    lora_optimizer.step()
    assert all(torch.equal(value, original_base[name]) for name, value in layer.base.state_dict().items())
    result = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": "1",
        },
        "softmax": {
            "manual": manual,
            "torch": probabilities.detach().tolist(),
            "rounded": [round(p, 3) for p in manual],
        },
        "cross_entropy_target_2": loss.item(),
        "cross_entropy_gradient": scores.grad.tolist(),
        "negative_log_probability_examples": loss_values,
        "gradient_w1": w.grad.item(),
        "finite_difference": finite_difference,
        "confidence_wrong_target_2": confidence_cases,
        "batch_groups": batches,
        "checkpoint": checkpoint_result,
        "dpo": {
            "loss": preference_loss.item(),
            "chosen_gradient": chosen.grad.item(),
            "rejected_gradient": rejected.grad.item(),
            "reference_gradients_none": True,
        },
        "lora": {
            "trainable_parameters": trainable,
            "base_weight_parameters": 192,
            "base_bias_parameters": 12,
            "base_unchanged": True,
        },
        "all_assertions_passed": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
