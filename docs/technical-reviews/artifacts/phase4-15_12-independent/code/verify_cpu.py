"""Independent bounded CPU checks for lesson 15.12; no training or checkpoints."""
import contextlib
import io
import json
import math
import os
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.modern import MoEFFN

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
original = (Path(__file__).resolve().parents[1] / "inputs/fence-1.py").read_text()
assert "backward" not in original and "optimizer" not in original
results = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads()}, "capacity_cases": [], "dropless_dispatch": []}
for factor, expected_capacity, expected_overflow in [(2 / 3, 2, [3, 0]), (1, 3, [2, 0]), (2, 6, [0, 0])]:
    code = original if factor == 2 / 3 else original.replace("capacity_factor = 2 / 3", f"capacity_factor = {factor}")
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, "original-fence-or-factor-only-variant", "exec"), namespace)
    assert namespace["capacity"] == expected_capacity
    assert namespace["counts"].tolist() == [5, 1]
    assert namespace["overflow"].tolist() == expected_overflow
    assert namespace["chosen"].tolist() == [0, 0, 0, 1, 0, 0]
    denominator = namespace["tokens"] * namespace["k"]
    results["capacity_cases"].append({"factor": factor, "capacity_per_expert_assignments": expected_capacity, "counts": [5, 1], "overflow": expected_overflow, "total_capacity": expected_capacity * 2, "assignment_denominator": denominator, "overflow_rate": sum(expected_overflow) / denominator, "stdout": stdout.getvalue(), "router_choice_unchanged": True})

# k=2 uses assignments, not unique tokens. No capacity policy is applied here.
selected = torch.tensor([[0, 1]] * 6)
capacity = math.ceil((2 / 3) * selected.shape[0] * selected.shape[1] / 2)
counts = torch.bincount(selected.flatten(), minlength=2)
overflow = (counts - capacity).clamp(min=0)
assert capacity == 4 and counts.tolist() == [6, 6] and overflow.tolist() == [2, 2]
results["top2_denominator_check"] = {"tokens": 6, "k": 2, "counts": counts.tolist(), "capacity": capacity, "overflow": overflow.tolist(), "assignments": 12, "overflow_rate": 4 / 12}

# Deliberately concentrate all six tokens on the same first expert.
for k in [1, 2]:
    torch.manual_seed(1512)
    model = MoEFFN(width=2, experts=2, top_k=k, hidden=3).eval()
    with torch.no_grad():
        model.router.weight.copy_(torch.tensor([[2.0, 0.0], [1.0, 0.0]]))
    x = torch.tensor([[1.0, 0.0]] * 6)
    received = [0, 0]
    handles = []
    for i, expert in enumerate(model.experts):
        def hook(module, args, i=i):
            received[i] += len(args[0])
        handles.append(expert.register_forward_pre_hook(hook))
    with torch.no_grad():
        output, auxiliary, chosen = model(x)
    for h in handles:
        h.remove()
    probabilities = model.router(x).softmax(-1)
    weights, selected_experts = probabilities.topk(k, dim=-1)
    if k > 1:
        weights = weights / weights.sum(-1, keepdim=True)
    with torch.no_grad():
        reference = torch.stack([sum(model.experts[int(selected_experts[row, slot])](x[row]) * weights[row, slot] for slot in range(k)) for row in range(6)])
    assert torch.allclose(output, reference, atol=1e-7, rtol=1e-6)
    expected = [6, 0] if k == 1 else [6, 6]
    assert received == expected and sum(received) == 6 * k
    assert chosen.tolist() == selected_experts.tolist()
    results["dropless_dispatch"].append({"k": k, "input_shape": list(x.shape), "chosen": chosen.tolist(), "expert_received_assignment_counts": received, "selected_assignment_denominator": 6 * k, "max_abs_reference_error": (output - reference).abs().max().item(), "model_updated": False})
print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
