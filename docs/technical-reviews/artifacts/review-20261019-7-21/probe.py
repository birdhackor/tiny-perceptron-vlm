"""核對目標對齊、遮罩、分母與有限代價的梯度。"""

import contextlib
import hashlib
import io
import json
import math
import platform
from pathlib import Path

import torch
import torch.nn.functional as F


def run_example(code):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "frozen_example", "exec"), namespace)
    return namespace, output.getvalue()


def grad_values(loss, parameters):
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if v is None else v for p, v in zip(parameters, values, strict=True)]


root = Path(__file__).resolve().parents[4]
source = root / "docs/course-review-20261019/freeze-01/sources/7.21.md"
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "42284d5529cb6e63a625b170862d9c6cbc798dd5a359266911c5b242f5220ccb"
code = raw.decode().split("```python\n", 1)[1].split("```", 1)[0]
assert code.count("weight = 0.5") == 1
normal, normal_output = run_example(code)
zero, zero_output = run_example(code.replace("weight = 0.5", "weight = 0.0"))
assert normal["next_scores"].shape == normal["later_scores"].shape == (4, 4)
assert normal["next_labels"].dtype == normal["later_labels"].dtype == torch.int64
assert normal["next_labels"].tolist() == [0, 1, 2, 3]
assert normal["later_labels"].tolist() == [1, 2, 3, -100]
assert not normal["next_scores"].requires_grad
assert not normal["later_scores"].requires_grad
assert not normal["total"].requires_grad
assert torch.equal(normal["next_scores"].softmax(-1), torch.full((4, 4), 0.25))
assert torch.equal(normal["later_scores"].softmax(-1), torch.full((4, 4), 0.25))

branches = {}
for name in ["next", "later"]:
    labels = normal[f"{name}_labels"]
    scores = normal[f"{name}_scores"]
    per_position = F.cross_entropy(scores, labels, ignore_index=-100, reduction="none")
    summed = F.cross_entropy(scores, labels, ignore_index=-100, reduction="sum")
    mean = F.cross_entropy(scores, labels, ignore_index=-100)
    count = int((labels != -100).sum())
    assert abs(mean.item() - math.log(4)) < 1e-6
    assert abs(summed.item() - count * math.log(4)) < 1e-6
    assert abs(mean.item() - summed.item() / count) < 1e-6
    assert zero[f"{name}_loss"].item() == mean.item()
    branches[name] = {
        "labels": labels.tolist(),
        "valid_targets": count,
        "per_position_loss": per_position.tolist(),
        "mean": mean.item(),
        "sum": summed.item(),
    }
assert abs(normal["total"].item() - 1.5 * math.log(4)) < 1e-6
assert abs(zero["total"].item() - math.log(4)) < 1e-6
perturbed = normal["later_scores"].clone()
perturbed[-1] = torch.tensor([100.0, -100.0, 30.0, -20.0])
perturbed.requires_grad_()
perturbed_loss = F.cross_entropy(perturbed, normal["later_labels"], ignore_index=-100)
perturbed_loss.backward()
assert perturbed_loss.item() == normal["later_loss"].item()
assert torch.count_nonzero(perturbed.grad[-1]).item() == 0

tokens = ["〈B〉", "雨", "停", "了", "〈E〉"]
alignment = []
for t in range(len(tokens) - 1):
    alignment.append({"read": tokens[t], "next": tokens[t + 1], "later": tokens[t + 2] if t + 2 < len(tokens) else None})
assert alignment[2]["later"] == "〈E〉"
assert alignment[3]["later"] is None

packed = ["〈B〉", "甲", "〈E〉", "〈B〉", "乙", "〈E〉", "PAD", "PAD"]
doc_ids = [0, 0, 0, 1, 1, 1, None, None]


def targets(distance):
    result = []
    for t in range(len(packed)):
        end = t + distance
        valid = (
            end < len(packed)
            and doc_ids[t] is not None
            and all(doc_ids[j] == doc_ids[t] for j in range(t, end + 1))
            and "〈E〉" not in packed[t:end]
        )
        result.append(packed[end] if valid else -100)
    return result


packed_near = targets(1)
packed_far = targets(2)
shifted_near = packed_near[1:] + [-100]
assert packed_far == ["〈E〉", -100, -100, "〈E〉", -100, -100, -100, -100]
assert shifted_near[2] == "乙" and packed_far[2] == -100
wrapped = torch.roll(normal["next_labels"], shifts=-1).tolist()
assert wrapped == [1, 2, 3, 0]
chat_tokens = ["BOS", "user", "Q", "turn_end", "assistant", "A", "turn_end"]
supervised_target_positions = {5, 6}
chat_masks = {
    str(d): [t + d in supervised_target_positions for t in range(len(chat_tokens))]
    for d in [1, 2]
}
assert chat_masks["1"] != chat_masks["2"]

torch.set_default_dtype(torch.float64)
trunk = torch.nn.Linear(3, 3, bias=False)
near = torch.nn.Linear(3, 3, bias=False)
far = torch.nn.Linear(3, 3, bias=False)
output = torch.nn.Linear(3, 4, bias=False)
modules = [trunk, near, far, output]
with torch.no_grad():
    for offset, module in enumerate(modules):
        module.weight.copy_(torch.sin(torch.arange(module.weight.numel()).reshape_as(module.weight) + offset + 1) / 5)
inputs = torch.arange(12).reshape(4, 3) / 10
hidden = torch.tanh(trunk(inputs))
near_loss = F.cross_entropy(output(torch.tanh(near(hidden))), normal["next_labels"])
far_loss = F.cross_entropy(output(torch.tanh(far(hidden))), normal["later_labels"], ignore_index=-100)
parameters = [m.weight for m in modules]
main_grad = grad_values(near_loss, parameters)
far_grad = grad_values(far_loss, parameters)
combined_grad = grad_values(near_loss + 0.5 * far_loss, parameters)
zero_grad = grad_values(near_loss + 0.0 * far_loss, parameters)
residual = max((g - a - 0.5 * b).abs().max().item() for g, a, b in zip(combined_grad, main_grad, far_grad, strict=True))
assert residual < 1e-14
assert all(torch.equal(g, a) for g, a in zip(zero_grad, main_grad, strict=True))
assert torch.count_nonzero(zero_grad[2]).item() == 0
assert main_grad[0].norm().item() > 0 and far_grad[0].norm().item() > 0
assert main_grad[1].norm().item() > 0 and far_grad[2].norm().item() > 0
assert all(g.norm().item() > 0 for g in combined_grad)

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-21/probe.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform()},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "normal_stdout": normal_output,
    "weight_zero_stdout": zero_output,
    "score_shape": list(normal["next_scores"].shape),
    "label_dtype": str(normal["next_labels"].dtype),
    "original_scores_require_grad": False,
    "uniform_probability": 0.25,
    "branches": branches,
    "totals": {"weight_half": normal["total"].item(), "weight_zero": zero["total"].item()},
    "derivation": {"ln_four": math.log(4), "three_ln_four": 3 * math.log(4), "four_ln_four": 4 * math.log(4), "one_and_half_ln_four": 1.5 * math.log(4)},
    "ignored_row_perturbation": {"loss": perturbed_loss.item(), "ignored_row_gradient": perturbed.grad[-1].tolist()},
    "alignment": alignment,
    "packed_example": {"tokens": packed, "document_ids": doc_ids, "near": packed_near, "far": packed_far, "naively_shifted_near": shifted_near, "scope": "每篇文件獨立，EOS 可作目標但不得跨越；PAD 不受監督。"},
    "wrapped_labels": wrapped,
    "assistant_target_masks": {"tokens": chat_tokens, "target_positions": sorted(supervised_target_positions), "distance_masks": chat_masks, "scope": "示範依目標位置監督助手回答與回答結尾的資料規則；不代表所有對話配方。"},
    "gradient_check": {"parameter_order": ["trunk", "near", "far", "output"], "combined_gradient_norms": [g.norm().item() for g in combined_grad], "far_loss_trunk_gradient_norm": far_grad[0].norm().item(), "main_loss_trunk_gradient_norm": main_grad[0].norm().item(), "gradient_linearity_max_residual": residual, "weight_zero_far_gradient_norm": zero_grad[2].norm().item(), "weight_zero_equals_main_gradient": True, "scope": "有限損失、確定性微型計算圖；未執行 optimizer 或訓練，不推論模型品質。"},
    "assertions": "全部通過",
}, ensure_ascii=False, indent=2))
