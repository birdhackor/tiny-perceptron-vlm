"""核對殘差算例、零分支練習、微小擾動與導數的適用範圍。"""

import json
import platform
import sys
from decimal import Decimal

import torch


def example(scale):
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    before = x.detach().clone()
    correction = scale * x
    y = x + correction
    total = y.sum()
    total.backward()
    assert torch.equal(x, before)
    assert torch.equal(y.detach(), (1 + scale) * before)
    assert torch.equal(x.grad, torch.full_like(x, 1 + scale))
    jacobian = torch.autograd.functional.jacobian(lambda z: z + scale * z, before)
    assert torch.equal(jacobian, (1 + scale) * torch.eye(2))
    return {
        "input": before.tolist(),
        "input_after_backward": x.detach().tolist(),
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "correction": correction.detach().tolist(),
        "output": y.detach().tolist(),
        "sum_shape": list(total.shape),
        "sum_value": total.item(),
        "gradient": x.grad.tolist(),
        "jacobian": jacobian.tolist(),
        "gradient_assertion": True,
    }


torch.set_default_device("cpu")
twice = example(2)
zero = example(0)
cancel = example(-1)

base = torch.tensor([1.0, 2.0], dtype=torch.float64)
perturbations = []
for index in range(2):
    changed = base.clone()
    changed[index] += 0.001
    direct = changed - base
    branch = 2 * changed - 2 * base
    combined = (changed + 2 * changed) - (base + 2 * base)
    expected = torch.zeros_like(base)
    expected[index] = 0.003
    assert torch.allclose(combined, expected, atol=1e-15, rtol=0)
    perturbations.append({
        "feature": index,
        "direct_change": direct.tolist(),
        "branch_change": branch.tolist(),
        "output_change": combined.tolist(),
        "absolute_tolerance": 1e-15,
    })

matrix = torch.tensor([[2.0, 1.0], [-3.0, 4.0]], dtype=torch.float64)
jacobian = torch.autograd.functional.jacobian(lambda z: z + matrix @ z, base)
expected_jacobian = torch.eye(2, dtype=torch.float64) + matrix
assert torch.equal(jacobian, expected_jacobian)

# 分支值很小不代表分支導數很小；直接路徑的單位項仍存在。
small_input = torch.zeros(2, dtype=torch.float64, requires_grad=True)
small_branch = -small_input + 0.000001
small_output = small_input + small_branch
small_output.sum().backward()
assert torch.equal(small_input.grad, torch.zeros_like(small_input))

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-4-2/verify-residual.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "platform": platform.platform(),
        "device": "cpu",
    },
    "twice": twice,
    "zero": zero,
    "decimal_derivation": {
        "input_change": str(Decimal("0.001")),
        "direct_change": str(Decimal("0.001")),
        "branch_change": str(2 * Decimal("0.001")),
        "output_change": str(3 * Decimal("0.001")),
        "sensitivity": str(3 * Decimal("0.001") / Decimal("0.001")),
    },
    "float64_perturbations": perturbations,
    "mixed_feature_jacobian": {
        "branch_matrix": matrix.tolist(),
        "observed": jacobian.tolist(),
        "expected": expected_jacobian.tolist(),
    },
    "limits": {
        "cancelling_branch": cancel,
        "small_branch_value": small_branch.detach().tolist(),
        "small_branch_output": small_output.detach().tolist(),
        "small_branch_total_gradient": small_input.grad.tolist(),
        "interpretation": "主路保有單位導數項，但總導數可以被分支抵消；殘差結構也不保證可逆或訓練成功。",
    },
    "result": "全部斷言通過",
}, ensure_ascii=False, indent=2))
