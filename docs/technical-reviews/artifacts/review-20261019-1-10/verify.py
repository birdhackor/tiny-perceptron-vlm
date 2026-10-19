"""核對單一路徑算例、微小增量、顯示操作與分支求導。"""

import hashlib
import json
import platform
from decimal import Decimal
from pathlib import Path

import torch


torch.set_default_device("cpu")
cases = []
for coefficient in (2, 3):
    w = torch.tensor(3.0, requires_grad=True)
    u = coefficient * w
    loss = u.square()
    detached = u.detach()
    before = w.item()
    loss.backward()
    expected = 2 * coefficient * coefficient * 3
    assert w.grad.item() == expected
    assert w.item() == before == 3
    assert not detached.requires_grad and detached.grad_fn is None
    assert u.requires_grad and u.grad_fn is not None
    assert detached.data_ptr() == u.data_ptr()
    cases.append({
        "coefficient": coefficient,
        "w": w.item(),
        "w_shape": list(w.shape),
        "w_dtype": str(w.dtype),
        "u": u.item(),
        "loss": loss.item(),
        "du_dw": coefficient,
        "dL_du": 2 * u.detach().item(),
        "dL_dw": w.grad.item(),
        "detached_requires_grad": detached.requires_grad,
        "original_u_requires_grad": u.requires_grad,
        "w_unchanged": w.item() == before,
        "grad_fn": {"u": type(u.grad_fn).__name__, "loss": type(loss.grad_fn).__name__},
    })

w0 = Decimal("3")
h = Decimal("0.001")
u0 = 2 * w0
du = 2 * h
exact_delta = (u0 + du) ** 2 - u0 ** 2
linear_delta = 2 * u0 * du
assert exact_delta == Decimal("0.024004")
assert linear_delta == Decimal("0.024")
assert exact_delta - linear_delta == du ** 2 == Decimal("0.000004")

x = torch.tensor(3.0, requires_grad=True)
left = 2 * x
right = 3 * x
combined = left + right
combined.backward()
assert x.grad.item() == 5

logits = torch.tensor([[0.0, 2.0, 0.0]], requires_grad=True)
probabilities = logits.softmax(dim=-1)
answer_cost = -probabilities[0, 1].log()
cross_entropy = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
assert torch.allclose(answer_cost, cross_entropy)

print(json.dumps({
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "platform": platform.platform(),
    },
    "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "cases": cases,
    "finite_increment_decimal": {
        "dw": str(h), "du": str(du),
        "exact_delta_L": str(exact_delta),
        "linear_delta_L": str(linear_delta),
        "difference_quotient": str(exact_delta / h),
        "derivative": "24",
        "linearization_error": str(exact_delta - linear_delta),
    },
    "branch_sum": {"w": x.item(), "left": left.item(), "right": right.item(), "output": combined.item(), "gradient": x.grad.item()},
    "answer_cost": {"logits": logits.detach().tolist(), "probabilities": probabilities.detach().tolist(), "negative_log_probability": answer_cost.item(), "cross_entropy": cross_entropy.item()},
    "result": "所有斷言通過",
}, ensure_ascii=False, indent=2))
