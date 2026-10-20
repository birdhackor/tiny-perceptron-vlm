"""獨立核對平方代價算例，以及更新、顯示精度與空目標的診斷。"""

import copy
import hashlib
import json
import math
import platform
import sys
from fractions import Fraction
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import loss_sum, masked_loss


def trajectory(lr):
    """以有理數逐步計算，再以誤差遞迴交叉核對。"""
    w = Fraction(1)
    rows = []
    for step in range(1, 4):
        gradient = 2 * (w - 3)
        w = w - lr * gradient
        expected_error = -2 * (1 - 2 * lr) ** step
        assert w - 3 == expected_error, "逐步更新與閉式解不一致"
        rows.append({"step": step, "gradient": str(gradient), "w": str(w),
                     "distance": str(abs(w - 3)), "loss": str((w - 3) ** 2)})
    return rows


torch.set_default_device("cpu")
w = torch.tensor(1.0, requires_grad=True)
((w - 3) ** 2).backward()
assert w.grad.item() == -4, "自動微分與手算梯度不一致"
single = []
for lr in (Fraction(1, 100), Fraction(1, 10), Fraction(2)):
    exact_w = 1 + 4 * lr
    exact_loss = (exact_w - 3) ** 2
    new = w.detach() - float(lr) * w.grad
    actual_loss = (new - 3).square().item()
    assert abs(new.item() - float(exact_w)) < 3e-7, "位置超過 float32 容忍差"
    assert abs(actual_loss - float(exact_loss)) < 3e-7, "代價超過 float32 容忍差"
    single.append({"lr": str(lr), "exact_w": str(exact_w), "exact_loss": str(exact_loss),
                   "float32_w": new.item(), "float32_loss": actual_loss})
assert w.item() == 1.0, "候選比較改動了共同起點"
paths = {str(lr): trajectory(lr) for lr in (Fraction(2), Fraction(1), Fraction(3, 4), Fraction(1, 2))}
assert [r["w"] for r in paths["2"]] == ["9", "-15", "57"], "發散位置不符"
assert [r["loss"] for r in paths["2"]] == ["36", "324", "2916"], "發散代價不符"
assert [r["w"] for r in paths["1"]] == ["5", "1", "5"], "等幅震盪位置不符"
assert [r["loss"] for r in paths["1"]] == ["4", "4", "4"], "等幅震盪代價不符"
assert [r["w"] for r in paths["3/4"]] == ["4", "5/2", "13/4"], "收斂震盪位置不符"
assert [r["loss"] for r in paths["3/4"]] == ["1", "1/4", "1/16"], "收斂震盪代價不符"
assert paths["1/2"][0]["w"] == "3", "一步到達最低點不符"

p = torch.nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.SGD([p], lr=0.1, momentum=0.9)
((p - 3) ** 2).backward()
before_step = p.item()
optimizer.step()
after_step = p.item()
assert before_step == 1 and math.isclose(after_step, 1.4, abs_tol=1e-6), "step 更新不符"
history = copy.deepcopy(optimizer.state_dict())
def next_update(restore_history):
    candidate = torch.nn.Parameter(torch.tensor(after_step))
    opt = torch.optim.SGD([candidate], lr=0.1, momentum=0.9)
    if restore_history:
        opt.load_state_dict(copy.deepcopy(history))
    ((candidate - 3) ** 2).backward()
    opt.step()
    return candidate.item()
restored_a, restored_b, fresh = next_update(True), next_update(True), next_update(False)
assert restored_a == restored_b and restored_a != fresh, "最佳化器歷史核對不符"
small = torch.tensor(1.0) - 0.00001 * torch.tensor(-4.0)
assert small.item() != 1 and f"{small.item():.4f}" == "1.0000", "顯示精度算例不符"
torch.manual_seed(17)
random_a = torch.randn(4)
torch.manual_seed(17)
random_b = torch.randn(4)
assert torch.equal(random_a, random_b), "同環境重設種子未重現亂數"

logits = torch.zeros(1, 2, 3, requires_grad=True)
labels = torch.tensor([[1, -100]])
total, count = loss_sum(logits, labels)
mean = masked_loss(logits, labels)
mean.backward()
received = logits.grad.abs().sum(dim=-1) > 0
assert count.item() == 1 and received.tolist() == [[True, False]], "有效目標或梯度遮罩不符"
try:
    masked_loss(logits.detach(), torch.full_like(labels, -100))
except ValueError as error:
    empty_error = str(error)
else:
    raise AssertionError("空目標未被拒絕")
nonfinite = torch.tensor([1.0, 0.0]) / 0.0
assert torch.isinf(nonfinite[0]) and torch.isnan(nonfinite[1]), "非有限值診斷不符"

print(json.dumps({
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": str(w.dtype)},
    "gradient": w.grad.item(), "initial_loss": (w - 3).square().item(), "unchanged_start": w.item(),
    "single_steps": single, "trajectories": paths,
    "diagnostics": {"before_optimizer_step": before_step, "after_optimizer_step": after_step,
                    "with_optimizer_history": restored_a, "repeated_history": restored_b, "fresh_optimizer": fresh,
                    "small_update": small.item(), "four_decimals": f"{small.item():.4f}",
                    "reseeded_random_equal": torch.equal(random_a, random_b),
                    "valid_count": count.item(), "gradient_mask": received.tolist(), "empty_target_error": empty_error,
                    "division_by_zero_isinf_isnan": [bool(torch.isinf(nonfinite[0])), bool(torch.isnan(nonfinite[1]))]},
    "input_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in
                     ["tiny_perceptron/model.py", "docs/course-review-20261019/freeze-01/sources/5.14.md"]},
    "result": "全部獨立算例與診斷核對通過；未量測真實模型的最佳學習率。"
}, ensure_ascii=False, indent=2))
