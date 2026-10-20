"""核對機率比、固定歷史紀錄、梯度與裁切的運算對象。"""

import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.posttraining import ppo_clipped_objective


def values(tensor):
    return tensor.detach().tolist()


old_probability = torch.tensor([0.2, 0.2], dtype=torch.float64, requires_grad=True)
new_probability = torch.tensor([0.4, 0.1], dtype=torch.float64, requires_grad=True)
old_log_probability = old_probability.log().detach()
ratio = (new_probability.log() - old_log_probability).exp()
ratio.sum().backward()

exercise = torch.tensor([0.3, 0.1], dtype=torch.float64)
exercise_ratio = (exercise.log() - old_log_probability).exp()
wrong_denominator = (exercise.log() - exercise.log().detach()).exp()

original_log = torch.tensor([-1.6], dtype=torch.float64, requires_grad=True)
detached_log = original_log.detach()
with torch.no_grad():
    original_log.add_(0.2)

tiny_logits = torch.tensor([-1000.0, 0.0], dtype=torch.float32)
small_log_ratio = torch.exp(torch.tensor(math.log(2.0), dtype=torch.float64))

clipping_cases = []
for ratio_value, advantage_value in [(1.3, 0.6), (2.0, 0.6), (1.3, -0.4)]:
    ratio_input = torch.tensor([ratio_value], dtype=torch.float64, requires_grad=True)
    old_log = torch.tensor([0.2], dtype=torch.float64).log()
    advantage = torch.tensor([advantage_value], dtype=torch.float64)
    result = ppo_clipped_objective(old_log + ratio_input.log(), old_log, advantage, clip_range=0.2)
    expected = min(ratio_value * advantage_value, min(max(ratio_value, 0.8), 1.2) * advantage_value)
    correct_gradient = torch.autograd.grad(result["surrogate"].sum(), ratio_input, retain_graph=True)[0]
    product = ratio_input * advantage
    direct_product_clipping = torch.minimum(product, product.clamp(0.8, 1.2))
    wrong_gradient = torch.autograd.grad(direct_product_clipping.sum(), ratio_input)[0]
    clipping_cases.append({
        "ratio": ratio_value,
        "advantage": advantage_value,
        "unclipped": values(result["unclipped"]),
        "clipped_ratio_times_advantage": values(result["clipped"]),
        "surrogate": values(result["surrogate"]),
        "hand_expected_surrogate": expected,
        "surrogate_gradient_wrt_ratio": values(correct_gradient),
        "direct_product_clipping_with_ratio_bounds": values(direct_product_clipping),
        "direct_product_clipping_gradient_wrt_ratio": values(wrong_gradient),
    })
    torch.testing.assert_close(result["surrogate"], torch.tensor([expected], dtype=torch.float64))

torch.testing.assert_close(ratio, torch.tensor([2.0, 0.5], dtype=torch.float64))
torch.testing.assert_close(exercise_ratio, torch.tensor([1.5, 0.5], dtype=torch.float64))
assert old_probability.grad is None

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-12/verify-ratio.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "dtype": "float64；小機率展示另用 float32",
        "platform": platform.platform(),
    },
    "implementation": {
        "path": "tiny_perceptron/posttraining.py",
        "sha256": hashlib.sha256((ROOT / "tiny_perceptron/posttraining.py").read_bytes()).hexdigest(),
    },
    "ratio": values(ratio),
    "direct_division": values(new_probability / old_probability),
    "rounded_original_ratio": [round(value, 4) for value in values(ratio)],
    "exercise_ratio": values(exercise_ratio),
    "rounded_exercise_ratio": [round(value, 4) for value in values(exercise_ratio)],
    "new_over_new": values(wrong_denominator),
    "old_probability_gradient": None if old_probability.grad is None else values(old_probability.grad),
    "new_probability_gradient": values(new_probability.grad),
    "detached_requires_grad": old_log_probability.requires_grad,
    "detach_shares_storage": original_log.data_ptr() == detached_log.data_ptr(),
    "detached_after_original_inplace_change": values(detached_log),
    "small_probability_example": {
        "logits": values(tiny_logits),
        "softmax_probabilities": values(tiny_logits.softmax(-1)),
        "log_softmax_values": values(tiny_logits.log_softmax(-1)),
        "ratio_from_log_difference_log_two": small_log_ratio.item(),
    },
    "advantage_product": 0.6 * 2.0,
    "clipping_cases": clipping_cases,
    "tolerance": "float64 算例 absolute tolerance 1e-12；頁面輸出按原程式四捨五入到小數四位。",
    "result": "機率比、練習與梯度核對相符；裁切 ratio 再乘優勢與直接按 ratio 的上下界裁切乘積，數值與梯度不同。",
}
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
