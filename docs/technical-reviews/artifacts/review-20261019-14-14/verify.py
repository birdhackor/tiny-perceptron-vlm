"""核對終止深度目標、浮點展示及解題器的逐輪代價。"""

import hashlib
import json
import math
import platform
import random
import sys
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tiny_perceptron.lesson_products import training
from tiny_perceptron.lesson_products.model import Config, LoopedSolver, Tokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.manual_seed(0)
loss_by_depth = {1: 1.2, 2: 0.8, 3: 0.7, 4: 0.9, 5: 0.6}
global_state = random.getstate()
draws = [random.Random(42).choice([1, 3, 5]) for _ in range(8)]
assert draws == [5] * 8
assert global_state == random.getstate()
deep_loss = sum(loss_by_depth[r] for r in range(1, 6)) / 5
exact_mean = sum(map(Fraction, ["1.2", "0.8", "0.7", "0.9", "0.6"])) / 5
assert exact_mean == Fraction(21, 25)
assert math.isclose(deep_loss, 0.84, abs_tol=1e-15)
assert round(deep_loss, 4) == 0.84

w = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
states = []
state = torch.zeros((), dtype=torch.float64)
for _ in range(5):
    state = state + w
    states.append(state)
losses = [F.cross_entropy(torch.stack([-s, s])[None], torch.tensor([1])) for s in states]
probabilities = {1: 0.2, 3: 0.3, 5: 0.5}
weighted_loss = sum(probabilities[r] * losses[r - 1] for r in probabilities)
expected_gradient = torch.autograd.grad(weighted_loss, w, retain_graph=True)[0]
per_depth_gradients = {
    r: float(torch.autograd.grad(losses[r - 1], w, retain_graph=True)[0]) for r in probabilities
}
sum_of_gradients = sum(probabilities[r] * per_depth_gradients[r] for r in probabilities)
assert math.isclose(float(expected_gradient), sum_of_gradients, abs_tol=1e-12)
supervised_mean = torch.stack(losses).mean()
float_loss = supervised_mean.item()
try:
    torch.tensor(float_loss).backward()
except RuntimeError as error:
    detached_error = str(error)
else:
    raise AssertionError("重新建立的純數值tensor不應保有原始求導路徑")
assert supervised_mean.requires_grad and supervised_mean.grad_fn is not None

conflict_w = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
first_ce = F.cross_entropy(torch.stack([-conflict_w, conflict_w])[None], torch.tensor([1]))
second_ce = F.cross_entropy(torch.stack([conflict_w, -conflict_w])[None], torch.tensor([1]))
conflict_gradients = [
    float(torch.autograd.grad(loss, conflict_w, retain_graph=True)[0]) for loss in (first_ce, second_ce)
]
assert conflict_gradients == [-1.0, 1.0]

tokenizer = Tokenizer()
config = Config(vocab_size=len(tokenizer), width=16, heads=4, ff=32)
model = LoopedSolver(config)
sequence = torch.tensor([tokenizer.encode("兩盒追蹤;A=2;B=6;事件=交換;查詢=B")])
data = {
    "product": "b",
    "tokens": sequence[:, None],
    "labels": torch.tensor([2]),
    "groups": {("boxes", 1): torch.tensor([0])},
}
captured = {}
original_ce = F.cross_entropy


def capture_ce(logits, labels, **options):
    values = original_ce(logits, labels, **options)
    captured.update(
        logits_shape=list(logits.shape),
        labels=labels.tolist(),
        labels_dtype=str(labels.dtype),
        options=options,
        losses=values.detach().tolist(),
    )
    return values


counts = {"shared": 0, "answer": 0}


def count_call(name):
    def hook(module, inputs, output):
        counts[name] += 1
    return hook


handles = [model.shared.register_forward_hook(count_call("shared")), model.answer.register_forward_hook(count_call("answer"))]
optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
with patch.object(training.F, "cross_entropy", side_effect=capture_ce):
    step_result = training.step(
        model, optimizer, data, torch.Generator().manual_seed(5), 2, "multi", [1, 1, 1, 1, 1]
    )
for handle in handles:
    handle.remove()
expected_ce = sum(captured["losses"]) / len(captured["losses"])
assert math.isclose(step_result["loss"], expected_ce, rel_tol=1e-6, abs_tol=1e-6)
assert captured["logits_shape"] == [10, 19]
assert captured["labels"] == [2] * 10
assert captured["labels_dtype"] == "torch.int64"
assert captured["options"] == {"reduction": "none"}
assert counts == {"shared": 5, "answer": 5}
assert step_result["terminal_depth"] == 5 and step_result["block_sequence_work"] == 10
assert float(model.shared.qkv.weight.grad.abs().sum()) > 0

paths = [
    "tiny_perceptron/lesson_products/model.py",
    "tiny_perceptron/lesson_products/training.py",
    "tiny_perceptron/lesson_products/train.py",
    "docs/lesson-products/jobs/train-b-v1.json",
    "docs/lesson-products/results/initial-b-execution.json",
]
receipt = json.loads((ROOT / paths[-1]).read_bytes())
assert receipt["job"]["mode"] == "multi"
assert receipt["job"]["depth_weights"] == [1] * 5
assert receipt["source_hashes"][paths[1]] == digest(ROOT / paths[1])
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-14/verify.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
    "rng": {"independent_reinitializations": draws, "global_state_unchanged": True},
    "arithmetic": {"depths": list(loss_by_depth), "selected_loss": loss_by_depth[draws[0]], "raw_mean": deep_loss, "exact_mean": str(exact_mean), "rounded_mean": round(deep_loss, 4)},
    "sampled_objective": {"probabilities": probabilities, "per_depth_gradients": per_depth_gradients, "weighted_gradient": float(expected_gradient), "sum_of_gradients": sum_of_gradients},
    "autograd": {"mean_requires_grad": supervised_mean.requires_grad, "mean_grad_fn": type(supervised_mean.grad_fn).__name__, "float_type": type(float_loss).__name__, "detached_tensor_error": detached_error},
    "same_label_conflict": {"label": 1, "gradients_at_zero": conflict_gradients, "scope": "人工設定的不同讀出，僅證明共享參數目標可能衝突"},
    "product_step": {"model_config": {"width": 16, "heads": 4, "ff": 32}, "batch_size": 2, "optimizer_lr": 0.0, "captured": captured, "call_counts": counts, "expected_mean_ce": expected_ce, "result": step_result},
    "repository_files": {path: digest(ROOT / path) for path in paths},
    "product_receipt": {"mode": receipt["job"]["mode"], "depth_weights": receipt["job"]["depth_weights"], "training_source_matches_receipt": True},
    "result": "全部斷言通過；無模型權重更新"
}, ensure_ascii=False, indent=2, allow_nan=False))
