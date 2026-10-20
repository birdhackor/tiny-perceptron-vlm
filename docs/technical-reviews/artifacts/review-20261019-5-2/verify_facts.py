"""離線核對平均代價、忽略目標、梯度與字元／byte 切詞。"""

import ast
import hashlib
import inspect
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.course_experiments.common import _nll, text_examples
from tiny_perceptron.data import IGNORE, ByteTokenizer, CharTokenizer, pad_batch
from tiny_perceptron.model import loss_sum, masked_loss

torch.set_num_threads(1)
torch.set_default_device("cpu")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_case(labels):
    logits = torch.zeros(2, 4, 5, requires_grad=True)
    total, count = loss_sum(logits, labels)
    mean = masked_loss(logits, labels)
    direct_mean = F.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1))
    mean.backward()
    valid = labels != IGNORE
    expected_gradient = torch.zeros_like(logits)
    for row, column in valid.nonzero().tolist():
        expected_gradient[row, column] = 0.2 / count.item()
        expected_gradient[row, column, labels[row, column]] -= 1 / count.item()
    torch.testing.assert_close(logits.grad, expected_gradient, rtol=0, atol=1e-7)
    assert torch.equal(logits.grad.abs().sum(-1) > 0, valid)
    assert torch.equal(direct_mean, mean)
    assert abs(total.item() - count.item() * math.log(5)) < 1e-6
    assert abs(mean.item() - math.log(5)) < 1e-7
    assert (logits.grad[~valid] == 0).all().item()
    return {
        "labels": labels.tolist(),
        "labels_dtype": str(labels.dtype),
        "logits_dtype": str(logits.dtype),
        "logits_shape": list(logits.shape),
        "flattened_logits_shape": list(logits.reshape(-1, 5).shape),
        "flattened_labels_shape": list(labels.reshape(-1).shape),
        "probabilities": logits.detach().softmax(-1)[0, 0].tolist(),
        "effective_mask": valid.tolist(),
        "count": count.item(),
        "sum": total.item(),
        "mean": mean.item(),
        "expected_sum_float64": count.item() * math.log(5),
        "expected_mean_float64": math.log(5),
        "default_cross_entropy_mean": direct_mean.item(),
        "received_gradient": (logits.grad.abs().sum(-1) > 0).tolist(),
        "gradient": logits.grad.tolist(),
        "scalar_shapes": {"count": list(count.shape), "sum": list(total.shape), "mean": list(mean.shape)},
        "item_types": {"count": type(count.item()).__name__, "sum": type(total.item()).__name__, "mean": type(mean.item()).__name__},
        "grad_fn": type(mean.grad_fn).__name__,
        "leaf": logits.is_leaf,
    }


labels = torch.tensor([[1, 2, 3, 4], [1, -100, -100, -100]])
original = inspect_case(labels)
changed_labels = labels.clone()
changed_labels[1, 1] = 2
changed = inspect_case(changed_labels)
restored = inspect_case(labels)
assert restored == original

try:
    loss_sum(torch.zeros(2, 4, 5), torch.full_like(labels, IGNORE))
except ValueError as error:
    all_ignored = {"exception": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("全數忽略目標時應拒絕計算平均")

ordered = torch.arange(40).reshape(2, 4, 5)
assert ordered.reshape(-1, 5).tolist() == [ordered[b, t].tolist() for b in range(2) for t in range(4)]
assert labels.reshape(-1).tolist() == [labels[b, t].item() for b in range(2) for t in range(4)]

scores = torch.sin(torch.arange(40, dtype=torch.float64)).reshape(2, 4, 5)
whole = scores.clone().requires_grad_()
whole_mean = masked_loss(whole, labels)
whole_mean.backward()
split = scores.clone().requires_grad_()
parts = [loss_sum(split[row:row + 1], labels[row:row + 1]) for row in range(2)]
combined_mean = sum(value for value, _ in parts) / sum(count for _, count in parts)
combined_mean.backward()
torch.testing.assert_close(combined_mean, whole_mean, rtol=0, atol=1e-12)
torch.testing.assert_close(split.grad, whole.grad, rtol=0, atol=1e-12)
wrong_mean = sum(value.detach() / count for value, count in parts) / 2
assert abs(wrong_mean.item() - whole_mean.item()) > 0.01
per_target = F.cross_entropy(scores.reshape(-1, 5), labels.reshape(-1), reduction="none").reshape(2, 4)
torch.testing.assert_close(per_target.sum() / 5, whole_mean.detach(), rtol=0, atol=1e-12)

single = torch.tensor(0.0, requires_grad=True)
single_optimizer = torch.optim.SGD([single], lr=0.1)
((single - torch.tensor([1.0, 3.0])) ** 2).mean().backward()
single_optimizer.step()
sequential = torch.tensor(0.0, requires_grad=True)
sequential_optimizer = torch.optim.SGD([sequential], lr=0.1)
for target in (1.0, 3.0):
    sequential_optimizer.zero_grad()
    ((sequential - target) ** 2).backward()
    sequential_optimizer.step()
assert not torch.equal(single.detach(), sequential.detach())

class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, ids, valid=None):
        return {"logits": torch.zeros(*ids.shape, 5) + self.anchor}

examples = [(torch.tensor([1, 2, 3, 4]), labels[0]), (torch.tensor([1]), labels[1, :1])]
evaluation = _nll(ConstantModel(), examples, batch_size=2)
assert evaluation["effective_tokens"] == 5
assert abs(evaluation["nll"] - math.log(5)) < 1e-7

command = [str(ROOT / ".venv/bin/python"), "scripts/train.py", "--task", "text", "--device", "cpu", "--width", "8", "--layers", "1", "--batch-size", "2", "--max-length", "8"]
environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
dry_run = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30, check=True)
training_report = json.loads(dry_run.stdout.strip().splitlines()[-1])
assert training_report["mode"] == "dry-run-no-weight-update"
assert len(training_report["history"]) == 1
assert training_report["history"][0]["effective_tokens"] == 16

source = ROOT / "docs/course-review-20261019/freeze-01/sources/5.2.md"
body = source.read_text(encoding="utf-8")
fences = body.split("```python\n")[1:]
excerpt = fences[1].split("```", 1)[0]
excerpt_functions = {node.name: ast.dump(node) for node in ast.parse(excerpt).body if isinstance(node, ast.FunctionDef)}
for function in (loss_sum, masked_loss):
    assert excerpt_functions[function.__name__] == ast.dump(ast.parse(inspect.getsource(function)).body[0])

byte = ByteTokenizer()
char = CharTokenizer("狗看貓，。")
encoded_word = {"text": "狗", "char_ids": char.encode("狗"), "byte_ids": byte.encode("狗"), "utf8": list("狗".encode("utf-8"))}
assert len(encoded_word["char_ids"]) == 1
assert len(encoded_word["byte_ids"]) == 3
byte_examples = text_examples(["狗"])
assert len(byte_examples[0][1]) == 4

paths = [
    "tiny_perceptron/model.py", "tiny_perceptron/data.py", "scripts/train.py",
    "scripts/course_experiments/common.py", "scripts/course_experiments/text.py",
    "docs/course-review-20261019/freeze-01/sources/5.2.md",
    "docs/course-review-20261019/freeze-01/sources/5.8.md",
    "docs/course-review-20261019/freeze-01/sources/5.9.md",
]
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-2/verify_facts.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "threads": str(torch.get_num_threads()), "cuda_build": str(torch.version.cuda)},
    "source_hashes": {path: sha(ROOT / path) for path in paths},
    "original": original,
    "changed": changed,
    "restored_equals_original": restored == original,
    "all_ignored": all_ignored,
    "reshape_order_preserved": True,
    "excerpt_matches_library_ast": True,
    "unequal_parts": {"dtype": str(scores.dtype), "input_rule": "sin(arange(40)).reshape(2,4,5)", "labels": labels.tolist(), "counts": [count.item() for _, count in parts], "per_target_losses": per_target.tolist(), "whole_mean": whole_mean.item(), "combined_mean": combined_mean.item(), "wrong_mean_of_means": wrong_mean.item(), "max_gradient_difference": (split.grad - whole.grad).abs().max().item(), "target_weights": [[0.125] * 4, [0.5]], "correct_target_weight": 0.2},
    "optimizer_comparison": {"loss": "(w-y)^2", "initial_weight": 0.0, "targets": [1.0, 3.0], "optimizer": "SGD", "learning_rate": 0.1, "one_mean_update": single.item(), "two_sequential_updates": sequential.item()},
    "experiment_evaluation": evaluation,
    "training_dry_run": {"command_argv": command, "exit_code": dry_run.returncode, "stdout": dry_run.stdout, "stderr": dry_run.stderr, "weights_updated": False},
    "tokenization": {"single_character": encoded_word, "byte_text_example_input": byte_examples[0][0].tolist(), "byte_text_example_targets": byte_examples[0][1].tolist(), "effective_targets_including_eos": len(byte_examples[0][1])},
    "result": "全部算例與程式斷言通過；字元與 byte 切詞的反例已核實。",
}, ensure_ascii=False, indent=2))
