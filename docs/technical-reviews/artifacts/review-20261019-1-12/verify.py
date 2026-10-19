"""獨立核對純量更新、求導模式與固定接字表實驗。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.course_experiments.text import _simple_examples
from tiny_perceptron.data import split_documents, toy_documents
from tiny_perceptron.simple import BigramLM
from tiny_perceptron.training import seed_everything


torch.set_num_threads(1)
frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/1.12.md"
code = re.findall(r"```python\n(.*?)```", frozen.read_text(), flags=re.S)[0]
calculations = []
for rate in ("0.1", "0.6", "2"):
    initial, target = Decimal(1), Decimal(3)
    gradient = 2 * (initial - target)
    updated = initial - Decimal(rate) * gradient
    calculations.append({"learning_rate": rate, "gradient": str(gradient),
                         "updated": str(updated), "loss": str((updated - target) ** 2)})

cases = []
for rate in ("0.1", "0.6", "2"):
    namespace, output = {}, io.StringIO()
    error = None
    with contextlib.redirect_stdout(output):
        try:
            exec(code.replace("0.1 * w.grad", f"{rate} * w.grad"), namespace)
        except AssertionError as exc:
            error = type(exc).__name__
    cases.append({"learning_rate": rate, "stdout": output.getvalue(), "exception": error,
                  "w": namespace["w"].item(), "before": namespace["before"].item(),
                  "after": namespace["after"].item(), "gradient": namespace["w"].grad.item()})
assert [case["exception"] for case in cases] == [None, None, "AssertionError"]
for case, exact in zip(cases, calculations, strict=True):
    assert abs(case["w"] - float(exact["updated"])) < 1e-6
    assert abs(case["after"] - float(exact["loss"])) < 1e-6
    assert case["before"] == 4 and case["gradient"] == -4

w = torch.tensor(1.0, requires_grad=True)
before = (w - 3).square()
before.backward()
before_update = {"w": w.item(), "gradient": w.grad.item(),
                 "distinct_storage": w.data_ptr() != w.grad.data_ptr()}
identity, storage = id(w), w.data_ptr()
with torch.no_grad():
    w -= 0.1 * w.grad
after = (w - 3).square()
autograd = {"after_backward": before_update, "same_tensor_object": id(w) == identity,
            "same_parameter_storage": w.data_ptr() == storage, "leaf": w.is_leaf,
            "parameter_requires_grad": w.requires_grad, "parameter_grad_fn": str(w.grad_fn),
            "gradient_after_update": w.grad.item(), "old_loss_after_update": before.item(),
            "new_loss_requires_grad": after.requires_grad, "new_loss_grad_fn": type(after.grad_fn).__name__}
assert before_update == {"w": 1.0, "gradient": -4.0, "distinct_storage": True}
assert autograd["same_tensor_object"] and autograd["same_parameter_storage"]
assert w.is_leaf and w.grad_fn is None and w.requires_grad and after.requires_grad
assert before.item() == 4 and w.grad.item() == -4

without_guard = code.replace("with torch.no_grad():\n    w -= 0.1 * w.grad", "w -= 0.1 * w.grad")
error_output, error_namespace = io.StringIO(), {}
with contextlib.redirect_stdout(error_output):
    try:
        exec(without_guard, error_namespace)
    except RuntimeError as exc:
        unguarded = {"exception": type(exc).__name__, "message": str(exc), "stdout": error_output.getvalue()}
    else:
        raise AssertionError("一般求導模式下的葉參數原地更新應拒絕")

parts = split_documents(toy_documents(), seed=42)
assert [len(parts[name]) for name in ("train", "validation", "test")] == [9, 1, 2]
assert not (set(parts["train"]) & set(parts["validation"]))
assert not (set(parts["train"]) & set(parts["test"]))
assert not (set(parts["validation"]) & set(parts["test"]))
vocabulary = {char: index + 2 for index, char in enumerate(sorted(set("".join(parts["train"]))))}
batches = {name: _simple_examples(documents, vocabulary, 1) for name, documents in parts.items()}
split_details = {}
for name, documents in parts.items():
    rows = [{"text": text, "family": text} for text in documents]
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    x, y = batches[name]
    split_details[name] = {"documents": documents, "records": len(documents),
                           "sha256": hashlib.sha256(raw).hexdigest(), "effective_targets": y.numel(),
                           "input_shape": list(x.shape), "label_shape": list(y.shape),
                           "input_dtype": str(x.dtype), "label_dtype": str(y.dtype),
                           "ignored_targets": int((y == -100).sum())}

seed_everything(42)
model = BigramLM(len(vocabulary) + 2)
x, y = batches["train"]
with torch.no_grad():
    initial_losses = {name: float(F.cross_entropy(model(a), b)) for name, (a, b) in batches.items()}
    initial_weights = model.table.weight.clone()
    initial_probabilities = model(x).softmax(-1)

# 另用基本梯度下降核對接字分數也能按此規則更新。
seed_everything(42)
plain = BigramLM(len(vocabulary) + 2)
plain_loss = F.cross_entropy(plain(x), y)
plain_loss.backward()
with torch.no_grad():
    plain.table.weight -= 0.1 * plain.table.weight.grad
    plain_after = float(F.cross_entropy(plain(x), y))
assert plain_after < initial_losses["train"]

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
history = []
for step in range(200):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(norm)
    optimizer.step()
    if step in (0, 199):
        history.append({"completed_updates": step + 1, "loss_before_update": float(loss.detach())})
with torch.no_grad():
    final_losses = {name: float(F.cross_entropy(model(a), b)) for name, (a, b) in batches.items()}
    per_target = -model(x).log_softmax(-1).gather(1, y[:, None]).squeeze(1)
    manual_mean = float(per_target.sum() / y.numel())
    probability_change = float((model(x).softmax(-1) - initial_probabilities).abs().max())
    parameter_change = float((model.table.weight - initial_weights).abs().max())
assert abs(manual_mean - final_losses["train"]) < 1e-6
assert probability_change > 0 and parameter_change > 0
record_path = ROOT / "docs/course-experiments/results/simple_models.json"
record = json.loads(record_path.read_text())
historical = record["results"]["runs"]["bigram"]
for name in parts:
    assert split_details[name]["sha256"] == record["results"]["data"][name]["sha256"]
    assert abs(initial_losses[name] - historical["before_nll"][name]) < 1e-5
    assert abs(final_losses[name] - historical["after_nll_same_post_update_time"][name]) < 1e-5
assert format(initial_losses["train"], ".4f") == "3.7733"
assert format(final_losses["train"], ".4f") == "1.0568"
state = optimizer.state[model.table.weight]
assert state["step"].item() == 200

paths = ["tiny_perceptron/simple.py", "tiny_perceptron/data.py", "tiny_perceptron/training.py",
         "scripts/course_experiments/text.py", "scripts/train_simple.py",
         "docs/course-experiments/results/simple_models.json",
         "docs/course-review-20261019/figure-renders.json",
         "course/figures/rewrite-01-update.svg"]
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-12/verify.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "torch_git_version": torch.version.git_version, "platform": platform.platform(),
                    "device": "cpu", "dtype": "torch.float32", "threads": torch.get_num_threads()},
    "decimal_calculations": calculations, "section_variants": cases,
    "autograd": autograd, "unguarded_update": unguarded,
    "bigram": {"seed": 42, "vocabulary_size": len(vocabulary) + 2, "context": 1,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "splits": split_details, "optimizer": "torch.optim.AdamW", "learning_rate": 0.01,
               "betas": list(optimizer.defaults["betas"]), "eps": optimizer.defaults["eps"],
               "weight_decay": optimizer.defaults["weight_decay"], "clip_grad_norm": 1.0,
               "updates": 200, "optimizer_state_step": state["step"].item(),
               "optimizer_state_keys": sorted(state), "before_nll": initial_losses,
               "after_nll": final_losses, "history": history, "manual_nll_mean": manual_mean,
               "nll_sum": float(per_target.sum()), "max_parameter_change": parameter_change,
               "max_probability_change": probability_change,
               "plain_gradient_descent_after_one_update": plain_after,
               "historical_record_environment": {key: record[key] for key in ("python_version", "torch_version", "seed", "device")},
               "historical_before_nll": historical["before_nll"],
               "historical_after_nll": historical["after_nll_same_post_update_time"],
               "historical_tolerance": 1e-5},
    "file_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
    "result": "所有斷言符合預期；大步長的教材斷言與一般求導模式的原地更新各自觸發預期例外。",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
