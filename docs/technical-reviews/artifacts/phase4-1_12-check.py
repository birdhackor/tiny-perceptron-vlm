"""Bounded CPU checks: exact fence, rate changes, no_grad removal, and report audit."""
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.simple import BigramLM

OUT = Path(__file__).resolve().parent
torch.set_default_device("cpu")
torch.set_num_threads(1)
assert torch.version.cuda is None
raw = (OUT / "phase4-1_12-original/fence-1.py").read_bytes()
original = raw.decode("utf-8")
results = {"environment": {"python": sys.version, "torch": str(torch.__version__),
            "torch_git_version": str(torch.version.git_version), "device": "cpu",
            "cuda_build": str(torch.version.cuda), "threads": torch.get_num_threads()},
           "original_fence_sha256": hashlib.sha256(raw).hexdigest(), "variants": []}
for name, code, expected_error, expected_w, expected_loss in (
    ("original", original, None, 1.4, 2.56),
    ("lr0", original.replace("0.1 * w.grad", "0.0 * w.grad"), "AssertionError", 1.0, 4.0),
    ("lr0_6", original.replace("0.1 * w.grad", "0.6 * w.grad"), None, 3.4, 0.16),
    ("lr2", original.replace("0.1 * w.grad", "2.0 * w.grad"), "AssertionError", 9.0, 36.0),
    ("without_no_grad", original.replace("with torch.no_grad():\n    w -=", "w -="), "RuntimeError", 1.0, None),
):
    variant_file = OUT / f"phase4-1_12-variant-{name}.py"
    variant_file.write_text(code)
    namespace = {"__name__": "__main__"}
    captured = io.StringIO()
    exception = None
    with contextlib.redirect_stdout(captured):
        try:
            exec(compile(code, str(variant_file), "exec"), namespace)
        except (AssertionError, RuntimeError) as error:
            exception = {"type": type(error).__name__, "message": str(error)}
    assert (exception["type"] if exception else None) == expected_error
    w = namespace["w"]
    before = namespace["before"]
    after = namespace.get("after")
    assert abs(w.item() - expected_w) <= 1e-6
    assert before.item() == 4.0 and w.grad.item() == -4.0
    assert w.is_leaf and w.requires_grad and torch.is_grad_enabled()
    if after is not None:
        assert abs(after.item() - expected_loss) <= 1e-6
        assert after.requires_grad and after.grad_fn is not None
    entry = {"name": name, "stdout": captured.getvalue(), "exception": exception,
             "w": w.item(), "before": before.item(), "grad": w.grad.item(),
             "after": after.item() if after is not None else None,
             "after_grad_fn": type(after.grad_fn).__name__ if after is not None else None,
             "parameter_leaf": w.is_leaf, "grad_mode_restored": torch.is_grad_enabled(),
             "tolerance": "abs <= 1e-6 against real-arithmetic w/loss; exact before=4 and grad=-4"}
    results["variants"].append(entry)
    print(json.dumps(entry, ensure_ascii=False))

# Only one SGD step on a two-target synthetic batch, not the 200-step recipe.
table = BigramLM(3)
with torch.no_grad():
    table.table.weight.zero_()
    table.table.weight[0, 1] = 2.0
inputs, answers = torch.tensor([[0], [2]]), torch.tensor([1, 0])
weights_before = table.table.weight.detach().clone()
loss_before = F.cross_entropy(table(inputs), answers)
loss_before.backward()
grad = table.table.weight.grad.detach().clone()
with torch.no_grad():
    table.table.weight -= 0.1 * grad
loss_after = F.cross_entropy(table(inputs), answers)
assert torch.allclose(table.table.weight, weights_before - 0.1 * grad, atol=1e-7, rtol=0)
assert not torch.equal(table.table.weight, weights_before)
assert loss_after.item() < loss_before.item()
results["one_table_sgd_step"] = {"input_shape": list(inputs.shape), "score_shape": [2, 3],
    "weight_shape": list(table.table.weight.shape), "targets": 2, "lr": 0.1,
    "before_nats_per_target": loss_before.item(), "after_nats_per_target": loss_after.item(),
    "grad": grad.tolist(), "weights_before": weights_before.tolist(),
    "weights_after": table.table.weight.detach().tolist(), "updates": 1}
print(json.dumps(results["one_table_sgd_step"], ensure_ascii=False))

# Audit immutable existing result and exact saved data; no historic training rerun.
report = json.loads((OUT / "phase4-1_12-original-experiment-result.json").read_text())
run = report["results"]["runs"]["bigram"]
data = report["results"]["data"]
denominators = {}
for split in ("train", "validation", "test"):
    path = OUT / "phase4-1_12-historical-data" / f"{split}.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert digest == data[split]["sha256"]
    assert len(rows) == data[split]["records"]
    denominators[split] = {"documents": len(rows),
                           "character_targets": sum(len(row["text"]) for row in rows),
                           "boundary_targets": len(rows),
                           "all_targets": sum(len(row["text"]) + 1 for row in rows),
                           "sha256": digest}
assert denominators["train"]["documents"] == 9 and run["steps"] == 200
assert report["step_scale"] == 1.0 and report["device"] == "cpu"
assert [entry["step"] for entry in run["history"]] == [1] + list(range(20, 201, 20))
assert run["history"][0]["loss_before_update"] == run["before_nll"]["train"]
assert run["history"][-1]["loss_before_update"] != run["after_nll_same_post_update_time"]["train"]
assert f'{run["before_nll"]["train"]:.4f}' == "3.7733"
assert f'{run["after_nll_same_post_update_time"]["train"]:.4f}' == "1.0568"
vocabulary_path = OUT / "phase4-1_12-historical-data/vocabulary.json"
vocabulary = json.loads(vocabulary_path.read_text())
saved_vocabulary = next(a for a in report["artifacts"] if a["path"] == "vocabulary.json")
assert hashlib.sha256(vocabulary_path.read_bytes()).hexdigest() == saved_vocabulary["sha256"]
assert (len(vocabulary) + 2) ** 2 == run["parameters"] == 289
results["historical_report_audit"] = {"revision": report["revision"],
    "original_environment": {k: report[k] for k in ("device", "seed", "torch_version", "python_version")},
    "before_nats_per_target": run["before_nll"]["train"],
    "after_nats_per_target": run["after_nll_same_post_update_time"]["train"],
    "last_pre_update_nats_per_target": run["history"][-1]["loss_before_update"],
    "rounded": ["3.7733", "1.0568"], "decimal_tolerance": "rounding error <= 0.00005",
    "denominators": denominators, "steps": run["steps"], "vocabulary_symbols": len(vocabulary) + 2,
    "parameters": run["parameters"], "scope": "recorded CPU report + hashed data audit, no training rerun"}
print(json.dumps(results["historical_report_audit"], ensure_ascii=False))
(OUT / "phase4-1_12-check-result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
