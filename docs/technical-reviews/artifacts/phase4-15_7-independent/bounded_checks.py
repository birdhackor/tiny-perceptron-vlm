"""Independent 15.7 CPU arithmetic, local gate variations, and existing raw-data checks.

No model weights are loaded and no model is trained. The only update is one
two-number hand-sized scores step, illustrating the direction in the lesson.
"""
import ast
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None
assert not torch.cuda.is_available()
torch.set_default_device("cpu")
checks = []


def record(name, value):
    checks.append({"name": name, "observed": value})


def gate_case(expert_output, renormalize):
    scores = torch.tensor([math.log(2), 0.0], dtype=torch.float64, requires_grad=True)
    prob = scores.softmax(-1)
    selected = prob.topk(1)
    p = selected.values[0]
    gate = p / p if renormalize else p
    output = gate * expert_output
    loss = output.square()
    loss.backward()
    expected_grad = [0.0, 0.0] if renormalize else [8 * expert_output**2 / 27, -8 * expert_output**2 / 27]
    expected_output = expert_output if renormalize else expert_output * 2 / 3
    torch.testing.assert_close(prob, torch.tensor([2 / 3, 1 / 3], dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(scores.grad, torch.tensor(expected_grad, dtype=torch.float64), atol=1e-12, rtol=0)
    assert math.isclose(output.item(), expected_output, abs_tol=1e-12)
    assert selected.indices.tolist() == [0] and not selected.indices.requires_grad
    return scores, {"expert_output": expert_output, "renormalize": renormalize, "prob": prob.tolist(), "output": output.item(), "loss": loss.item(), "gradient": scores.grad.tolist(), "selected_index": selected.indices.tolist(), "index_requires_grad": selected.indices.requires_grad}


for expert_output in (2.0, 1.0):
    for renormalize in (False, True):
        scores, values = gate_case(expert_output, renormalize)
        record("gate_case", values)
        if expert_output == 2.0 and not renormalize:
            before_loss = values["loss"]
            with torch.no_grad():
                scores -= 0.01 * scores.grad
                after_prob = scores.softmax(-1)
                after_loss = (2 * after_prob.topk(1).values[0]).square().item()
            assert scores[0] < math.log(2) and scores[1] > 0
            assert after_prob[0] < 2 / 3 and after_loss < before_loss
            record("one_two_score_update", {"step_size": 0.01, "scores": scores.tolist(), "prob": after_prob.tolist(), "before_loss": before_loss, "after_loss": after_loss})

# A locally fixed top-2 set has variable relative gates. This is not a claim
# that every task/expert configuration must have a nonzero task gradient.
scores = torch.tensor([math.log(2), 0.0, -3.0], dtype=torch.float64, requires_grad=True)
selected = scores.softmax(-1).topk(2)
gates = selected.values / selected.values.sum()
output = (gates * torch.tensor([2.0, 1.0], dtype=torch.float64)).sum()
output.square().backward()
torch.testing.assert_close(gates, torch.tensor([2 / 3, 1 / 3], dtype=torch.float64), atol=1e-12, rtol=0)
assert scores.grad[:2].abs().min() > 0.1
record("top2_normalized", {"indices": selected.indices.tolist(), "gates": gates.tolist(), "output": output.item(), "gradient": scores.grad.tolist()})

# Finite differences stay inside the top-1 region; an index switch is not
# represented by a derivative of integer indices.
epsilon = 1e-6
s = torch.tensor([math.log(2), 0.0], dtype=torch.float64)
finite_differences = []
for i in range(2):
    shift = torch.zeros(2, dtype=torch.float64)
    shift[i] = epsilon
    loss_plus = (2 * (s + shift).softmax(-1).topk(1).values[0]).square()
    loss_minus = (2 * (s - shift).softmax(-1).topk(1).values[0]).square()
    finite_differences.append(((loss_plus - loss_minus) / (2 * epsilon)).item())
torch.testing.assert_close(torch.tensor(finite_differences, dtype=torch.float64), torch.tensor([32 / 27, -32 / 27], dtype=torch.float64), atol=1e-8, rtol=0)
record("finite_difference", {"epsilon": epsilon, "gradient": finite_differences, "tolerance": 1e-8})
record("euclidean_norm", {"components": [3, 4], "squared_sum": 3**2 + 4**2, "norm": math.sqrt(3**2 + 4**2)})

# Read named raw measurements; never print/search result explanation fields.
raw_path = OUT / "moe-original-results.json"
raw = json.loads(raw_path.read_bytes())
variant = raw["results"]["variants"]["top1_aux0"]
before = variant["router_gradients_before"]
after = variant["router_gradients_after"]
assert before["effective_targets"] == after["effective_targets"] == 1004
assert [round(x, 5) for x in before["task_gradient_norms"]] == [0.00668, 0.00658]
assert [round(x, 5) for x in after["task_gradient_norms"]] == [0.02446, 0.01400]
assert all(x > 0 for x in before["task_gradient_norms"] + after["task_gradient_norms"])
training = variant["training"]
assert training["steps"] == training["optimizer_updates"] == training["requested_steps"] == 180
assert training["skipped_updates"] == 0 and training["auxiliary_weight"] == 0
record("existing_gradient_measurements", {"before": before, "after": after, "updates": training["optimizer_updates"], "auxiliary_coefficient": training["auxiliary_weight"], "auxiliary_scaled_before": [0.0 * x for x in before["unweighted_auxiliary_gradient_norms"]]})

# Reconstruct only the existing first-eight validation window denominator
# from local raw stories; no downloads, model forward, or training.
data_path = ROOT / "data/training/text-initial/tinystories-train-512.jsonl"
data_bytes = data_path.read_bytes()
assert hashlib.sha256(data_bytes).hexdigest() == raw["assets"][0]["files"][9]["sha256"]
records = [json.loads(line) for line in data_bytes.splitlines()]
for row in records:
    row["family"] = row["text_sha256"]
groups, seen = {}, set()
for row in records:
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
    if encoded in seen:
        continue
    seen.add(encoded)
    groups.setdefault(str(row["family"]), []).append(row)
keys = sorted(groups)
random.Random(raw["seed"]).shuffle(keys)
a = min(max(1, int(len(keys) * 0.8)), len(keys) - 2)
b = min(max(a + 1, int(len(keys) * 0.9)), len(keys) - 1)
validation = [row for key in keys[a:b] for row in groups[key]]
validation_sha = hashlib.sha256(json.dumps(validation, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
assert validation_sha == raw["results"]["dataset"]["validation"]["sha256"]
assert len(validation) == raw["results"]["dataset"]["validation"]["records"] == 51
windows = []
for row in validation:
    ids = [1] + [byte + 8 for byte in row["text"].encode("utf-8")] + [2]
    for start in range(0, len(ids) - 1, 128):
        chunk = ids[start : start + 129]
        windows.append({"record_id": row["id"], "start": start, "inputs": len(chunk) - 1, "targets": len(chunk) - 1, "first_target": chunk[1], "last_target": chunk[-1]})
first_eight = windows[:8]
assert len(first_eight) == 8 and max(w["inputs"] for w in first_eight) <= 128
assert sum(w["targets"] for w in first_eight) == 1004
record("existing_validation_denominator", {"local_data_path": str(data_path.relative_to(ROOT)), "local_data_sha256": hashlib.sha256(data_bytes).hexdigest(), "validation_records": len(validation), "validation_sha256": validation_sha, "first_eight_windows": first_eight, "effective_targets": sum(w["targets"] for w in first_eight), "max_input_positions": 128, "pad_targets_counted": False})

environment = {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads())}
result = {"status": "all bounded assertions passed", "environment": environment, "checks": checks, "scope": "Original lesson fence separately executed unchanged. Here: four two-score cases, one two-score update, one top-2 gradient case, finite differences, Euclidean arithmetic, selected existing raw measurements and existing first-eight raw-data windows. No existing model rerun or retraining."}
(OUT / "bounded-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
