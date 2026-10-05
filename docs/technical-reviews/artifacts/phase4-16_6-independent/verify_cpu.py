"""Bounded independent 16.6 CPU and existing-measurement checks; no training run."""
import ast
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer, render_chat, pad_batch
from tiny_perceptron.model import loss_sum

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
result = {
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git": str(torch.version.git_version),
        "device": "cpu", "threads": "1",
    },
    "scope": "Original scalar fence and bounded variants; audit preserved raw measurements, no model weights loaded, no existing training or evaluation rerun.",
}

namespace = {}
exec(compile((OUT / "fence-1.py").read_bytes(), "16.6-original-fence", "exec"), namespace)
assert all(namespace[name].item() == 1.0 for name in ("whole", "accumulated", "wrong"))
assert namespace["x"][:1].tolist() == [1.0]
assert namespace["x"][1:].tolist() == [2.0, 3.0]
result["original_fence_no_update"] = {name: namespace[name].item() for name in ("whole", "accumulated", "wrong")}

def gradients(split):
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    whole = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    proper = whole.detach().clone().requires_grad_()
    wrong = whole.detach().clone().requires_grad_()
    (whole * x).square().mean().backward()
    for part in (x[:split], x[split:]):
        ((proper * part).square().sum() / len(x)).backward()
        ((wrong * part).square().mean() / 2).backward()
    got = [v.grad.item() for v in (whole, proper, wrong)]
    assert math.isclose(got[0], 28 / 3, abs_tol=1e-12)
    assert math.isclose(got[1], 28 / 3, abs_tol=1e-12)
    assert math.isclose(got[2], 7.5 if split == 1 else 11.5, abs_tol=1e-12)
    return got

result["split_1_2"] = gradients(1)
result["exercise_split_2_1"] = gradients(2)
equal_losses = torch.ones(11, dtype=torch.float64)
result["unequal_count_coincidence"] = {
    "counts": [1, 10],
    "token_mean": equal_losses.mean().item(),
    "mean_of_means": ((equal_losses[:1].mean() + equal_losses[1:].mean()) / 2).item(),
}
assert result["unequal_count_coincidence"]["token_mean"] == 1.0
assert result["unequal_count_coincidence"]["mean_of_means"] == 1.0
equal_parts = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
assert torch.equal(equal_parts.mean(), (equal_parts[:2].mean() + equal_parts[2:].mean()) / 2)

# Tokens are flattened over batch and sequence; the final logits axis is vocabulary.
examples = [render_chat([{"role": "user", "content": "q"}, {"role": "assistant", "content": answer}])
            for answer in ("OK", "answer", "x")]
x, labels, valid = pad_batch(examples)
torch.manual_seed(42)
logits = torch.randn(*labels.shape, 264, dtype=torch.float64, requires_grad=True)
total, count = loss_sum(logits, labels)
(total / count).backward()
expected_grad = logits.grad.detach().clone()
counts = [int((labels[:1] != -100).sum()), int((labels[1:] != -100).sum())]
separate = logits.detach().clone().requires_grad_()
for sl in (slice(None, 1), slice(1, None)):
    subtotal, _ = loss_sum(separate[sl], labels[sl])
    (subtotal / count).backward()
assert torch.allclose(expected_grad, separate.grad, atol=1e-12, rtol=0)
assert torch.count_nonzero(expected_grad[labels == -100]).item() == 0
result["valid_answer_token_variant"] = {
    "axes": ["batch", "sequence", "vocabulary"],
    "label_shape": list(labels.shape), "logits_shape": list(logits.shape),
    "valid_targets": int(count), "micro_counts": counts,
    "gradient_max_difference": float((expected_grad - separate.grad).abs().max()),
    "excluded_target_gradients_nonzero": int(torch.count_nonzero(expected_grad[labels == -100])),
}

# One illustrative update distinguishes accumulating at fixed weights from stepping per part.
def one_update(per_part=False):
    w = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
    optimizer = torch.optim.SGD([w], lr=.1)
    optimizer.zero_grad(set_to_none=True)
    for part in (torch.tensor([1.], dtype=torch.float64), torch.tensor([2., 3.], dtype=torch.float64)):
        ((w * part).square().sum() / 3).backward()
        if per_part:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    if not per_part:
        before_clip = w.grad.item()
        norm = torch.nn.utils.clip_grad_norm_([w], 100.)
        assert math.isclose(norm.item(), abs(before_clip), abs_tol=1e-12)
        optimizer.step()
    return w.item()
one, two = one_update(), one_update(True)
assert math.isclose(one, 1 - .1 * 28 / 3, abs_tol=1e-12)
assert not math.isclose(one, two, abs_tol=1e-12)
result["one_vs_two_demonstration_updates"] = {"one_step": one, "step_per_part": two}

# Existing evidence is read only at the selected measurement/sample/provenance pointers.
raw_path = OUT / "primary/efficiency-original.json"
raw = json.loads(raw_path.read_text())
pointers = ["/revision", "/device", "/seed", "/torch_version", "/python_version", "/gpu", "/step_scale",
            "/results/runtime", "/results/dataset", "/results/accumulation"]
measurements = {"original_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "original_gpu_environment": {k: raw[k] for k in ["device", "seed", "torch_version", "python_version", "gpu"]},
                "accumulation_probe": raw["results"]["accumulation"], "variants": {}}
probe = raw["results"]["accumulation"]
assert probe["effective_token_counts"] == [7, 13]
assert sum(probe["effective_token_counts"]) == probe["shared_denominator"] == 20
assert format(probe["gradient_max_error"], ".5e") == "2.08616e-07"
tok = ByteTokenizer()
keys = ["requested_steps", "all_requested_attempts_completed", "batch_size", "micro_batch_sizes", "gradient_clip_norm",
        "steps", "optimizer_updates", "skipped_updates", "effective_tokens", "warm_step_median_seconds",
        "memory_allocated_before_bytes", "peak_memory_allocated_bytes", "peak_additional_allocated_bytes"]
for label in ["ordinary", "accumulated"]:
    prefix = "/results/update_variants/" + label
    item = raw["results"]["update_variants"][label]
    tr = item["training"]
    test = item["heldout"]["test"]
    for key in keys: pointers.append(prefix + "/training/" + key)
    for key in ["nll", "nll_sum", "effective_tokens", "examples", "matches", "records", "exact_match", "eos_rate"]:
        pointers.append(prefix + "/heldout/test/" + key)
    for index, sample in enumerate(test["samples"]):
        for key in ["expected", "generated_ids", "exact", "eos"]:
            pointers.append(prefix + "/heldout/test/samples/" + str(index) + "/" + key)
    exact, ended, expected_count = 0, 0, 0
    for s in test["samples"]:
        generated = s["generated_ids"]
        is_ended = tok.eos_id in generated
        content = generated[:generated.index(tok.eos_id)] if is_ended else generated
        is_exact = content == tok.encode(s["expected"])
        assert is_exact == s["exact"] and is_ended == s["eos"]
        exact += is_exact
        ended += is_ended
        expected_count += len(tok.encode(s["expected"])) + 1
    assert exact == test["matches"] == 5 and len(test["samples"]) == test["records"] == 10
    assert ended == 10 and test["eos_rate"] == 1.0
    assert expected_count == test["effective_tokens"] == 69
    assert math.isclose(test["nll_sum"] / 69, test["nll"], abs_tol=1e-12)
    assert format(test["nll"], ".5f") == "0.41778"
    assert tr["optimizer_updates"] == tr["steps"] == tr["requested_steps"] == 40
    assert tr["skipped_updates"] == 0 and tr["all_requested_attempts_completed"] is True
    assert tr["effective_tokens"] == 2328 and tr["batch_size"] == 8
    assert tr["micro_batch_sizes"] == ([8] if label == "ordinary" else [3, 5])
    assert tr["peak_memory_allocated_bytes"] - tr["memory_allocated_before_bytes"] == tr["peak_additional_allocated_bytes"]
    measurements["variants"][label] = {
        "training": {key: tr[key] for key in keys},
        "test": {"matches_from_raw_tokens": exact, "records": len(test["samples"]), "eos": ended,
                 "expected_tokens_with_eos": expected_count, "nll_sum": test["nll_sum"], "nll": test["nll"]},
        "rounded_milliseconds": format(tr["warm_step_median_seconds"] * 1000, ".3f"),
        "rounded_memory_MiB": {key: format(tr[key] / 2**20, ".3f") for key in
            ["memory_allocated_before_bytes", "peak_memory_allocated_bytes", "peak_additional_allocated_bytes"]},
    }
error = raw["results"]["update_variants"]["accumulated"]["weight_max_error_from_ordinary_after_updates"]
assert format(error, ".5e") == "1.38730e-05"
pointers.append("/results/update_variants/accumulated/weight_max_error_from_ordinary_after_updates")
measurements["weight_max_error_from_original_measurement"] = error
measurements["inspected_json_pointers"] = pointers
measurements["limits"] = "Original saved aggregates for gradient/weight maxima, update tokens, median timings and allocator peaks. Per-step latency samples and checkpoint weights were not loaded or recreated; sample exact-match and test target denominator independently recomputed."
result["existing_measurement_audit"] = measurements

original = ast.parse((OUT / "primary/architecture-original.py").read_bytes())
current = ast.parse((ROOT / "scripts/course_experiments/architecture.py").read_bytes())
contract_names = ["_train", "_accumulation_probe", "_mechanism_examples", "_nll", "_forward", "_gradients", "_gradient_error", "_heldout", "run_efficiency"]
result["original_current_method_ast_equal"] = {}
for name in contract_names:
    old = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
    now = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name)
    equal = ast.dump(old, include_attributes=False) == ast.dump(now, include_attributes=False)
    assert equal, name
    result["original_current_method_ast_equal"][name] = equal

for path in ["scripts/course_experiments/common.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py"]:
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == raw["code_sha256"][path]

(OUT / "cpu-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"split_1_2": result["split_1_2"], "exercise_split_2_1": result["exercise_split_2_1"],
                  "tokens": result["valid_answer_token_variant"], "one_vs_two_steps": result["one_vs_two_demonstration_updates"],
                  "raw_variants": measurements["variants"], "contract_AST_equal": result["original_current_method_ast_equal"]}, ensure_ascii=False, indent=2))
