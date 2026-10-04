"""Independent bounded CPU checks for 11.1; no optimization or downloads."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import math
import sys
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F
from tiny_perceptron.multimodal import VisionEncoder, expand_modalities, scene

ROOT = Path(__file__).resolve().parents[6]
ART = Path(__file__).resolve().parent
torch.set_num_threads(1)
observations = {
    "environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu",
                    "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads()},
    "scope": "CPU FP32 deterministic demonstrations and CPU rescoring of existing GPU records; no model training or GPU replication",
}

# Execute the exact current first fenced block, not a transcription.
section = (ART / "section.raw.md").read_text()
block = section.split("```python\n", 1)[1].split("```", 1)[0]
namespace = {}
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    exec(compile(block, "11.1-current-first-code-block", "exec"), namespace)
assert printed.getvalue() == "輸出形狀 (1, 8)\n對輸入差異敏感 True\n"
projector, a, b, difference = [namespace[key] for key in ["projector", "a", "b", "difference"]]
delta = projector(a) - projector(b)
independent_norm = math.sqrt(sum(float(x) ** 2 for x in delta.detach().flatten()))
assert math.isclose(difference.item(), independent_norm, abs_tol=1e-7)
assert torch.allclose(delta, -projector.weight.sum(1).unsqueeze(0), atol=1e-7, rtol=1e-6)
observations["exact_original_block"] = {
    "stdout": printed.getvalue(), "seed": 0, "dtype": str(a.dtype),
    "inputs": {"a": a.tolist(), "b": b.tolist()},
    "weight": projector.weight.detach().tolist(), "bias": projector.bias.detach().tolist(),
    "outputs": {"a": projector(a).detach().tolist(), "b": projector(b).detach().tolist()},
    "difference_shape": list(difference.shape), "difference_numel": difference.numel(),
    "difference": difference.item(), "independent_sqrt_sum_squares": independent_norm,
    "item_python_type": type(difference.item()).__name__,
    "difference_requires_grad": difference.requires_grad,
    "optimization_steps": 0, "language_model": "absent", "labels": "absent",
}

# The hypothetical row in the prose has fixed, exactly representable values.
row = nn.Linear(4, 1)
with torch.no_grad():
    row.weight.copy_(torch.tensor([[1., 2., 0., 0.]]))
    row.bias.fill_(0.5)
assert row(a).item() == 0.5 and row(b).item() == 3.5
observations["specified_row"] = {"weights": [1, 2, 0, 0], "bias": 0.5,
                                  "zero_input_output": row(a).item(), "one_input_output": row(b).item()}

# Re-execute the same block with the stated insertion before difference.
modified = block.replace("difference =", "with torch.no_grad():\n    projector.weight.zero_()\ndifference =", 1)
zero_ns, zero_stdout = {}, io.StringIO()
with contextlib.redirect_stdout(zero_stdout):
    exec(compile(modified, "11.1-exercise-before-difference", "exec"), zero_ns)
zp = zero_ns["projector"]
assert zero_stdout.getvalue() == "輸出形狀 (1, 8)\n對輸入差異敏感 False\n"
assert torch.equal(zp.bias, projector.bias)
assert torch.equal(zp(a), zp(b)) and zero_ns["difference"].item() == 0.0
observations["zero_before_difference"] = {
    "stdout": zero_stdout.getvalue(), "seed": 0, "difference": zero_ns["difference"].item(),
    "outputs_equal": torch.equal(zp(a), zp(b)), "bias_unchanged": torch.equal(zp.bias, projector.bias),
    "weight_requires_grad": zp.weight.requires_grad, "weight_grad_fn": str(zp.weight.grad_fn),
    "optimization_steps": 0,
}

# A materialized value does not automatically re-run its producing operations.
old_value = difference.item()
with torch.no_grad():
    returned = projector.weight.zero_()
assert returned is projector.weight
new_value = (projector(a) - projector(b)).norm().item()
assert difference.item() == old_value and new_value == 0.0
observations["zero_after_difference"] = {"old_before_mutation": old_value,
    "old_after_mutation": difference.item(), "fresh_recomputation": new_value,
    "zero_returns_same_object": returned is projector.weight}

# Verify why no_grad is appropriate for this leaf Parameter modification.
leaf = nn.Linear(4, 8)
try:
    leaf.weight.zero_()
except RuntimeError as exc:
    observations["expected_leaf_inplace_failure_without_no_grad"] = str(exc)
else:
    raise AssertionError("Expected in-place update of leaf Parameter to fail in grad mode")

# Distinct vectors/logits need not give distinct or correct decoded answers.
bad_logits = torch.tensor([[0., 0., 10.], [1., 0., 10.]])
names = ["red", "blue", "green"]
predictions = [names[i] for i in bad_logits.argmax(-1).tolist()]
assert predictions == ["green", "green"]
assert (bad_logits[0] - bad_logits[1]).norm().item() == 1.0
observations["logit_counterexample"] = {"question": "what color?", "image_targets": ["red", "blue"],
    "answer_vocabulary": names, "logits": bad_logits.tolist(), "predictions": predictions,
    "logit_difference_norm": 1.0, "correct_answers": 0, "image_cases": 2,
    "scope": "constructed counterexample, not a trained VLM or performance measurement"}

# Algebraic information loss and an ignorant downstream module have counterexamples.
same = nn.Linear(4, 8)
identical_features = torch.zeros(2, 4)
mapped = same(identical_features)
assert torch.equal(mapped[0], mapped[1])
observations["lost_encoder_information"] = {"two_color_labels": ["red", "blue"],
    "encoded_features": identical_features.tolist(), "mapped_equal": True,
    "condition": "same prompt, deterministic encoder/projector/decoder, no distinguishing side information"}

# Direct prerequisites: feature gradient signal, width conversion, marker expansion.
torch.manual_seed(0)
encoder = VisionEncoder(width=8)
classifier = nn.Linear(8, 2)
images = torch.stack([scene("red", "square"), scene("blue", "circle")])
features = encoder(images)
logits = classifier(features.mean(1))
loss = F.cross_entropy(logits, torch.tensor([0, 1]))
loss.backward()
assert tuple(features.shape) == (2, 16, 8) and tuple(logits.shape) == (2, 2)
assert encoder.projection.weight.grad.norm().item() > 0
observations["prerequisite_gradient"] = {"seed": 0, "feature_shape": list(features.shape),
    "logit_shape": list(logits.shape), "ce": loss.item(),
    "projection_gradient_norm": encoder.projection.weight.grad.norm().item(), "optimization_steps": 0}

torch.manual_seed(0)
vision = torch.randn(1, 16, 8)
adapter = nn.Linear(8, 12)
converted = adapter(vision)
assert tuple(converted.shape) == (1, 16, 12) and tuple(adapter.weight.shape) == (12, 8)
observations["prerequisite_width"] = {"seed": 0, "input": list(vision.shape),
    "output": list(converted.shape), "weight": list(adapter.weight.shape)}

embedding = nn.Embedding(264, 8)
ids, labels = torch.tensor([1, 5, 4, 73, 2]), torch.tensor([-100, -100, -100, 73, 2])
modal_features = {5: torch.randn(3, 8)}
x, y = expand_modalities(ids, labels, embedding, modal_features, {5})
assert tuple(x.shape) == (1, 6, 8)
assert y.tolist() == [[-100, -100, -100, -100, 73, 2]]
assert torch.equal(x[0, 4], embedding(torch.tensor(4)))
observations["prerequisite_insertion"] = {"input_ids": ids.tolist(), "labels": labels.tolist(),
    "visual_positions": 3, "x_shape": list(x.shape), "y": y.tolist(),
    "first_answer_target_index": 4, "input_at_index_4": "assistant role embedding, id 4",
    "scope": "insertion/shift only, not answer semantics"}

# Rescore fixed run records independently; never represent this as a GPU rerun.
rescored = {}
for filename, branch in [("encoders.json", "vision"), ("real_modal.json", "fashion-mnist")]:
    path = ROOT / "docs/course-experiments/results" / filename
    raw = path.read_bytes()
    record = json.loads(raw)
    result = record["results"][branch]
    split_summary, scoring = {}, {}
    family_sets = []
    for split, contents in result["data"]["splits"].items():
        rows = contents["records"]
        assert len(rows) == contents["count"]
        families = {row["family"] for row in rows}
        assert len(families) == len(rows)
        family_sets.append(families)
        split_summary[split] = {"count": len(rows), "classes": dict(Counter(row["answer"] for row in rows)),
            "offsets": sorted({row["offset"] for row in rows}) if branch == "vision" else "not applicable",
            "families": sorted(families)}
    assert not any(family_sets[i] & family_sets[j] for i in range(3) for j in range(i + 1, 3))
    for stage in ["before", "validation", "test"]:
        reported = result[stage]
        rows = reported["samples"]
        prediction_key = "predicted" if branch == "vision" else "generated"
        correct = sum(row[prediction_key] == row["target"] for row in rows)
        assert correct == reported["correct"]
        assert len(rows) == reported.get("count", reported.get("examples"))
        assert correct / len(rows) == reported.get("accuracy", reported.get("exact_match"))
        score = {"correct": correct, "count": len(rows), "fraction": f"{correct}/{len(rows)}",
            "independently_compared_rows": [{"target": row["target"], "prediction": row[prediction_key],
                "match": row[prediction_key] == row["target"]} for row in rows]}
        if branch == "fashion-mnist":
            eos_count = sum(bool(row["generated_ids"]) and row["generated_ids"][-1] == 2 for row in rows)
            answer_targets = sum(len(row["target"].encode("utf-8")) + 1 for row in rows)
            assert eos_count / len(rows) == reported["eos_rate"]
            assert answer_targets == reported["effective_tokens"]
            score.update(eos_count=eos_count, answer_targets_including_eos=answer_targets)
        scoring[stage] = score
    train = result["training"]
    history = train["history"]
    assert len(history) == train["steps"] == 250
    assert [row["step"] for row in history] == list(range(1, 251))
    effective_targets = sum(row["effective_targets"] for row in history)
    assert effective_targets == train["effective_targets"]
    rescored[branch] = {"path": str(path.relative_to(ROOT)), "file_sha256": hashlib.sha256(raw).hexdigest(),
        "original_run": {key: record[key] for key in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "timing_scope"]},
        "data": split_summary, "score": scoring,
        "training": {key: train[key] for key in ["steps", "initial_loss", "final_loss", "parameters", "trainable_parameters", "seconds", "timing_scope", "weights_changed"]},
        "history_length": len(history), "summed_effective_targets": effective_targets,
        "history_effective_target_range": [min(row["effective_targets"] for row in history), max(row["effective_targets"] for row in history)],
        "scope": "independent arithmetic on original fixed predictions/history; no checkpoint execution, no GPU rerun, no claims about generalization outside this split"}

# Parameter count of the synthetic 10.5 run can also be recomputed from code.
count_model = nn.Sequential(VisionEncoder(width=16), nn.Linear(16, 6))
parameter_count = sum(parameter.numel() for parameter in count_model.parameters())
assert parameter_count == 1446
rescored["vision"]["independent_parameter_count"] = parameter_count
observations["prerequisite_record_rescoring"] = rescored
observations["assertions"] = "all passed"
(ART / "probe-results.json").write_text(json.dumps(observations, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(observations, ensure_ascii=False, indent=2))
