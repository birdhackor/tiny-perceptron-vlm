"""Fresh bounded CPU verification for lesson 2.5; no checkpoint or data downloads."""
import hashlib
import json
import math
import platform
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import split_documents, toy_documents
from tiny_perceptron.simple import ContextMLP

OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)
assert str(torch.__version__) == "2.14.1+cpu"
environment = {
    "python": platform.python_version(), "torch": str(torch.__version__),
    "device": "cpu", "threads": str(torch.get_num_threads()),
    "platform": platform.platform(), "default_dtype": str(torch.get_default_dtype()),
    "cuda_available": str(torch.cuda.is_available()),
}
print("ENVIRONMENT", json.dumps(environment))
(OUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

# Execute the exact source block separately from the expanded probes.
raw = (OUT / "source.initial.md").read_text()
code = raw.split("```python\n", 1)[1].split("```", 1)[0]
print("EXACT LESSON BLOCK")
exec(compile(code, "course/chapters/02.md#2.5", "exec"))
examples = [("紅色物體是", "圓"), ("藍色物體是", "方")]
slices = {str(c): [[p[-c:], a] for p, a in examples] for c in [1, 3, 4, 5]}
assert slices == {
    "1": [["是", "圓"], ["是", "方"]],
    "3": [["物體是", "圓"], ["物體是", "方"]],
    "4": [["色物體是", "圓"], ["色物體是", "方"]],
    "5": [["紅色物體是", "圓"], ["藍色物體是", "方"]],
}
long = ["紅色的小小物體是", "藍色的小小物體是"]
assert all(len(p) == 8 for p in long)
exercise = {str(c): [p[-c:] for p in long] for c in [7, 8]}
assert exercise["7"] == ["色的小小物體是"] * 2
assert exercise["8"] == long
print("SLICES", json.dumps(slices, ensure_ascii=False))
print("EXERCISE", json.dumps(exercise, ensure_ascii=False))

record_path = ROOT / "docs/course-experiments/results/simple_models.json"
record = json.loads(record_path.read_text())
documents = toy_documents()
parts = split_documents(documents, seed=42)
assert len(documents) == len(set(documents)) == 12
assert {name: len(values) for name, values in parts.items()} == {"train": 9, "validation": 1, "test": 2}
assert parts["validation"] == ["顏色=紅；形狀=圓。"]
split_fingerprints = {}
for name, texts in parts.items():
    rows = [{"text": text, "family": text} for text in texts]
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    assert digest == record["results"]["data"][name]["sha256"]
    split_fingerprints[name] = digest
vocabulary = {c: i + 2 for i, c in enumerate(sorted(set("".join(parts["train"]))))}
assert len(vocabulary) + 2 == 17
denominators = {name: sum(len(text) + 1 for text in texts) for name, texts in parts.items()}
assert denominators == {"train": 103, "validation": 11, "test": 22}
print("SPLITS", json.dumps(parts, ensure_ascii=False))
print("VOCABULARY", json.dumps(vocabulary, ensure_ascii=False))
print("DENOMINATORS INCLUDING ONE EOS PER DOCUMENT", json.dumps(denominators))

def batch(texts, context):
    rows, labels = [], []
    for text in texts:
        ids = [vocabulary.get(c, 1) for c in text] + [0]
        padded = [0] * context + ids
        for position, answer in enumerate(ids):
            rows.append(padded[position:position + context])
            labels.append(answer)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def evaluate(model, batches):
    result = {}
    with torch.no_grad():
        for name, (x, y) in batches.items():
            logits = model(x)
            nll = float(F.cross_entropy(logits, y))
            # Independent scalar log-sum-exp, calculated from emitted float32 logits
            # using Python double arithmetic; does not call log_softmax or CE.
            scalar_losses = []
            for row, target in zip(logits.tolist(), y.tolist(), strict=True):
                maximum = max(row)
                log_z = maximum + math.log(math.fsum(math.exp(z - maximum) for z in row))
                scalar_losses.append(log_z - row[target])
            manual = math.fsum(scalar_losses) / len(scalar_losses)
            assert abs(manual - nll) < 2e-6
            result[name] = {
                "nll": nll, "manual_nll_python_float64": manual,
                "nll_sum_python_float64": math.fsum(scalar_losses),
                "target_probability_geometric_mean": math.exp(-manual),
                "denominator": len(scalar_losses),
                "per_target_nll_python_float64": scalar_losses,
            }
    return result

runs = {}
for context in [1, 3, 5]:
    random.seed(42)
    torch.manual_seed(42)
    model = ContextMLP(17, context=context, width=16).cpu()
    batches = {name: batch(texts, context) for name, texts in parts.items()}
    parameters = sum(p.numel() for p in model.parameters())
    manual_parameters = 17 * 16 + (context * 16 * 16 + 16) + (16 * 17 + 17)
    assert parameters == manual_parameters == {1: 833, 3: 1345, 5: 1857}[context]
    before = evaluate(model, batches)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    assert optimizer.defaults["weight_decay"] == 0.01
    x, y = batches["train"]
    history = []
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        if step in [0, 199]:
            history.append({"step": step + 1, "loss_before_update": float(loss.detach()), "norm_before_clip": float(norm)})
    after = evaluate(model, batches)
    reference = record["results"]["runs"][f"mlp{context}"]
    assert reference["parameters"] == parameters and reference["steps"] == 200
    differences = {}
    for name in batches:
        expected_before = reference["before_nll"][name]
        expected_after = reference["after_nll_same_post_update_time"][name]
        differences[name] = {"before": before[name]["nll"] - expected_before, "after": after[name]["nll"] - expected_after}
        assert abs(differences[name]["before"]) < 2e-6
        assert abs(differences[name]["after"]) < 2e-6
    same = model(torch.tensor([[0] * context, [0] * context]))
    assert torch.equal(same[0], same[1])
    runs[str(context)] = {
        "parameters": parameters, "parameter_derivation": manual_parameters,
        "width": 16, "seed": 42, "updates": 200,
        "model": "embedding(17,16) -> flatten -> Linear(C*16,16) -> tanh -> Linear(16,17)",
        "optimizer_defaults": optimizer.defaults,
        "gradient_norm_clip": 1.0, "before": before, "after": after,
        "history_endpoints": history, "difference_from_existing_record": differences,
        "identical_input_output_equal": True,
        "input_shapes": {n: list(xx.shape) for n, (xx, yy) in batches.items()},
        "input_dtype": str(x.dtype), "parameter_dtype": str(next(model.parameters()).dtype),
    }
    print("BOUNDED CPU RERUN", json.dumps({"context": context, "parameters": parameters, "updates": 200,
        "after": {n: r["nll"] for n, r in after.items()}, "reference_difference": differences}))

result = {
    "environment": environment, "scope": "Own full rerun of only three tiny MLPs, 200 updates each, CPU; no general-quality, GPU, timing or scaling claim.",
    "existing_record_path": str(record_path.relative_to(ROOT)),
    "existing_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
    "existing_record_environment": {k: record[k] for k in ["revision", "device", "seed", "torch_version", "python_version", "step_scale", "evidence_status", "timing_scope"]},
    "slices": slices, "exercise": exercise, "documents": parts,
    "split_fingerprints": split_fingerprints, "vocabulary": vocabulary,
    "denominators": denominators, "runs": runs,
    "repository_code_sha256": {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in [
        Path("tiny_perceptron/simple.py"), Path("tiny_perceptron/data.py"),
        Path("scripts/course_experiments/text.py"), Path("scripts/course_experiments/common.py"),
        Path("scripts/course_experiments/run.py"), Path("tiny_perceptron/training.py")]},
}
(OUT / "probe-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print("ALL ASSERTIONS PASSED")
