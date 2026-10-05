"""Independent bounded audit: arithmetic, construction, raw aggregation, no model training."""
from pathlib import Path
import ast
import hashlib
import json
import math
import random
import sys
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[4]
PROOF = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.model import TinyLM, ModelConfig, loss_sum
from tiny_perceptron.data import ByteTokenizer, load_jsonl, IGNORE
from scripts.course_experiments.common import split_records, records_sha256, text_examples

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
raw = json.loads((PROOF / "raw/moe.json").read_bytes())
historical = PROOF / "code/scripts/course_experiments/architecture.py"
assert hashlib.sha256(historical.read_bytes()).hexdigest() == raw["code_sha256"]["scripts/course_experiments/architecture.py"]
for file in ["tiny_perceptron/model.py", "tiny_perceptron/modern.py", "tiny_perceptron/data.py", "tiny_perceptron/attention.py", "scripts/course_experiments/common.py"]:
    assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == raw["code_sha256"][file]

# Execute only the AST-located counting methods; no experiment runner is imported.
from tiny_perceptron.modern import MoEFFN
tree = ast.parse(historical.read_bytes())
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_parameter_budget", "_matched_width"}]
ns = {"TinyLM": TinyLM, "MoEFFN": MoEFFN, "replace": replace}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(historical), "exec"), ns)
print("ENV", json.dumps({"python": sys.version, "torch": torch.__version__, "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda)}, ensure_ascii=False))

for experts in (4, 8):
    width, hidden = 4, 2
    per = 2 * width * hidden
    router = width * experts
    total = experts * per + router
    for k in (1, 2):
        active = k * per + router
        row = [experts, k, total, active, total / (2 * width), active / (2 * width)]
        expected = {(4, 1): [4, 1, 80, 32, 10, 4], (4, 2): [4, 2, 80, 48, 10, 6], (8, 1): [8, 1, 160, 48, 20, 6], (8, 2): [8, 2, 160, 64, 20, 8]}
        assert row == expected[(experts, k)]
        print("TOY experts,k,total,active,dense_total_hidden,dense_active_hidden", row)

config = ModelConfig(width=64, layers=2, heads=4, max_length=128)
for key, variant in raw["results"]["variants"].items():
    model = TinyLM(ModelConfig(**variant["model"]["config"]))
    budget = ns["_parameter_budget"](model)
    assert budget == variant["budget"]
    assert sum(p.numel() for p in model.parameters()) == variant["model"]["parameters"]
    print("MODEL", key, budget)
    if variant["comparison"]:
        target = variant["comparison"]["target_parameters"]
        width, delta = ns["_matched_width"](config, target)
        assert width == variant["model"]["config"]["width"]
        assert delta == variant["comparison"]["actual_minus_target"]
        print("MATCH", key, width, delta, "relative_abs_gap", abs(delta) / target)

records = load_jsonl(PROOF / "raw/tinystories-train-512.jsonl")
for record in records:
    assert hashlib.sha256(record["text"].encode()).hexdigest() == record["text_sha256"]
    record["family"] = record.get("text_sha256", records_sha256([{"text": record["text"]}]))
data = split_records(records, 42)
family_sets = [{row["family"] for row in data[split]} for split in ["train", "validation", "test"]]
assert not any(family_sets[i] & family_sets[j] for i in range(3) for j in range(i + 1, 3))
for split, rows in data.items():
    assert len(rows) == raw["results"]["dataset"][split]["records"]
    assert records_sha256(rows) == raw["results"]["dataset"][split]["sha256"]
    examples = text_examples(rows, max_length=128)
    targets = sum(len(y) for _, y in examples)
    eos = sum(int((y == ByteTokenizer.eos_id).sum()) for _, y in examples)
    assert targets == sum(len(row["text"].encode()) + 1 for row in rows)
    assert eos == len(rows)
    if split != "train":
        for variant in raw["results"]["variants"].values():
            assert targets == variant["heldout"][split]["effective_tokens"]
            assert len(examples) == variant["heldout"][split]["examples"]
    print("DATA", split, "records", len(rows), "examples", len(examples), "effective_targets", targets, "EOS_targets", eos, "records_sha256", records_sha256(rows))

examples = text_examples(data["train"], max_length=128)
sampler = random.Random(42)
sampled_targets = sum(sum(len(y) for _, y in sampler.choices(examples, k=16)) for _ in range(180))
assert sampled_targets == 337761
print("SAMPLING_ONLY_NO_UPDATES", "seed", 42, "steps", 180, "batch_size", 16, "effective_training_targets", sampled_targets)
expected_losses = {
    "dense_active_top1": (2.29999, 2.32093), "dense_active_top2": (2.28200, 2.30906), "dense_total": (2.22603, 2.24943),
    "top1_aux0": (2.30115, 2.32369), "top1_aux0.01": (2.29775, 2.31963), "top2_aux0": (2.28011, 2.30320), "top2_aux0.01": (2.27819, 2.30170)
}
for key, variant in raw["results"]["variants"].items():
    t = variant["training"]
    assert t["steps"] == t["optimizer_updates"] == t["requested_steps"] == 180
    assert t["skipped_updates"] == 0 and t["effective_tokens"] == sampled_targets
    assert len(variant["heldout"]["validation"]["samples"]) == len(variant["heldout"]["test"]["samples"]) == 8
    for i, split in enumerate(["validation", "test"]):
        held = variant["heldout"][split]
        ratio = held["nll_sum"] / held["effective_tokens"]
        assert math.isclose(ratio, held["nll"], rel_tol=0, abs_tol=1e-12)
        assert round(ratio, 5) == expected_losses[key][i]
        print("RAW_NLL", key, split, "sum", held["nll_sum"], "denominator", held["effective_tokens"], "ratio", ratio, "rounded5", format(ratio, ".5f"))

best = raw["results"]["variants"]
top2 = best["top2_aux0.01"]["heldout"]["validation"]["nll"]
assert top2 < best["dense_active_top2"]["heldout"]["validation"]["nll"]
assert top2 > best["dense_total"]["heldout"]["validation"]["nll"]
assert raw["results"]["teacher_variant"] == "top2_aux0.01"
sample = best["top2_aux0.01"]["heldout"]["test"]["samples"][0]
assert sample["prompt"] == "Once upon a time, there "
assert sample["generated"] == "tthe the the the the the the the"
assert ByteTokenizer().decode(sample["generated_ids"]) == sample["generated"]
print("SAMPLE", json.dumps(sample, ensure_ascii=False))

# A two-target CPU check confirms the class axis and ignored-label denominator.
logits = torch.tensor([[[1., 2., 3.], [4., 2., 1.], [0., 0., 0.]]], dtype=torch.float64)
labels = torch.tensor([[2, 0, IGNORE]])
summed, count = loss_sum(logits, labels)
manual = -logits.log_softmax(-1)[0, 0, 2] - logits.log_softmax(-1)[0, 1, 0]
assert int(count) == 2 and torch.allclose(summed, manual, atol=1e-14, rtol=0)
print("NLL_BOUNDED logits[B,T,C]=[1,3,3] labels[B,T]=[1,3]", "sum", float(summed), "effective_targets", int(count), "mean_nats", float(summed/count))
layer = MoEFFN(4, experts=4, top_k=2, hidden=2)
with torch.no_grad():
    output, auxiliary, chosen = layer(torch.randn(2, 3, 4))
counts = torch.bincount(chosen.flatten(), minlength=4)
fractions = counts.double() / counts.sum()
assert chosen.shape == (6, 2) and int(counts.sum()) == 12
assert output.shape == (2, 3, 4) and math.isclose(float(fractions.sum()), 1.0, abs_tol=1e-14)
print("ROUTING_BOUNDED", "tokens", 6, "k", 2, "dispatch_denominator", int(counts.sum()), "counts", counts.tolist(), "fractions_sum", float(fractions.sum()))
print("PASS: bounded arithmetic/construction/raw aggregation; no training, GPU, weights loading, or score regeneration")
