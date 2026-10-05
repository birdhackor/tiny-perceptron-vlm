"""7.14 independent bounded verification. No model loading, training, or downloads.

The AST loader runs only the named data/split/token helpers from immutable original
source snapshots. Reconstructed JSONL must equal the recorded run SHA before use.
The numeric checks independently use float64 logsumexp and analytic derivatives.
"""
import ast
import copy
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_named(relative, names, namespace):
    path = BASE / "inputs/original" / relative
    tree = ast.parse(path.read_bytes(), filename=str(path))
    found = []
    for node in tree.body:
        selected = isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        if isinstance(node, ast.Assign):
            selected = any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        if selected:
            # Decorated functions are not selected in this verification.
            assert not getattr(node, "decorator_list", [])
            found.append(node)
    exec(compile(ast.Module(body=found, type_ignores=[]), str(path), "exec"), namespace)
    assert all(n in namespace for n in names)


torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
environment = {
    "python": sys.version,
    "executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_git_version": torch.version.git_version,
    "device": "cpu",
    "platform": platform.platform(),
    "verification_kind": "original-result audit and bounded scalar/token checks; no retraining",
}
write("environment.json", environment)

# B=1 and C=4; natural-log units (nats); one long-integer target.
numeric = []
for wrong_score in [1.0, 5.0, 8.0]:
    x = torch.tensor([[0.0, 0.0, 3.0, wrong_score]], dtype=torch.float64, requires_grad=True)
    probabilities = x.softmax(dim=1)
    item = {"wrong_score": wrong_score, "probabilities": probabilities.detach().tolist()[0], "losses": {}}
    for label in [2, 3]:
        label_tensor = torch.tensor([label], dtype=torch.long)
        observed = F.cross_entropy(x, label_tensor)
        independent = math.log(sum(math.exp(s) for s in [0.0, 0.0, 3.0, wrong_score])) - [0.0, 0.0, 3.0, wrong_score][label]
        assert math.isclose(observed.item(), independent, rel_tol=0, abs_tol=1e-12)
        gradient, = torch.autograd.grad(observed, x, retain_graph=True)
        expected_gradient = probabilities.detach().clone()
        expected_gradient[0, label] -= 1
        assert torch.allclose(gradient, expected_gradient, atol=1e-12, rtol=0)
        item["losses"][str(label)] = observed.item()
        item["gradient_label_" + str(label)] = gradient.tolist()[0]
    numeric.append(item)
assert numeric[0]["losses"]["3"] > numeric[1]["losses"]["3"] > numeric[2]["losses"]["3"]
assert numeric[0]["losses"]["2"] < numeric[1]["losses"]["2"] < numeric[2]["losses"]["2"]
for observed, printed in [(numeric[0]["probabilities"][2], .810), (numeric[0]["probabilities"][3], .110), (numeric[0]["losses"]["2"], .211), (numeric[0]["losses"]["3"], 2.211), (numeric[1]["losses"]["3"], .139)]:
    assert abs(observed - printed) <= .0005
x = torch.tensor([[0., 0., 3., 1.]], dtype=torch.float64, requires_grad=True)
before = F.cross_entropy(x, torch.tensor([3]))
before.backward()
updated = x.detach() - .1 * x.grad
assert F.cross_entropy(updated, torch.tensor([3])) < before
assert updated[0, 3] > x.detach()[0, 3]
try:
    F.cross_entropy(x, torch.tensor([4]))
except IndexError as error:
    boundary = str(error)
else:
    raise AssertionError("Class ID 4 must be rejected for 4 candidates")
numeric_facts = {"rows": numeric, "original_shapes": {"input": [1, 4], "target": [1], "class_axis": 1}, "units": "nats per target; natural logarithm", "tolerance": "float64 analytic equality absolute 1e-12; text rounded values absolute 0.0005", "boundary_class4": boundary, "one_logit_step": {"before_wrong_loss": before.item(), "after_wrong_loss": F.cross_entropy(updated, torch.tensor([3])).item(), "updated_logits": updated.tolist()[0]}}
write("numeric.json", numeric_facts)

ns = {"torch": torch, "random": random, "json": json, "hashlib": hashlib}
load_named("tiny_perceptron/data.py", ["IGNORE", "SPECIALS", "ByteTokenizer", "render_chat", "shifted"], ns)
load_named("scripts/prepare_data.py", ["conversation", "generate_records"], ns)
load_named("scripts/course_experiments/common.py", ["split_records", "records_sha256", "text_examples"], ns)
report_path = BASE / "inputs/current/docs/course-experiments/results/sft_ablation.json"
recorded = json.loads(report_path.read_bytes())
assert recorded["revision"] == "52964f650787cef393d18647a350471637208f67"
assert recorded["seed"] == 42 and recorded["step_scale"] == 1.0 and recorded["status"] == "completed"
for relative in ["scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/training.py"]:
    assert sha((BASE / "inputs/original" / relative).read_bytes()) == recorded["code_sha256"][relative]
parts = ns["split_records"](ns["generate_records"]("attributes-sft"), seed=42)
splits = {}
for name, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    manifest = recorded["results"]["data"]["attributes"][name]
    assert sha(raw) == manifest["sha256"]
    families = sorted({r["family"] for r in rows})
    assert len(rows) == manifest["records"] and len(families) == manifest["families"]
    (OUT / ("original-attributes-" + name + ".jsonl")).write_bytes(raw)
    splits[name] = {"records": len(rows), "families": families, "sha256": sha(raw)}
assert not (set(splits["train"]["families"]) & set(splits["validation"]["families"]))
assert not (set(splits["train"]["families"]) & set(splits["test"]["families"]))
assert not (set(splits["validation"]["families"]) & set(splits["test"]["families"]))

# Independently interpret the question rule and audit the reconstructed labels.
for rows in parts.values():
    for row in rows:
        bits = row["messages"][0]["content"].split(";")
        attrs = dict(bit.split("=", 1) for bit in bits[:-1])
        task = bits[-1]
        answer = {"describe": attrs["shape"], "shape?": attrs["shape"], "color?": attrs["color"], "pitch?": attrs["pitch"], "joint?": attrs["shape"] + "," + attrs["pitch"]}[task]
        assert row["messages"][-1]["content"] == answer
clean = parts["train"]
noisy = copy.deepcopy(clean)
corruptions = []
for index, row in enumerate(noisy):
    answer = row["messages"][-1]["content"]
    if answer in ("circle", "square") and len(corruptions) < max(1, len(noisy) // 10):
        row["messages"][-1]["content"] = "square" if answer == "circle" else "circle"
        corruptions.append({"row": index, "family": row["family"], "correct": answer, "wrong": row["messages"][-1]["content"]})
assert corruptions == recorded["results"]["corruptions"]
assert [i for i, (a, b) in enumerate(zip(clean, noisy)) if a != b] == [0, 1, 5, 6]
assert all(a["messages"][0] == b["messages"][0] for a, b in zip(clean, noisy))
sampling = {}
for name, rows in [("clean", clean), ("noisy", noisy)]:
    training = recorded["results"]["runs"][name]["training"]
    assert ns["records_sha256"](rows) == training["records_sha256"]
    assert training["steps"] == 300 and training["records"] == 45
    examples = ns["text_examples"](rows, mode="sft", max_length=128)
    lengths = [int((y != ns["IGNORE"]).sum()) for x, y in examples]
    rng = random.Random(42)
    selected_indices = [rng.choices(range(len(rows)), k=16) for _ in range(300)]
    effective = sum(lengths[i] for batch in selected_indices for i in batch)
    corrupted_draws = sum(i in [0, 1, 5, 6] for batch in selected_indices for i in batch)
    assert effective == training["effective_tokens"] == 33733
    sampling[name] = {"steps": 300, "batch_size": 16, "draws": 4800, "effective_targets": effective, "unique_dataset_targets": sum(lengths), "records_sha256": training["records_sha256"], "sample_index_sequence_sha256": sha(json.dumps(selected_indices).encode()), "draws_of_changed_rows": corrupted_draws, "targets_of_changed_rows": 7 * corrupted_draws}
assert sampling["clean"] == sampling["noisy"] | {"records_sha256": sampling["clean"]["records_sha256"]}

tok = ns["ByteTokenizer"]()
evaluations = {}
for name in ["clean", "noisy"]:
    evaluations[name] = {}
    for split in ["validation", "test"]:
        e = recorded["results"]["runs"][name]["attributes"][split]
        assert len(e["samples"]) == len(parts[split]) == e["records"] == e["examples"]
        matches = eos_count = 0
        audited_samples = []
        for i, (sample, row) in enumerate(zip(e["samples"], parts[split])):
            assert sample["messages"] == row["messages"][:-1] and sample["expected"] == row["messages"][-1]["content"]
            ids = sample["generated_ids"]
            ended = tok.eos_id in ids
            raw = ids[:ids.index(tok.eos_id)] if ended else ids
            assert tok.decode(raw) == sample["generated"]
            exact = raw == tok.encode(sample["expected"])
            assert exact == sample["exact"] and ended == sample["eos"]
            assert all(8 <= token < tok.vocab_size for token in raw)
            assert ids[-1] == tok.eos_id and ids.count(tok.eos_id) == 1 and len(ids) <= 32
            matches += exact
            eos_count += ended
            audited_samples.append({"row": i, "prompt": sample["messages"][0]["content"], "expected": sample["expected"], "raw_ids": ids, "decoded": tok.decode(raw), "exact_recomputed": exact, "eos_recomputed": ended})
        count = sum(len(tok.encode(r["messages"][-1]["content"])) + 1 for r in parts[split])
        assert count == e["effective_tokens"]
        mean = e["nll_sum"] / count
        assert math.isclose(mean, e["nll"], rel_tol=0, abs_tol=1e-12)
        assert matches == e["matches"] and matches / len(e["samples"]) == e["exact_match"]
        assert eos_count / len(e["samples"]) == e["eos_rate"] == 1.0
        evaluations[name][split] = {"matches": matches, "records": len(e["samples"]), "eos_count": eos_count, "effective_targets_including_eos": count, "recorded_nll_sum": e["nll_sum"], "recomputed_mean_nll": mean, "rounded5": f"{mean:.5f}", "samples": audited_samples}
assert evaluations["clean"]["validation"]["matches"] == 2
assert evaluations["noisy"]["validation"]["matches"] == 1
assert evaluations["clean"]["test"]["matches"] == 7
assert evaluations["noisy"]["test"]["matches"] == 6
assert evaluations["clean"]["test"]["rounded5"] == "0.37101"
assert evaluations["noisy"]["test"]["rounded5"] == "0.34698"

# Before metrics match the direct SFT model's stored after metrics; no weights opened.
sft = json.loads((BASE / "inputs/current/docs/course-experiments/results/sft.json").read_bytes())
assert recorded["results"]["before"]["A_attributes"] == sft["results"]["after"]
assert sft["results"]["checkpoint"] == "model.pt"
assert tok.encode("2") == [58] and tok.encode("3") == [59]
hidden_role = tok.encode("circle") + [tok.user_id]
assert tok.decode(hidden_role) == "circle" and hidden_role != tok.encode("circle")

# Bounded perturbation in target accounting, not a new dataset or trained model.
short = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "x"}]
long = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "xxxxxx"}]
short_count = int((ns["render_chat"](short)[1] != ns["IGNORE"]).sum())
long_count = int((ns["render_chat"](long)[1] != ns["IGNORE"]).sum())
assert (short_count, long_count) == (2, 7)
audit = {"run_revision": recorded["revision"], "original_run_environment": {k: recorded[k] for k in ["seed", "torch_version", "python_version", "device", "gpu", "step_scale", "evidence_status"]}, "original_result_sha256": sha(report_path.read_bytes()), "splits": splits, "corruptions": corruptions, "fraction": {"changed_records": 4, "training_records": 45, "percent": 100 * 4 / 45, "changed_families": sorted({c["family"] for c in corruptions})}, "sampling": sampling, "evaluations": evaluations, "base_evaluation_equal_to_direct_sft_after": True, "token_id_boundary": {"classification_ids": [0, 1, 2, 3], "byte_digit2": tok.encode("2"), "byte_digit3": tok.encode("3"), "hidden_role_decode_does_not_pass_exact": True}, "bounded_target_length_variation": {"short_ascii_target_count_including_eos": short_count, "long_ascii_target_count_including_eos": long_count}, "limitations": "Counts, input hashes, raw-ID matches, EOS, budgets and stored NLL aggregation verified. Stored NLL sums are original CUDA measurements, not freshly computed logits. No weights opened, optimizer run, training rerun, or model/download performed. One seed and 1 validation/2 test families cannot estimate broad causal effect or a general contamination law."}
write("original-result-audit.json", audit)
print(json.dumps({"numeric": numeric_facts, "result_summary": {"split_record_counts": {s: x["records"] for s,x in splits.items()}, "corruption_fraction_percent": 100*4/45, "corruptions": corruptions, "sampling": sampling, "validation_matches": [2, 1], "test_matches": [7, 6], "test_nll": [evaluations[n]["test"]["recomputed_mean_nll"] for n in ["clean", "noisy"]], "base_evaluation_equal": True}, "all_checks_passed": True}, ensure_ascii=False, indent=2))
