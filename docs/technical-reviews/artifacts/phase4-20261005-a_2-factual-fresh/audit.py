"""Bounded CPU inspection of existing A.2 raw records; no generation or training."""
import ast
import copy
import hashlib
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments.applications import _icl_records, _rag_counterfactual_contexts, _rag_splits
from scripts.course_experiments.common import records_sha256, split_records

HERE = Path(__file__).resolve().parent
record = json.loads((HERE / "raw/rag.json").read_bytes())
r = record["results"]
assert not torch.cuda.is_available() and torch.version.cuda is None
for p in ("applications.py", "common.py", "run.py"):
    name = "scripts/course_experiments/" + p
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == record["code_sha256"][name]

splits = _rag_splits(record["seed"])
icl = split_records(_icl_records(), record["seed"])
families = {name: {row["family"] for row in rows} for name, rows in splits.items()}
assert not families["train"] & families["validation"]
assert not families["train"] & families["test"]
assert not families["validation"] & families["test"]
for name, rows in splits.items():
    summary = r["split"][name]
    assert len(rows) == summary["records"]
    assert len(families[name]) == summary["families"]
    assert records_sha256(rows) == summary["sha256"]
training = splits["train"] + icl["train"]
assert len(training) == r["training"]["records"] == 894
assert records_sha256(training) == r["training"]["records_sha256"]
assert r["training"]["steps"] == r["training"]["planned_steps"] == 1000
assert r["training"]["step_scale"] == 1.0

# Recreate only the deterministic held-out facts; this does not run a model.
rng = random.Random(record["seed"] + 100)
facts = []
for key in sorted(families["test"]):
    source = f"D{rng.randrange(10)}"
    address = f"{rng.choice('ABCD')}{rng.randrange(10)}"
    facts.append({"family": key, "key": key, "address": address, "source": source})
assert len(facts) == 12
assert records_sha256(facts) == r["test_facts_sha256"]
by_key = {x["key"]: x for x in facts}
rows = r["samples"]
assert len(rows) == 84
assert Counter(x["mode"] for x in rows) == {mode: 12 for mode in r["metrics"]}
checked = {}
for row in rows:
    gen = row["generation"]
    assert gen["candidate_count"] == 1 and gen["temperature"] == 0.0 and gen["max_new_tokens"] == 16
    assert len(gen["samples"]) == 1
    sample = gen["samples"][0]
    ids = sample["generated_ids"]
    assert ids[-1] == 2 and sample["eos"] and all(i >= 8 for i in ids[:-1])
    text = bytes(i - 8 for i in ids[:-1]).decode("utf-8")
    assert text == sample["generated"]
    assert len(ids) == sample["generated_tokens"] == gen["generated_tokens"]
    prompt = [1]
    for message in gen["messages"]:
        role = {"system": 7, "user": 3, "assistant": 4}[message["role"]]
        prompt += [role] + [b + 8 for b in message["content"].encode("utf-8")] + [2]
    prompt += [4]
    assert prompt == gen["input_ids"] and len(prompt) == gen["input_tokens"]
    assert gen["seconds"] > 0
    assert row["fact"] == by_key[row["family"]]
    assert row["family"] in families["test"] and row["family"] not in families["train"]
    expected_fact = row["context_fact"]["address"] + "[" + row["context_fact"]["source"] + "]"
    correct = text == expected_fact
    assert correct == row["fact_answer_correct"]
    checked[(row["family"], row["mode"])] = {"generated": text, "correct": correct, "fact": row["context_fact"]}
    if row["mode"] in ("correct_context", "changed_address_context", "changed_source_context"):
        assert row["expected"] == expected_fact
        assert row["exact_match"] == correct
        wanted = [{"id": row["context_fact"]["source"], "text": row["family"] + " address=" + row["context_fact"]["address"]}]
        assert row["documents"] == wanted
        assert gen["messages"] == [{"role": "user", "content": "Docs:\n[" + wanted[0]["id"] + "] " + wanted[0]["text"] + "\nFind:" + row["family"] + "\nReply:address[source] or UNKNOWN"}]

counts = {}
for mode in ("correct_context", "changed_address_context", "changed_source_context"):
    correct = sum(checked[(key, mode)]["correct"] for key in by_key)
    assert correct == r["metrics"][mode]["fact_answer_correct"]["numerator"]
    assert r["metrics"][mode]["fact_answer_correct"]["denominator"] == 12
    counts[mode] = {"correct": correct, "questions": 12}
    if mode != "correct_context":
        pairs = sum(checked[(key, mode)]["correct"] and checked[(key, "correct_context")]["correct"] for key in by_key)
        assert pairs == r["paired_counterfactuals"][mode]["both_answers_correct"]["numerator"]
        assert r["paired_counterfactuals"][mode]["both_answers_correct"]["denominator"] == 12
        counts[mode]["both_correct"] = pairs

interventions = []
for fact in facts:
    untouched = copy.deepcopy(fact)
    variants = _rag_counterfactual_contexts(fact)
    assert fact == untouched
    for mode, case in variants.items():
        changed = [key for key in fact if case["fact"][key] != fact[key]]
        assert changed == (["address"] if mode == "changed_address_context" else ["source"])
        assert checked[(fact["key"], mode)]["fact"] == case["fact"]
    interventions.append({"family": fact["family"], "changed_only": ["address", "source"], "original_preserved": True})
examples = {}
for key, mode in (("K017", "changed_address_context"), ("K028", "changed_source_context")):
    examples[key] = {"baseline": checked[(key, "correct_context")]["generated"], "changed": checked[(key, mode)]["generated"]}
assert examples == {"K017": {"baseline": "C2[D1]", "changed": "C3[D1]"}, "K028": {"baseline": "B9[D9]", "changed": "B9[D0]"}}

# Verify ordering and absence of weight-update operations in the sampling method.
tree = ast.parse((ROOT / "scripts/course_experiments/applications.py").read_bytes())
run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_rag")
sample = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sample")
calls = [(n.func.id, n.lineno) for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
new_line = next(line for name, line in calls if name == "new_lm")
fit_line = next(line for name, line in calls if name == "_fit_lm")
fact_line = next(n.lineno for n in ast.walk(run) if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "facts" for x in n.targets))
sample_lines = [line for name, line in calls if name == "_sample"]
assert new_line < fit_line < fact_line < min(sample_lines)
assert sum(name == "_fit_lm" for name, line in calls) == 1
assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("backward", "step", "load_state_dict") for n in ast.walk(sample))

print(json.dumps({"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "cuda_build": str(torch.version.cuda)}, "original_run": {"revision": record["revision"], "run_id": record["modal"]["run_id"], "device": record["device"], "gpu": record["gpu"], "training_seconds": r["training"]["seconds"], "training_steps": r["training"]["steps"]}, "split_families": {k: len(v) for k,v in families.items()}, "training_records": len(training), "test_facts": len(facts), "raw_generation_rows": len(rows), "decoded_all_saved_tokens": True, "counts": counts, "examples": examples, "ordering": {"new_lm_line": new_line, "fit_line": fit_line, "new_facts_line": fact_line, "sample_lines": sample_lines}, "changes": interventions, "scope": "Existing raw data and short deterministic CPU checks only; no model sampling, weight loading, training, or new scores."}, ensure_ascii=False, indent=2))
