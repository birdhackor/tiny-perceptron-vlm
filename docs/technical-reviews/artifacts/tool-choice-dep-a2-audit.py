"""A.2 CPU audit of existing raw RAG records; never train or regenerate answers."""

import ast
import contextlib
import hashlib
import io
import json
import platform
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.applications import (
    _grounding,
    _icl_records,
    _prompt_ids,
    _rag_counterfactual_contexts,
    _rag_question,
    _rag_splits,
    _split_summary,
)
from scripts.course_experiments.common import Context, new_lm, records_sha256, split_records, text_examples
from scripts.course_experiments.run import experiment_spec
from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.retrieval import retrieve

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
report_path = ROOT / "docs/course-experiments/results/rag.json"
report = json.loads(report_path.read_bytes())
results = report["results"]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
relevant = [
    "scripts/course_experiments/applications.py",
    "scripts/course_experiments/common.py",
    "scripts/course_experiments/run.py",
    "tiny_perceptron/data.py",
    "tiny_perceptron/model.py",
    "tiny_perceptron/retrieval.py",
]
code_hashes = {p: sha(ROOT / p) for p in relevant}
assert all(report["code_sha256"][p] == value for p, value in code_hashes.items())

splits = _rag_splits(42)
icl = split_records(_icl_records(), 42)
assert _split_summary(splits) == results["split"]
assert _split_summary(icl) == results["icl_split"]
families = {part: {row["family"] for row in rows} for part, rows in splits.items()}
assert not (families["train"] & families["validation"] or families["train"] & families["test"] or families["validation"] & families["test"])
for part, rows in splits.items():
    for row in rows:
        for doc in row["documents"]:
            assert doc["text"].split()[0] in families[part]

training_rows = splits["train"] + icl["train"]
assert len(training_rows) == results["training"]["records"] == 894
assert records_sha256(training_rows) == results["training"]["records_sha256"]
examples = text_examples(training_rows, "sft", 160)
sampler = random.Random(42)
effective_tokens = 0
for _ in range(1000):
    effective_tokens += sum(int((labels != IGNORE).sum()) for _, labels in sampler.choices(examples, k=24))
assert effective_tokens == results["training"]["effective_tokens"] == 157361

rng = random.Random(142)
facts = []
for key in sorted(families["test"]):
    source = f"D{rng.randrange(10)}"
    address = f"{rng.choice('ABCD')}{rng.randrange(10)}"
    facts.append({"family": key, "key": key, "address": address, "source": source})
corpus = [{"id": f"doc-{row['key']}", "text": f"{row['key']} address={row['address']}"} for row in facts]
corpus += [{"id": f"noise-{index}", "text": f"Z{index:03d} address={rng.choice('ABCD')}{rng.randrange(10)}"} for index in range(60)]
assert records_sha256(facts) == results["test_facts_sha256"]
assert records_sha256(corpus) == results["corpus_sha256"]
artifact_hashes = {a["path"]: a["sha256"] for a in report["artifacts"]}
for name, value in [("dataset.json", splits), ("icl-dataset.json", icl), ("retrieval-corpus.json", corpus), ("generations.json", results["samples"])]:
    serialized = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    assert hashlib.sha256(serialized).hexdigest() == artifact_hashes[name]

tok = ByteTokenizer()
rows = results["samples"]
assert len(rows) == 84
fresh_scores = []
for row in rows:
    hits = retrieve(row["query"], corpus, k=1)
    assert [d["id"] for d in hits] == row["retrieved_ids"]
    assert (any(d["id"] == f"doc-{row['family']}" for d in hits)) == row["retrieval_hit"]
    assert row["fact"] == next(f for f in facts if f["family"] == row["family"])
    assert row["generation"]["messages"] == [{"role": "user", "content": _rag_question(row["family"], row["documents"])}]
    assert _prompt_ids(row["generation"]["messages"], tok) == row["generation"]["input_ids"]
    assert len(row["generation"]["input_ids"]) == row["generation"]["input_tokens"]
    assert row["generation"]["temperature"] == 0.0
    assert row["generation"]["candidate_count"] == 1
    assert row["generation"]["max_new_tokens"] == 16
    assert row["generation"]["input_tokens"] + 16 <= 160
    sample = row["generation"]["samples"][0]
    ids = sample["generated_ids"]
    assert ids[-1] == tok.eos_id and sample["eos"]
    assert all(i >= 8 for i in ids[:-1]) and not sample["invalid_special_tokens"]
    text = bytes(i - 8 for i in ids[:-1]).decode("utf-8")
    assert text == sample["generated"]
    if row["mode"].startswith("changed_"):
        original_hash = records_sha256(splits)
        fact_before = dict(row["fact"])
        intervention = _rag_counterfactual_contexts(row["fact"])[row["mode"]]
        assert intervention["fact"] == row["context_fact"] and intervention["documents"] == row["documents"]
        assert row["fact"] == fact_before and records_sha256(splits) == original_hash
    else:
        assert row["context_fact"] == row["fact"]
    answer = f"{row['context_fact']['address']}[{row['context_fact']['source']}]"
    expected = "UNKNOWN" if row["mode"] in ("without_context", "distractor_only") or (row["mode"] == "retrieved_context" and not hits) else answer
    fresh = _grounding(sample, row["documents"], expected)
    assert all(row[k] == value for k, value in fresh.items())
    correct = text == answer
    assert correct == row["fact_answer_correct"]
    fresh_scores.append({"family": row["family"], "mode": row["mode"], "generated": text, "raw_ids": ids, "expected": expected, "fact_answer_correct": correct, **fresh})

counts = {}
for mode in sorted({r["mode"] for r in fresh_scores}):
    selected = [r for r in fresh_scores if r["mode"] == mode]
    assert len(selected) == 12
    counts[mode] = {field: {"numerator": sum(r[field] for r in selected), "denominator": 12} for field in ("exact_match", "fact_answer_correct", "citation_valid", "supported_by_cited_source", "unknown")}
    for field, value in counts[mode].items():
        assert value["numerator"] == results["metrics"][mode][field]["numerator"]
        assert results["metrics"][mode][field]["denominator"] == 12
baseline = {r["family"]: r for r in fresh_scores if r["mode"] == "correct_context"}
paired = {}
for mode in ("changed_address_context", "changed_source_context"):
    changed = [r for r in fresh_scores if r["mode"] == mode]
    both = sum(r["fact_answer_correct"] and baseline[r["family"]]["fact_answer_correct"] for r in changed)
    changed_correct = sum(r["fact_answer_correct"] for r in changed)
    assert both == results["paired_counterfactuals"][mode]["both_answers_correct"]["numerator"]
    assert changed_correct == results["paired_counterfactuals"][mode]["changed_answer_correct"]["numerator"]
    paired[mode] = {"both_correct": both, "changed_correct": changed_correct, "denominator": 12}

ctx = Context("cpu", Path("/tmp/tool-choice-dep-a2-unused-output"), Path("/nonexistent"), ROOT / "assets/training", seed=42)
model = new_lm(ctx, width=64, layers=2, max_length=160, heads=2, backend="sdpa")
assert model.description() == results["model"]
tree = ast.parse((ROOT / "scripts/course_experiments/applications.py").read_text())
run_rag = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_rag")
calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in ast.walk(run_rag) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))]
assert "dependency" not in calls and "load_checkpoint" not in calls
spec = experiment_spec("rag")
assert spec["module"] == "applications" and spec["function"] == "run_rag"
assert spec["lessons"] == [f"A.{i}" for i in range(1, 8)]
assert results["training"]["steps"] == results["training"]["planned_steps"] == 1000
assert results["training"]["step_scale"] == report["step_scale"] == 1.0
assert report["evidence_status"] == "complete_run" and not report["unfinished_schedules"]
assert {"model.pt", "model-step-250.pt", "model-step-500.pt", "model-step-750.pt", "model-step-1000.pt"}.issubset(artifact_hashes)

exercise_code = (ROOT / "docs/technical-reviews/artifacts/tool-choice-dep-a2-original-fence-1.py").read_text()
assert exercise_code.count('"青街8號"') == 1
exercise_code = exercise_code.replace('"青街8號"', '"紅街9號"')
stdout = io.StringIO()
namespace = {}
with contextlib.redirect_stdout(stdout):
    exec(compile(exercise_code, "A.2 exercise address-only", "exec"), namespace)
assert stdout.getvalue() == "[來源公告1] 小小書店在2027年3月1日搬到紅街9號。\n問題：書店新地址？請依資料回答並標示來源。\n由資料確定的答案 紅街9號\n"

audit = {
    "reviewer_task": "/root/technical_dep_a2",
    "audited_at_utc": datetime.now(UTC).isoformat(),
    "command": "CUDA_VISIBLE_DEVICES='' .venv/bin/python docs/technical-reviews/artifacts/tool-choice-dep-a2-audit.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "cuda_available": str(torch.cuda.is_available())},
    "formal_report_path": report_path.relative_to(ROOT).as_posix(),
    "formal_report_sha256": sha(report_path),
    "formal_environment": {k: report[k] for k in ("revision", "seed", "device", "gpu", "torch_version", "python_version", "step_scale", "evidence_status")},
    "code_hashes_match_formal_run": code_hashes,
    "plan_sha256": sha(ROOT / "docs/course-experiments/plan.json"),
    "plan_rag_spec": spec,
    "split": _split_summary(splits),
    "icl_split": _split_summary(icl),
    "disjoint_family_and_distractor_membership": "verified",
    "mixed_training_records": len(training_rows),
    "batch_size": 24,
    "sampling_steps_without_training": 1000,
    "effective_answer_targets_recomputed": effective_tokens,
    "test_facts_sha256": records_sha256(facts),
    "corpus_sha256": records_sha256(corpus),
    "mode_counts_recomputed_from_raw_ids": counts,
    "paired_recomputed": paired,
    "examples": [r for r in fresh_scores if r["family"] in ("K017", "K028") and r["mode"] in ("correct_context", "changed_address_context", "changed_source_context")],
    "all_raw_prompt_ids_and_answers": "84/84 round-trip checks passed; EOS present and no illegal control token",
    "model_description_checked_without_training": model.description(),
    "model_uses_dependency_loader": False,
    "exercise": {"modified_field": "source.address only", "code": exercise_code, "stdout": stdout.getvalue()},
    "checkpoint_names": sorted(p for p in artifact_hashes if p.endswith(".pt")),
    "limitations": ["Existing CUDA report is audited; no model training, GPU run, or checkpoint generation is repeated.", "Private original checkpoints are absent locally; reported checkpoint hashes are not rechecked against weight files.", "Model quality results apply to one seed and 12 synthetic ASCII shop families; no natural-language RAG guarantee.", "plan.dependencies lists sft as scheduling metadata, but run_rag initializes a new TinyLM and never loads it."],
    "result": "All assertions passed",
}
print(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
