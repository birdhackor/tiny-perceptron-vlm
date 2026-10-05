"""A.5 independent, bounded CPU checks; no model creation, loading or generation."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import runpy
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.retrieval import retrieve


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    (HERE / name).write_bytes(raw)
    return raw


def original_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    # Exact existing bodies, without running module-level code or training helpers.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return {n.name: [n.lineno, n.end_lineno] for n in selected}


assert torch.version.cuda is None
assert not torch.cuda.is_available()
environment = {
    "python": platform.python_version(), "python_executable": sys.executable,
    "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
    "device": "cpu", "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "original fence, paper variation, recompute saved measurements and deterministic synthetic inputs; no weights or model calls",
}
save("environment.json", environment)

# Run the unchanged original fence as its own process and preserve actual stdout.
argv = [sys.executable, str(HERE / "fence-1.py")]
cp = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
(HERE / "original.stdout.txt").write_text(cp.stdout)
(HERE / "original.stderr.txt").write_text(cp.stderr)
assert cp.returncode == 0
expected_stdout = (
    "有正確來源 0.6666666666666666\n答案正確 0.3333333333333333\n"
    "乙店地址 已找到但答錯\n丙店地址 尚未找到所需來源\n"
)
assert cp.stdout == expected_stdout
save("original.execution.json", {
    "command_argv": argv, "cwd": str(ROOT), "exit_code": cp.returncode,
    "source_sha256": sha((HERE / "fence-1.py").read_bytes()),
    "stdout_sha256": sha(cp.stdout.encode()), "stderr_sha256": sha(cp.stderr.encode()),
    "environment_file": "environment.json",
})
with contextlib.redirect_stdout(io.StringIO()):
    fence_ns = runpy.run_path(str(HERE / "fence-1.py"))
examples = [dict(row) for row in fence_ns["examples"]]
examples[1]["correct"] = True
variant = {"source_hits": sum(e["hit"] for e in examples),
           "correct_answers": sum(e["correct"] for e in examples), "questions": len(examples)}
assert variant == {"source_hits": 2, "correct_answers": 2, "questions": 3}

published = json.loads((HERE / "frozen/docs/course-experiments/results/rag.json").read_bytes())
artifact_refs = {a["path"]: a for a in published["artifacts"]}
code_paths = ["scripts/course_experiments/applications.py", "scripts/course_experiments/common.py",
              "tiny_perceptron/retrieval.py", "tiny_perceptron/data.py"]
for path in code_paths:
    raw = (ROOT / path).read_bytes()
    assert sha(raw) == published["code_sha256"][path], path
    frozen = HERE / "frozen" / path
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if frozen.exists():
        assert frozen.read_bytes() == raw
    else:
        frozen.write_bytes(raw)

ns = {"random": random, "json": json, "hashlib": hashlib, "re": re}
common_ranges = original_functions(ROOT / "scripts/course_experiments/common.py",
                                  ["split_records", "records_sha256"], ns)
app_ranges = original_functions(ROOT / "scripts/course_experiments/applications.py",
    ["_rag_question", "_rag_records", "_rag_splits", "_grounding", "_conversation", "_prompt_ids"], ns)

# Reproduce synthetic input bytes from the hash-matching original methods only.
splits = ns["_rag_splits"](published["seed"])
for key, records in splits.items():
    expected = published["results"]["split"][key]
    assert len(records) == expected["records"]
    assert len({r["family"] for r in records}) == expected["families"]
    assert ns["records_sha256"](records) == expected["sha256"]
sets = {key: {r["key"] for r in records} for key, records in splits.items()}
assert sets["train"].isdisjoint(sets["validation"] | sets["test"])
assert sets["validation"].isdisjoint(sets["test"])
dataset_raw = save("reconstructed-dataset.json", splits)
assert len(dataset_raw) == artifact_refs["dataset.json"]["bytes"]
assert sha(dataset_raw) == artifact_refs["dataset.json"]["sha256"]

rng = random.Random(published["seed"] + 100)
facts = []
for key in sorted(sets["test"]):
    source = f"D{rng.randrange(10)}"
    address = f"{rng.choice('ABCD')}{rng.randrange(10)}"
    facts.append({"family": key, "key": key, "address": address, "source": source})
assert len(facts) == 12
assert ns["records_sha256"](facts) == published["results"]["test_facts_sha256"]
corpus = [{"id": f"doc-{f['key']}", "text": f"{f['key']} address={f['address']}"} for f in facts]
corpus += [{"id": f"noise-{i}", "text": f"Z{i:03d} address={rng.choice('ABCD')}{rng.randrange(10)}"}
           for i in range(60)]
assert ns["records_sha256"](corpus) == published["results"]["corpus_sha256"]
corpus_raw = save("reconstructed-retrieval-corpus.json", corpus)
assert sha(corpus_raw) == artifact_refs["retrieval-corpus.json"]["sha256"]
assert len(corpus_raw) == artifact_refs["retrieval-corpus.json"]["bytes"]

# Preserve byte-exact original raw generations embedded in the published result.
# All 84 rows are serialized for the original hash; only the two relevant modes are scored here.
all_rows = published["results"]["samples"]
generation_raw = save("original-generations.json", all_rows)
assert sha(generation_raw) == artifact_refs["generations.json"]["sha256"]
assert len(generation_raw) == artifact_refs["generations.json"]["bytes"]

tok = ByteTokenizer()
ns["ByteTokenizer"] = ByteTokenizer
facts_by_key = {f["key"]: f for f in facts}
selected = [(i, r) for i, r in enumerate(all_rows) if r["mode"] in ("correct_context", "retrieved_context")]
assert len(selected) == 24
aggregates = {}
pointers = []
comparison = []
for mode in ["correct_context", "retrieved_context"]:
    rows = [(i, r) for i, r in selected if r["mode"] == mode]
    keys = [r["family"] for _, r in rows]
    assert len(keys) == len(set(keys)) == 12
    assert set(keys) == sets["test"]
    totals = dict(context_hit=0, exact_match=0, fact_answer_correct=0, citation_valid=0,
                  supported_by_cited_source=0, unknown=0, correct_unknown_on_miss=0)
    for i, row in rows:
        fact = facts_by_key[row["family"]]
        assert row["fact"] == fact == row["context_fact"]
        assert row["paired_baseline_mode"] is None
        query = row["query"]
        original_index = sorted(sets["test"]).index(fact["key"])
        assert query == (fact["key"] if original_index % 4 else fact["key"].replace("K", "shop"))
        hits = retrieve(query, corpus, k=1)
        assert row["retrieved_ids"] == [h["id"] for h in hits]
        assert row["retrieval_hit"] == any(h["id"] == f"doc-{fact['key']}" for h in hits)
        assert row["citation_id_map"] == {h["id"]: fact["source"] for h in hits}
        docs = ([{"id": fact["source"], "text": f"{fact['key']} address={fact['address']}"}]
                if mode == "correct_context" else [{"id": fact["source"], "text": h["text"]} for h in hits])
        assert row["documents"] == docs
        # Context presence is recomputed from the supplied documents; retrieval_hit on
        # correct_context rows still refers to the separate lexical retrieval attempt.
        context_hit = any(d["id"] == fact["source"] and
                          d["text"] == f"{fact['key']} address={fact['address']}" for d in docs)
        gold = f"{fact['address']}[{fact['source']}]"
        expected = gold if context_hit else "UNKNOWN"
        assert row["expected"] == expected
        gen = row["generation"]
        assert len(gen["samples"]) == gen["candidate_count"] == 1
        assert gen["temperature"] == 0.0 and gen["max_new_tokens"] == 16
        prompt = ns["_rag_question"](fact["key"], docs)
        assert gen["messages"] == [{"role": "user", "content": prompt}]
        assert gen["input_ids"] == ns["_prompt_ids"](gen["messages"], tok)
        assert len(gen["input_ids"]) == gen["input_tokens"]
        sample = gen["samples"][0]
        ids = sample["generated_ids"]
        assert sample["eos"] == bool(ids and ids[-1] == tok.eos_id)
        content_ids = ids[:-1] if sample["eos"] else ids
        assert sample["generated"] == tok.decode(content_ids)
        assert sample["invalid_special_tokens"] == [v for v in content_ids if v < 8]
        assert len(ids) == sample["generated_tokens"] == gen["generated_tokens"]
        clean = not sample["invalid_special_tokens"]
        text = sample["generated"]
        parse = re.fullmatch(r"([A-E][0-9])\[(D[0-9])\]", text)
        address, citation = (parse[1], parse[2]) if parse else (None, None)
        independent = {
            "exact_match": bool(clean and text == expected),
            "fact_answer_correct": bool(clean and text == gold),
            "parsed_address": address, "parsed_citation": citation,
            "citation_valid": bool(clean and citation in {d["id"] for d in docs}),
            "supported_by_cited_source": bool(clean and parse and any(
                d["id"] == citation and f"address={address}" in d["text"] for d in docs)),
            "unknown": bool(clean and text == "UNKNOWN"),
        }
        for field, value in independent.items():
            assert row[field] == value, (i, field)
        original_grounding = ns["_grounding"](sample, docs, expected)
        for field, value in original_grounding.items():
            assert row[field] == value
        totals["context_hit"] += context_hit
        for field in ["exact_match", "fact_answer_correct", "citation_valid", "supported_by_cited_source", "unknown"]:
            totals[field] += independent[field]
        totals["correct_unknown_on_miss"] += not context_hit and independent["unknown"]
        comparison.append({"pointer": f"/results/samples/{i}", "mode": mode,
                           "family": row["family"], "supplied_documents": docs,
                           "context_hit": context_hit, "lexical_retrieval_hit": row["retrieval_hit"],
                           "gold_address_and_source": gold, "protocol_expected": expected,
                           "saved_generated_text": text, **independent})
        for field in ["family", "mode", "fact", "context_fact", "paired_baseline_mode", "query", "documents",
                      "retrieved_ids", "retrieval_hit", "citation_id_map", "expected", "exact_match", "fact_answer_correct",
                      "parsed_address", "parsed_citation", "citation_valid", "supported_by_cited_source", "unknown"]:
            pointers.append(f"/results/samples/{i}/{field}")
        for field in ["messages", "input_ids", "input_tokens", "candidate_count", "temperature", "max_new_tokens", "generated_tokens"]:
            pointers.append(f"/results/samples/{i}/generation/{field}")
        for field in ["generated", "generated_ids", "eos", "invalid_special_tokens", "generated_tokens"]:
            pointers.append(f"/results/samples/{i}/generation/samples/0/{field}")
    for field in ["exact_match", "fact_answer_correct", "citation_valid", "supported_by_cited_source", "unknown"]:
        recorded = published["results"]["metrics"][mode][field]
        assert recorded["numerator"] == totals[field] and recorded["denominator"] == len(rows)
        assert recorded["rate"] == totals[field] / len(rows)
    aggregates[mode] = {**totals, "denominator": len(rows)}
assert aggregates["correct_context"]["context_hit"] == 12
assert aggregates["correct_context"]["fact_answer_correct"] == aggregates["correct_context"]["exact_match"] == 5
assert aggregates["retrieved_context"]["context_hit"] == 9
assert aggregates["retrieved_context"]["fact_answer_correct"] == 3
assert aggregates["retrieved_context"]["correct_unknown_on_miss"] == 3
assert aggregates["retrieved_context"]["exact_match"] == 6
assert published["results"]["retrieval_recall_at_1"] == {"numerator": 9, "denominator": 12, "rate": .75}

# Score-only fixtures test distinct protocol/truth/source criteria, not model behavior.
fixtures = []
for name, text, docs, expected in [
    ("absent_truth_UNKNOWN", "UNKNOWN", [], "UNKNOWN"),
    ("valid_but_wrong_address", "C3[D1]", [{"id": "D1", "text": "K017 address=C3"}], "C2[D1]"),
    ("truth_with_unlisted_citation", "C2[D1]", [{"id": "D2", "text": "K017 address=C2"}], "C2[D1]"),
    ("invalid_control_even_if_text_matches", "C2[D1]", [{"id": "D1", "text": "K017 address=C2"}], "C2[D1]"),
]:
    scored = ns["_grounding"]({"generated": text, "invalid_special_tokens": [0] if name.startswith("invalid") else []}, docs, expected)
    fixtures.append({"name": name, **scored})
assert fixtures[0]["exact_match"] and fixtures[0]["unknown"] and not fixtures[0]["citation_valid"]
assert not fixtures[1]["exact_match"] and fixtures[1]["citation_valid"] and fixtures[1]["supported_by_cited_source"]
assert fixtures[2]["exact_match"] and not fixtures[2]["citation_valid"] and not fixtures[2]["supported_by_cited_source"]
assert not fixtures[3]["exact_match"] and not fixtures[3]["citation_valid"]

result = {"original_fence": "exit 0, exact stdout match", "paper_variant": variant,
          "dataset_families": {k: len(v) for k, v in sets.items()}, "test_families": sorted(sets["test"]),
          "test_facts_hash_matches": True, "corpus_hash_matches": True,
          "original_generations_bytes_and_hash_match": True, "original_code_hashes_match": code_paths,
          "pairing": "same 12 unique held-out keys and exact same fact values; one saved response per mode/key",
          "aggregates": aggregates, "score_only_fixtures": fixtures,
          "api_coverage": {"original_app_functions": app_ranges, "original_common_functions": common_ranges,
                           "retrieve": "tiny_perceptron/retrieval.py:8-20", "ByteTokenizer": "tiny_perceptron/data.py:12-27",
                           "inference_contract_inspected": "applications.py:33-103 and run_rag:297-442; not executed"},
          "scope": "24 saved responses rescored; deterministic synthetic dataset/corpus reproduced with original hashes; no new model answers, training, or weights"}
save("paired-measurements.json", comparison)
save("verification-results.json", result)
save("inspected-pointers.json", {
    "source": "frozen/docs/course-experiments/results/rag.json", "sample_leaf_pointers": pointers,
    "additional_raw_pointers": ["/seed", "/device", "/python_version", "/torch_version", "/revision", "/code_sha256",
        "/artifacts", "/results/split", "/results/test_facts_sha256", "/results/corpus_sha256", "/results/retrieval_recall_at_1",
        "/results/metrics/correct_context/{exact_match,fact_answer_correct,citation_valid,supported_by_cited_source,unknown}",
        "/results/metrics/retrieved_context/{exact_match,fact_answer_correct,citation_valid,supported_by_cited_source,unknown}",
        "/hf/{repo,revision,prefix,result_revision,private_full_experiment_output}"],
    "opaque_preservation": "Whole source bytes preserved; unneeded narrative fields were not printed or interpreted.",
    "original_generation_preservation": "All 84 raw generation rows serialized only to recover the exact original bytes/hash; substantive inspection is limited to 24 correct_context/retrieved_context rows."
})
print(json.dumps({"original_fence": result["original_fence"], "paper_variant": variant,
                  "aggregates": aggregates, "pairing": result["pairing"], "provenance_hashes": "all matched",
                  "models_loaded_or_called": 0}, ensure_ascii=False, indent=2))
