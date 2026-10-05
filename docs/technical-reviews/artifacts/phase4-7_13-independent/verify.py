"""Bounded CPU verification: original-fence variations and historical result replay.

No training, generation from a downloaded model, new data acquisition, or weight files.
Historical pure helpers are compiled unchanged from their saved AST nodes.
"""
import ast
import hashlib
import importlib.util
import json
import platform
import random
import sys
from pathlib import Path

import torch

A = Path(__file__).resolve().parent
H = A / "inputs/historical"
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


fence = (A / "fence-1.py").read_text()
variants = {}
for name, code, expected_assertion in [
    ("original_3", fence, False),
    ("answer_2_old_assertion", fence.replace('"answer": 3', '"answer": 2'), True),
    (
        "answer_2_updated_assertion",
        fence.replace('"answer": 3', '"answer": 2').replace(
            'and not scores["1+1內容正確"]', 'and scores["1+1內容正確"]'
        ),
        False,
    ),
    ("result_2", fence.replace('"answer": 3', '"result": 2'), True),
]:
    ns = {}
    failed = False
    try:
        exec(compile(code, name, "exec"), ns)
    except AssertionError:
        failed = True
    assert failed == expected_assertion
    variants[name] = {"scores": ns["scores"], "assertion_failed": failed}
    (A / "variations").mkdir(exist_ok=True)
    (A / "variations" / f"{name}.py").write_text(code)
assert variants["original_3"]["scores"] == {"格式符合": True, "1+1內容正確": False}
assert variants["answer_2_updated_assertion"]["scores"] == {"格式符合": True, "1+1內容正確": True}
assert variants["result_2"]["scores"] == {"格式符合": False, "1+1內容正確": False}

contract_variants = []
for reply, wanted in [
    ('{"answer": true}', (False, False)),
    ('{"answer": false}', (False, False)),
    ('{"answer": 2.0}', (False, True)),
    ('{"answer": 2, "extra": 1}', (False, True)),
]:
    parsed = json.loads(reply)
    scores = (
        isinstance(parsed, dict) and set(parsed) == {"answer"} and type(parsed["answer"]) is int,
        parsed.get("answer") == 2,
    )
    assert scores == wanted
    contract_variants.append({"reply": reply, "format_content": scores, "python_type": type(parsed["answer"]).__name__})
assert isinstance(True, int) and isinstance(False, int)
assert type(True) is not int and type(False) is not int

spec = importlib.util.spec_from_file_location("historical_data", H / "tiny_perceptron/data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)
ns = {"json": json, "hashlib": hashlib, "random": random, "torch": torch,
      "ByteTokenizer": data.ByteTokenizer, "render_chat": data.render_chat, "shifted": data.shifted}
load_functions(H / "scripts/prepare_data.py", ["conversation", "generate_records"], ns)
load_functions(H / "scripts/course_experiments/common.py", ["split_records", "text_examples", "records_sha256"], ns)
report_raw = (A / "inputs/results/sft.json").read_bytes()
report = json.loads(report_raw)
results = report["results"]
parts = ns["split_records"](ns["generate_records"]("attributes-sft"), seed=report["seed"])
assert sum(map(len, parts.values())) == 60
families = {k: {r["family"] for r in rows} for k, rows in parts.items()}
assert not any(families[l] & families[r] for l, r in [("train", "validation"), ("train", "test"), ("validation", "test")])
(A / "reconstructed-inputs").mkdir(exist_ok=True)
split_summary = {}
for split, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    (A / "reconstructed-inputs" / f"{split}.jsonl").write_bytes(raw)
    manifest = results["data"][split]
    assert sha(raw) == manifest["sha256"]
    assert len(rows) == manifest["records"] and len(families[split]) == manifest["families"]
    examples = ns["text_examples"](rows, "sft", 128)
    split_summary[split] = {"records": len(rows), "families": sorted(families[split]),
                            "effective_answer_targets_including_eos": sum(int((y != -100).sum()) for x, y in examples),
                            "sha256": sha(raw)}
assert ns["records_sha256"](parts["train"]) == results["training"]["records_sha256"]

tok = data.ByteTokenizer()
replay = {}
for branch, evaluations in [
    ("before_direct_sft", results["before"]),
    ("after_direct_sft", results["after"]),
    ("before_pretrain_sft", results["pretrain_then_sft"]["before_sft"]),
    ("after_pretrain_sft", results["pretrain_then_sft"]["after_sft"]),
]:
    replay[branch] = {}
    for split, evaluation in evaluations.items():
        samples = evaluation["samples"]
        assert len(samples) == len(parts[split]) == evaluation["records"] == evaluation["examples"]
        exact_count = eos_count = 0
        rows = []
        for index, (sample, record) in enumerate(zip(samples, parts[split], strict=True)):
            assert sample["messages"] == record["messages"][:-1]
            assert sample["expected"] == record["messages"][-1]["content"]
            ids = sample["generated_ids"]
            assert len(ids) <= 32
            raw_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            exact = raw_ids == tok.encode(sample["expected"])
            eos = tok.eos_id in ids
            assert exact is sample["exact"] and eos is sample["eos"]
            assert tok.decode(raw_ids) == sample["generated"]
            exact_count += exact
            eos_count += eos
            rows.append({"row": index, "question": sample["messages"][0]["content"],
                         "expected": sample["expected"], "generated": sample["generated"],
                         "recomputed_exact": exact, "recomputed_eos": eos})
        assert exact_count == evaluation["matches"]
        assert exact_count / len(samples) == evaluation["exact_match"]
        assert eos_count / len(samples) == evaluation["eos_rate"]
        effective = split_summary[split]["effective_answer_targets_including_eos"]
        assert effective == evaluation["effective_tokens"]
        assert abs(evaluation["nll_sum"] / effective - evaluation["nll"]) <= 1e-12
        replay[branch][split] = {"matches": exact_count, "records": len(samples), "eos": eos_count,
                               "effective_targets": effective, "recomputed_nll_ratio": evaluation["nll_sum"] / effective,
                               "samples": rows}


def counted_targets(records, mode, steps):
    examples = ns["text_examples"](records, mode, 128)
    rng = random.Random(report["seed"])
    # Only reconstruct which examples were sampled, not any gradient or parameter update.
    return sum(int((y != -100).sum()) for _ in range(steps) for x, y in rng.choices(examples, k=16))


direct_tokens = counted_targets(parts["train"], "sft", 900)
text_rows = [{"text": r["messages"][0]["content"] + r["messages"][1]["content"], "family": r["family"]} for r in parts["train"]]
pretrain_tokens = counted_targets(text_rows, "text", 250)
assert direct_tokens == results["training"]["effective_tokens"] == 101091
assert pretrain_tokens == results["pretrain_then_sft"]["pretraining"]["effective_tokens"] == 191742
assert direct_tokens == results["pretrain_then_sft"]["sft"]["effective_tokens"]
assert results["training"]["steps"] == results["pretrain_then_sft"]["sft"]["steps"] == 900
dataset_bytes = (json.dumps(parts, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
(A / "reconstructed-inputs/dataset.json").write_bytes(dataset_bytes)
artifact_hashes = {a["path"]: a["sha256"] for a in report["artifacts"]}
assert sha(dataset_bytes) == artifact_hashes["dataset.json"]
downstream = {}
for name in ["quantization", "distillation"]:
    downstream_report = json.loads((A / "inputs/results" / f"{name}.json").read_bytes())
    downstream_result = downstream_report["results"]
    if name == "distillation":
        downstream_result = downstream_result["tasks"]["attributes"]
    provenance = downstream_result["teacher_provenance"]
    dataset = downstream_result["data"]
    assert provenance["sha256"] == artifact_hashes["model.pt"]
    assert provenance["checkpoint"] == "/course/course-v1/sft/model.pt"
    assert provenance["metadata"]["seed"] == report["seed"]
    assert dataset["sha256"] == sha(dataset_bytes)
    assert dataset["counts"] == {k: len(v) for k, v in parts.items()}
    downstream[name] = {"revision": downstream_report["revision"], "checkpoint_sha256": provenance["sha256"],
                        "dataset_sha256": dataset["sha256"], "counts": dataset["counts"]}

output = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu",
                    "cuda_build": str(torch.version.cuda), "threads": str(torch.get_num_threads()), "executable": sys.executable},
    "scope": "Replay original results and exact historical helpers. No training, weights, or regenerated model answers.",
    "tolerance": "All count/hash/ID/boolean comparisons exact; NLL_sum/effective_tokens division absolute error <= 1e-12.",
    "original_report_sha256": sha(report_raw), "original_revision": report["revision"],
    "fence_variants": variants, "contract_variants": contract_variants,
    "splits": split_summary, "evaluation_replay": replay,
    "training_budget_replay": {"seed": report["seed"], "batch_size": 16,
                              "direct_sft": {"steps": 900, "sampled_records": 14400, "effective_targets": direct_tokens},
                              "pretrain_then_sft": {"steps": 1150, "sampled_records": 18400,
                                                    "effective_targets": pretrain_tokens + direct_tokens}},
    "downstream_provenance_replay": downstream,
    "weights_not_loaded": True,
}
(A / "verification.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in output.items() if k != "evaluation_replay"}, ensure_ascii=False, indent=2))
print("direct_sft_test=5/10; pretrain_sft_test=7/10; all sample and denominator assertions passed")
