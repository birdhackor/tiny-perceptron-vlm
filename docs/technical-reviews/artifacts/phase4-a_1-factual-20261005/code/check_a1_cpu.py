"""Independent bounded CPU checks; no model inference, training or data download."""
import ast
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
assert torch.version.cuda is None and not torch.cuda.is_available()


def extract_definitions(path, names, namespace):
    tree = ast.parse(path.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    tree = ast.Module(body=nodes, type_ignores=[])
    exec(compile(tree, str(path), "exec"), namespace)


namespace = {"__name__": "__main__"}
print("ORIGINAL FENCE BEGIN")
exec(compile((ROOT / "code/original-fence-1.py").read_bytes(), "original-fence-1.py", "exec"), namespace)
print("ORIGINAL FENCE END")
question = namespace["question"]
assert namespace["zero"] == question
assert namespace["few"] == "示例：1→2；2→4。\n" + question
assert namespace["changed"] == "示例：1→3；2→6。\n" + question
assert all(p.endswith(question) for p in [namespace["zero"], namespace["few"], namespace["changed"]])
assert "；".join([]) == ""
reversed_prompt = "示例：" + "；".join(reversed(namespace["examples"])) + "。\n" + question
assert reversed_prompt == "示例：2→4；1→2。\n" + question
linear = [2 * 3, 3 * 3]
alternative = [2 * x + (x - 1) * (x - 2) for x in (1, 2, 3)]
assert linear == [6, 9] and alternative == [2, 4, 8]
assert len("->") == 2 and len("->".encode("utf-8")) == 2
assert len("→") == 1 and len("→".encode("utf-8")) == 3
assert all(len(chr(i).encode("utf-8")) == 1 for i in range(128))

ns = {"random": random, "hashlib": hashlib, "json": json}
extract_definitions(ROOT / "code/applications.py", ["_conversation", "_icl_question", "_icl_records", "_prompt_ids"], ns)
extract_definitions(ROOT / "code/common.py", ["split_records", "records_sha256"], ns)
ns["SPECIALS"] = ("<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "<image>", "<audio>", "<system>")
extract_definitions(ROOT / "code/data.py", ["ByteTokenizer"], ns)
raw_path = ROOT / "raw/rag-original-result.json"
record = json.loads(raw_path.read_bytes())
for original, basename in [("scripts/course_experiments/applications.py", "applications.py"), ("scripts/course_experiments/common.py", "common.py"), ("tiny_perceptron/data.py", "data.py")]:
    assert hashlib.sha256((ROOT / "code" / basename).read_bytes()).hexdigest() == record["code_sha256"][original]
splits = ns["split_records"](ns["_icl_records"](), record["seed"])
split_check = {}
family_sets = {}
for name, rows in splits.items():
    families = sorted({row["family"] for row in rows})
    actual = {"records": len(rows), "families": len(families), "sha256": ns["records_sha256"](rows)}
    assert actual == record["results"]["icl_split"][name]
    family_sets[name] = set(families)
    split_check[name] = {**actual, "family_ids": families}
assert not family_sets["train"] & family_sets["validation"]
assert not family_sets["train"] & family_sets["test"]
assert not family_sets["validation"] & family_sets["test"]
assert {r["value"] for r in splits["test"]} == {3}
assert len(ns["_icl_records"]()) == 168
for row in ns["_icl_records"]():
    assert row["messages"][0]["content"].isascii()
    expected = "UNKNOWN" if row["offset"] is None else str((row["value"] + row["offset"]) % 10)
    assert row["messages"][-1]["content"] == expected
tok = ns["ByteTokenizer"]()
samples = record["results"]["icl_samples_same_model_weights"]
assert len(samples) == 4
expected_modes = [("zero_shot", None, False, "UNKNOWN"), ("few_shot", 2, False, "9"), ("changed_examples", 5, False, "9"), ("reversed_examples", 2, True, "8")]
sample_check = []
for row, (mode, offset, reverse, observed) in zip(samples, expected_modes, strict=True):
    assert row["mode"] == mode and row["family"] == "map:3"
    g = row["generation"]
    prompt = ns["_icl_question"](3, offset, reverse)
    assert g["messages"] == [{"role": "user", "content": prompt}]
    assert g["input_ids"] == ns["_prompt_ids"](g["messages"], tok)
    assert g["input_tokens"] == len(g["input_ids"])
    assert g["temperature"] == 0.0 and g["candidate_count"] == 1 and g["max_new_tokens"] == 16
    assert len(g["samples"]) == 1
    s = g["samples"][0]
    assert s["generated"] == observed
    assert s["generated_ids"] == tok.encode(observed) + [tok.eos_id]
    assert s["eos"] is True and s["invalid_special_tokens"] == []
    assert s["generated_tokens"] == g["generated_tokens"] == len(s["generated_ids"])
    assert g["forward_input_tokens"] == g["input_tokens"] + s["generated_tokens"] - 1
    expected = "UNKNOWN" if offset is None else str((3 + offset) % 10)
    correct = observed == expected and not s["invalid_special_tokens"]
    assert row["expected"] == expected and row["correct"] == correct
    sample_check.append({"mode": mode, "prompt": prompt, "expected": expected, "observed": observed, "correct_recomputed": correct, "input_tokens": g["input_tokens"], "generated_ids": s["generated_ids"], "eos": s["eos"]})
assert sum(x["correct_recomputed"] for x in sample_check[1:]) == 0
assert sample_check[1]["observed"] != sample_check[3]["observed"]
train = record["results"]["training"]
assert train["records"] == record["results"]["split"]["train"]["records"] + len(splits["train"])
assert train["steps"] == train["planned_steps"] == 1000 and train["step_scale"] == 1.0
facts = {
    "python": sys.version,
    "environment": {"python_executable": sys.executable, "python_prefix": sys.prefix, "torch": str(torch.__version__), "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available())},
    "original_fence_checks": "Three expected strings, identical final question, no model invocation.",
    "bounded_variants": {"reverse_order": reversed_prompt, "empty_join": "；".join([])},
    "paper_expectations": {"double_3": linear[0], "triple_3": linear[1], "alternative_fits_two_examples": alternative},
    "encoding": {"ascii_range": [0, 127], "ascii_values_checked": 128, "ascii_utf8_bytes_each": 1, "ascii_arrow_characters": 2, "ascii_arrow_bytes": 2, "unicode_arrow_characters": 1, "unicode_arrow_bytes": 3},
    "split_recomputed_from_original_functions": split_check,
    "sample_measurements_recomputed": sample_check,
    "training_record_composition": {"rag_train_records": record["results"]["split"]["train"]["records"], "icl_train_records": len(splits["train"]), "combined_records": train["records"], "recorded_steps": train["steps"]},
    "rag_original_result_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    "execution_scope": "Original string fence and deterministic records/encoding/measurement checks only. No inference with any trained model, no checkpoint load, no training, no dataset download.",
}
(ROOT / "cpu-verification.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(facts, ensure_ascii=False, indent=2))
print("ALL CHECKS PASSED")
