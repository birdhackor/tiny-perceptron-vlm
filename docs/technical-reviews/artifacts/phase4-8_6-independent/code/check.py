"""Bounded independent checks of section 8.6; no model training or inference."""

import ast
import contextlib
import hashlib
import io
import json
import platform
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "inputs/historical"
RESULT = json.loads((ROOT / "inputs/docs/course-experiments/results/style.json").read_text())
observations = {"environment": {"python": sys.version, "executable": sys.executable,
                               "platform": platform.platform(), "device": "cpu"}}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def original_nodes(relative, names):
    path = HIST / relative
    raw = path.read_bytes()
    assert sha(raw) == RESULT["code_sha256"][relative]
    tree = ast.parse(raw)
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    assert {node.name for node in selected} == set(names)
    return selected


# Run the exact original fence again, independently of the extraction helper.
fence = (ROOT / "execution/fence-1.py").read_bytes()
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    namespace = {}
    exec(compile(fence, "original-fence-1.py", "exec"), namespace)
expected_stdout = ("{'task': '確認日期', 'date': None} → 你希望哪一天？\n"
                   "{'task': '確認日期', 'date': '2026-10-03'} → 已確認日期：2026-10-03\n")
assert stream.getvalue() == expected_stdout
assert namespace["target"] == "已確認日期：2026-10-03"
assert type(namespace["requests"]) is list
assert all(type(row) is dict for row in namespace["requests"])
assert all(row["date"] is None or type(row["date"]) is str for row in namespace["requests"])
observations["original_fence"] = {"sha256": sha(fence), "stdout": stream.getvalue(),
                                   "types": "list[dict]; date NoneType or str; target str"}

# Preserve the original branch AST. Only substitute the input requests assignment.
original_tree = ast.parse(fence)
branch = original_tree.body[1:]
def apply_original_branch(row):
    stream = io.StringIO()
    namespace = {"requests": [row]}
    with contextlib.redirect_stdout(stream):
        exec(compile(ast.Module(body=branch, type_ignores=[]), "original-branch", "exec"), namespace)
    return namespace["target"]

variants = [({"task": "確認日期", "date": None}, "你希望哪一天？"),
            ({"task": "確認日期", "date": "2027-01-15"}, "已確認日期：2027-01-15"),
            ({"task": "確認日期", "date": ""}, "已確認日期："),
            ({"task": "確認日期", "date": "明天"}, "已確認日期：明天")]
observations["input_variants"] = []
for row, expected in variants:
    got = apply_original_branch(row)
    assert got == expected and type(got) is str
    observations["input_variants"].append({"input": row, "output": got})
for row, error_type in [({"task": "確認日期"}, KeyError),
                        ({"task": "確認日期", "date": 20261003}, TypeError)]:
    try:
        apply_original_branch(row)
    except error_type as error:
        observations["input_variants"].append({"input": row, "exception": type(error).__name__})
    else:
        raise AssertionError("Expected input-contract boundary exception")

# The exercise is a new handwritten extension, not a claimed original program.
def date_time_target(row):
    missing = [label for field, label in [("date", "日期"), ("time", "時間")]
               if row[field] is None]
    return "請提供" + "與".join(missing) + "。" if missing else (
        "已確認日期與時間：" + row["date"] + " " + row["time"])
observations["exercise_four_cases"] = []
for date, time, expected in [
    ("2026-10-03", "14:00", "已確認日期與時間：2026-10-03 14:00"),
    (None, "14:00", "請提供日期。"),
    ("2026-10-03", None, "請提供時間。"),
    (None, None, "請提供日期與時間。"),
]:
    row = {"date": date, "time": time}
    got = date_time_target(row)
    assert got == expected
    observations["exercise_four_cases"].append({"input": row, "output": got})

# Execute only original pure-data helpers and the bounded data-generator block.
# No imports of historical training code, no fit_lm, no model or checkpoint.
namespace = {"json": json, "random": random, "hashlib": hashlib,
             "ctx": SimpleNamespace(seed=RESULT["seed"])}
nodes = original_nodes("scripts/course_experiments/common.py", ["split_records", "records_sha256"])
nodes += original_nodes("scripts/course_experiments/behavior.py", ["_conversation", "_style_record", "_style_metrics"])
nodes += original_nodes("scripts/course_experiments/text.py", ["arithmetic_records"])
nodes += original_nodes("tiny_perceptron/data.py", ["ByteTokenizer"])
namespace["SPECIALS"] = ("<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "<image>", "<audio>", "<system>")
exec(compile(ast.Module(body=nodes, type_ignores=[]), "historical-pure-data-helpers", "exec"), namespace)
namespace["arithmetic"] = namespace["split_records"](namespace["arithmetic_records"](), seed=RESULT["seed"])
behavior = (HIST / "scripts/course_experiments/behavior.py").read_text()
run_style = next(node for node in ast.parse(behavior).body
                 if isinstance(node, ast.FunctionDef) and node.name == "run_style")
start = next(i for i, node in enumerate(run_style.body)
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id == "conditional")
end = next(i for i, node in enumerate(run_style.body[start:], start)
           if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
           and isinstance(node.value.func, ast.Name) and node.value.func.id == "write_json")
generator_nodes = run_style.body[start:end]
assert all(not isinstance(node, ast.Expr) for node in generator_nodes)
fragment = "\n".join(ast.get_source_segment(behavior, node) for node in generator_nodes) + "\n"
(ROOT / "code/historical-data-generator-fragment.py").write_text(fragment)
exec(compile(ast.Module(body=generator_nodes, type_ignores=[]), "historical-generator-fragment", "exec"), namespace)
conditional, date_parts = namespace["conditional"], namespace["date_parts"]
dataset_raw = (json.dumps(conditional, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
expected_dataset_sha = next(a["sha256"] for a in RESULT["artifacts"] if a["path"] == "dataset.json")
assert sha(dataset_raw) == expected_dataset_sha
(ROOT / "inputs/reconstructed-dataset.json").write_bytes(dataset_raw)
date_family_sets = {split: {row["family"] for row in rows} for split, rows in date_parts.items()}
assert len(namespace["dates"]) == 48
assert [len(date_parts[split]) for split in ["train", "validation", "test"]] == [38, 4, 6]
assert [len(date_family_sets[split]) for split in ["train", "validation", "test"]] == [19, 2, 3]
assert not date_family_sets["train"] & date_family_sets["validation"]
assert not date_family_sets["train"] & date_family_sets["test"]
assert not date_family_sets["validation"] & date_family_sets["test"]
assert date_family_sets["test"] == {"date-2026-10-01", "date-2026-10-04", "date-2026-10-21"}
observations["split_reconstruction"] = {"dataset_sha256": sha(dataset_raw),
    "original_dataset_sha256_match": True, "date_splits": {}, "combined_splits": {}}
for split, rows in date_parts.items():
    families = date_family_sets[split]
    assert all(sum(row["family"] == family for row in rows) == 2 for family in families)
    missing = [row for row in rows if "date=?" in row["messages"][0]["content"]]
    assert len(missing) == len(families)
    assert all(row["family"].removeprefix("date-") not in row["messages"][0]["content"] for row in missing)
    observations["split_reconstruction"]["date_splits"][split] = {
        "date_groups": len(families), "records": len(rows), "families": sorted(families),
        "missing_date_records": len(missing), "supplied_date_records": len(rows) - len(missing)}
for split, rows in conditional.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    assert sha(raw) == RESULT["results"]["data"][split]["sha256"]
    assert len(rows) == RESULT["results"]["data"][split]["records"]
    (ROOT / f"inputs/reconstructed-{split}.jsonl").write_bytes(raw)
    observations["split_reconstruction"]["combined_splits"][split] = {
        "records": len(rows), "sha256": sha(raw), "matches_original_saved_split_hash": True}
assert namespace["records_sha256"](conditional["train"]) == RESULT["results"]["training"]["records_sha256"]
assert RESULT["results"]["training"]["steps"] == 1000

# Independently inspect every saved date sample, raw token IDs and historical rubric.
tok = namespace["ByteTokenizer"]()
observations["date_evaluations"] = {}
for when in ["before", "after"]:
    observations["date_evaluations"][when] = {}
    for split in ["validation", "test"]:
        report = RESULT["results"][when][split]
        assert len(report["samples"]) == len(conditional[split]) == report["records"]
        exact_count = 0
        for row, sample in zip(conditional[split], report["samples"], strict=True):
            assert sample["messages"] == row["messages"][:-1]
            assert sample["expected"] == row["messages"][-1]["content"]
            generated = sample["generated_ids"]
            assert all(type(i) is int and 0 <= i < 264 for i in generated)
            assert generated[-1] == tok.eos_id and generated.count(tok.eos_id) == 1
            assert all(i >= len(namespace["SPECIALS"]) for i in generated[:-1])
            assert tok.decode(generated[:-1]) == sample["generated"]
            exact = generated[:-1] == tok.encode(sample["expected"])
            assert exact == sample["exact"]
            exact_count += exact
        assert exact_count == report["matches"]
        assert report["exact_match"] == exact_count / len(conditional[split])
        reconstructed_metrics = namespace["_style_metrics"]({"samples": json.loads(json.dumps(report["samples"]))}, conditional[split])["rubric"]
        assert reconstructed_metrics == report["rubric"]
        date_samples = [sample for sample in report["samples"] if sample.get("kind") == "clarification"]
        missing = [sample for sample in date_samples if "date=?" in sample["messages"][0]["content"]]
        supplied = [sample for sample in date_samples if "date=?" not in sample["messages"][0]["content"]]
        got = {"date_records": len(date_samples), "missing_denominator": len(missing),
               "missing_correct": sum(sample["exact"] for sample in missing),
               "supplied_denominator": len(supplied), "supplied_correct": sum(sample["exact"] for sample in supplied),
               "exact_correct": sum(sample["exact"] for sample in date_samples),
               "eos_count": sum(sample["eos"] for sample in date_samples),
               "samples": [{"input": sample["messages"][0]["content"],
                            "expected": sample["expected"], "generated": sample["generated"],
                            "exact": sample["exact"], "generated_ids": sample["generated_ids"]} for sample in date_samples]}
        observations["date_evaluations"][when][split] = got
test = observations["date_evaluations"]["after"]["test"]
assert (test["date_records"], test["missing_correct"], test["supplied_correct"], test["exact_correct"]) == (6, 3, 0, 3)
assert test["samples"][0]["input"] == "task=date;date=2026-10-01;confirm"
assert test["samples"][0]["generated"] == "已確認2026-10-10。"
assert observations["date_evaluations"]["before"]["test"]["exact_correct"] == 0
observations["scope"] = {
    "retraining": False, "model_inference": False, "weights_accessed": False,
    "numerical_tolerance": "Exact integer, UTF-8 text, token-ID, fraction and SHA-256 comparisons; no approximate floats needed.",
    "unit_and_denominator": "Dates are YYYY-MM-DD text, not timestamps. Counts are date groups and paired records, not tokens or batches. Test 3/6 = missing 3/3 plus supplied 0/3.",
    "date_input_contract": "Explicit date key; None sentinel or supplied string. Empty/invalid date strings are not validated and absent key raises KeyError.",
    "id_cue": "Missing-date prompts use id=day. Group family is not rendered as input; this does not establish absence of every indirect date cue.",
    "no_general_language_claim": "Only fixed task=date templates are measured, no arbitrary-language date extraction or calendar scheduling."}
output = json.dumps(observations, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
(ROOT / "execution/bounded-results.json").write_text(output)
print(output, end="")
print("PASS: original fence, six boundary variants, four exercise cases, original dataset and split hashes, historical date tokens and rubric.")
