"""Independent, bounded CPU recheck of section 18.11; never load weights or train."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_nodes(filename, names, namespace):
    path = ART / "inputs" / filename
    tree = ast.parse(path.read_bytes())
    nodes = [
        node for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets))
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return [{"path": str(path.relative_to(ROOT)), "name": getattr(n, "name", "SPECIALS"),
             "line": n.lineno, "end_line": n.end_lineno} for n in nodes]


result = {"environment": {"python": sys.version, "executable": sys.executable,
    "platform": platform.platform(), "device": "cpu", "training": "none",
    "weights_loaded": "none"}, "original_fence": {}, "variants": {}, "raw_rechecks": {}}
raw = (ART / "code/fence-1.py").read_bytes()
for label, code in [("original", raw), ("string_2", raw.replace(
    b'teacher_answers = ["\xe4\xb8\x8d\xe7\x9f\xa5\xe9\x81\x93",', b'teacher_answers = ["2",', 1)),
    ("integer_2", raw.replace(b'teacher_answers = ["\xe4\xb8\x8d\xe7\x9f\xa5\xe9\x81\x93",',
                             b'teacher_answers = [2,', 1))]:
    ns, stdout = {}, io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, str(ART / "code/fence-1.py") + ":" + label, "exec"), ns)
        observed = {"stdout": stdout.getvalue(), "agreement": ns["agreement"],
            "checks": ns["checks"], "pass_rate": round(sum(ns["checks"]) / 3, 4)}
        assert ns["student_answers"] == ns["teacher_answers"]
        assert ns["student_answers"] is not ns["teacher_answers"]
        if label == "original":
            assert observed["agreement"] == 1.0 and observed["checks"] == [False, False, True]
            assert observed["pass_rate"] == 0.3333
            result["original_fence"] = observed
        else:
            assert observed["checks"] == [True, False, True] and observed["pass_rate"] == 0.6667
            result["variants"][label] = observed
    except AttributeError as exc:
        assert label == "integer_2" and "strip" in str(exc)
        result["variants"][label] = {"error_type": type(exc).__name__, "error": str(exc),
            "stdout_before_error": stdout.getvalue()}
assert result["original_fence"]["stdout"] == "教師一致率 1.0\n逐題合格 [False, False, True]\n任務合格率 0.3333\n"
result["variants"]["wording"] = {"exact_match": "不知道" == "資訊不足",
    "task_acceptable": ns["acceptable"]("不知道", None)}
assert result["variants"]["wording"] == {"exact_match": False, "task_acceptable": True}

raw_report = json.loads((ART / "inputs/distillation-original.json").read_bytes())
data = json.loads((ART / "inputs/style-dataset-original.json").read_bytes())
task = raw_report["results"]["tasks"]["style_transfer"]
assert sha(ART / "inputs/style-dataset-original.json") == task["data"]["sha256"]
for filename, original_path in [("compression-experiment-revision.py", "scripts/course_experiments/compression.py"),
    ("behavior-experiment-revision.py", "scripts/course_experiments/behavior.py"),
    ("tokenization-experiment-revision.py", "tiny_perceptron/tokenization.py"),
    ("common-experiment-revision.py", "scripts/course_experiments/common.py"),
    ("data-experiment-revision.py", "tiny_perceptron/data.py"),
    ("model-experiment-revision.py", "tiny_perceptron/model.py")]:
    assert sha(ART / "inputs" / filename) == raw_report["code_sha256"][original_path]
namespace = {"json": json}
locators = selected_nodes("data-experiment-revision.py", {"SPECIALS", "ByteTokenizer"}, namespace)
locators += selected_nodes("tokenization-experiment-revision.py", {"generation_report"}, namespace)
locators += selected_nodes("behavior-experiment-revision.py", {"_style_metrics"}, namespace)
tok = namespace["ByteTokenizer"]()
generation_report = namespace["generation_report"]
metrics = namespace["_style_metrics"]
families = {split: set(row["family"] for row in rows) for split, rows in data.items()}
assert {k: len(v) for k,v in data.items()} == task["data"]["counts"]
assert {k: len(v) for k,v in families.items()} == task["data"]["families"]
assert not any(families[a] & families[b] for a,b in [
    ("train", "validation"), ("train", "test"), ("validation", "test")])
result["raw_rechecks"]["dataset"] = {"records": {k:len(v) for k,v in data.items()},
    "families": {k:len(v) for k,v in families.items()}, "family_overlap": 0,
    "dataset_sha256": sha(ART / "inputs/style-dataset-original.json"),
    "test_arithmetic_families": len(set(r["family"] for r in data["test"] if "a" in r)),
    "test_clarification_records": sum("a" not in r for r in data["test"])}

budgets = []
for method in ["w32_ce", "w32_teacher_hard", "w32_ce_kl"]:
    run = task["runs"][method]
    training = run["training"]
    budgets.append({k: training[k] for k in ["steps", "optimizer_updates", "batch_size",
        "initialization_sha256", "batch_plan_sha256", "training_examples",
        "training_sequence_chunks", "effective_supervised_tokens"]} |
        {"parameter_count": run["storage"]["parameter_count"]})
    for split in ["validation", "test"]:
        ev = run[split]
        assert math.isclose(ev["nll_sum"] / ev["supervised_tokens"], ev["answer_nll"], abs_tol=1e-12)
assert budgets[0] == budgets[1] == budgets[2]
result["raw_rechecks"]["matched_budgets"] = budgets[0]
result["raw_rechecks"]["objective"] = {m: task["runs"][m]["training"]["objective"]
    for m in ["w32_ce", "w32_teacher_hard", "w32_ce_kl"]}

all_metrics, selected = {}, []
items = [("teacher", task["teacher_test"], task["teacher_style"])] + [
    (m, task["runs"][m]["test"], task["runs"][m]["style"])
    for m in ["w32_ce", "w32_teacher_hard", "w32_ce_kl"]]
for label, evaluation, saved_style in items:
    samples = copy.deepcopy(evaluation["generated_samples"])
    assert len(samples) == len(data["test"]) == evaluation["examples"] == 27
    for i, (sample, row) in enumerate(zip(samples, data["test"], strict=True)):
        question, expected = row["messages"][-2]["content"], row["messages"][-1]["content"]
        assert sample["question"] == question and sample["family"] == row["family"]
        assert sample["expected"] == expected
        ids = sample["generated_ids"]
        raw_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        assert sample["exact"] is (raw_ids == tok.encode(expected))
        visible = generation_report(tok, ids)
        sample.update(visible)
        sample["generated"] = visible["answer"]
        if row.get("style") == "json" and row["family"] == "2+2":
            parsed = json.loads(visible["answer"])
            assert set(parsed) == {"answer"} and type(parsed["answer"]) is int
            assert parsed["answer"] == (3 if label == "teacher" else 10)
            assert parsed["answer"] != row["a"] + row["b"]
            selected.append({"model": label, "test_index": i, "question": question,
                "expected": expected, "generated": visible["answer"],
                "generated_ids": ids, "parsed_value": parsed["answer"], "content_correct": False,
                "json_integer_format_valid": True})
    recomputed = metrics({"samples": samples}, data["test"])
    assert recomputed["rubric"] == saved_style["rubric"]
    content_count = sum(g["content_correct"] for g in recomputed["rubric"].values())
    assert content_count == saved_style["content_correct"]
    assert math.isclose(content_count / 27, saved_style["content_accuracy"], abs_tol=1e-12)
    all_metrics[label] = recomputed["rubric"]
result["raw_rechecks"]["rubrics_recomputed"] = all_metrics
result["raw_rechecks"]["selected_json_samples"] = selected
result["raw_rechecks"]["clarification_truth"] = [
    {"test_index": i, "family": r["family"], "question": r["messages"][-2]["content"],
     "expected": r["messages"][-1]["content"],
     "expected_ids": tok.encode(r["messages"][-1]["content"])}
    for i,r in enumerate(data["test"]) if "a" not in r]

def probe(text, row, invalid_id=None):
    ids = tok.encode(text) + [tok.eos_id]
    if invalid_id is not None:
        ids.insert(0, invalid_id)
    expected = row["messages"][-1]["content"]
    sample = {"generated": text, "generated_ids": ids,
        "exact": ids[:-1] == tok.encode(expected)}
    return metrics({"samples": [sample]}, [row])["rubric"]

json_row = next(r for r in data["test"] if r.get("style") == "json" and r["family"] == "2+2")
probes = {"correct_integer": probe('{"answer": 4}', json_row),
    "boolean": probe('{"answer": true}', json_row),
    "illegal_control": probe('{"answer": 4}', json_row, tok.bos_id)}
assert probes["correct_integer"]["json"]["content_correct"] == 1
assert probes["boolean"]["json"]["json_valid"] == 0
assert probes["illegal_control"]["json"]["json_valid"] == 0
date_row = next(r for r in data["test"] if "a" not in r)
probes["date_full_raw_match"] = probe(date_row["messages"][-1]["content"], date_row)
probes["date_correct_visible_illegal_control"] = probe(date_row["messages"][-1]["content"], date_row, tok.bos_id)
assert probes["date_full_raw_match"]["clarification"]["content_correct"] == 1
assert probes["date_correct_visible_illegal_control"]["clarification"]["content_correct"] == 0
result["raw_rechecks"]["rubric_probes"] = probes
result["method_ast_locators"] = locators
(ART / "execution/verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
