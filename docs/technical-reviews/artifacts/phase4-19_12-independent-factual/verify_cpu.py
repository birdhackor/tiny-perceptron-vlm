"""Bounded CPU audit of existing records and handcrafted scoring cases; no model load/training."""
from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
from unittest.mock import patch
import ast
import contextlib
import hashlib
import io
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.capstone import build_dataset, digest, evaluate_rows, parse_action, calculator_runtime, expected_final

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
env = {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__),
       "torch_git_version": str(torch.version.git_version), "device": "cpu", "cuda_build": str(torch.version.cuda),
       "cuda_available": str(torch.cuda.is_available()), "platform": platform.platform(),
       "scope": "Only dataset construction, stored-record arithmetic, and mocked score-criterion cases. No weights loaded or new model scores."}
(OUT / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

inspection = {}
def raw(path):
    p = OUT / "original" / path
    return json.loads(p.read_bytes())
def inspect(path, pointers):
    o = raw(path)
    item = {"sha256": hashlib.sha256((OUT / "original" / path).read_bytes()).hexdigest(),
            "top_level_key_types": {k:type(v).__name__ for k,v in o.items()}, "pointers": {}}
    for pointer in pointers:
        v = o
        for k in pointer.strip("/").split("/"):
            v = v[int(k)] if isinstance(v, list) else v[k]
        item["pointers"][pointer] = v
    inspection[path] = item
    return o

capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile((OUT / "original-fence.py").read_bytes(), "course/chapters/19.md#19.12:original-fence", "exec"), {})
(OUT / "original-fence.stdout.txt").write_text(capture.getvalue())
print("ORIGINAL FENCE\n" + capture.getvalue())

splits, manifest = build_dataset()
data = inspect("docs/course-experiments/capstone-evidence/deployment/data.json", ["/manifest/version", "/manifest/seed", "/manifest/counts", "/manifest/sha256", "/manifest/families", "/splits/test"])
assert splits == data["splits"]
assert manifest["sha256"] == data["manifest"]["sha256"]
assert len(splits["test"]) == 90
for s, rows in splits.items():
    assert digest(rows) == data["manifest"]["sha256"][s]
assert all(not (set(manifest["families"][a][category]) & set(manifest["families"][b][category]))
           for a,b in [("train","validation"),("train","test"),("validation","test")]
           for category in manifest["families"][a])
counts = Counter(r["task"] for r in splits["test"])
print("DATA", json.dumps({"test_count":len(splits["test"]), "task_counts":dict(sorted(counts.items())), "split_counts":{s:len(rows) for s,rows in splits.items()}}, ensure_ascii=False))

test = inspect("docs/course-experiments/capstone-evidence/deployment/test-joint.json", ["/count", "/action_correct", "/end_to_end_correct", "/by_task", "/checkpoint_sha256", "/protocol", "/records"])
rows = {r["id"]:r for r in data["splits"]["test"]}
assert len(rows) == len(test["records"]) == len({r["id"] for r in test["records"]}) == 90
assert set(rows) == {r["id"] for r in test["records"]}
group = defaultdict(lambda: {"count":0,"action_correct":0,"end_to_end_correct":0})
wrong_calculator = []
for record in test["records"]:
    row = rows[record["id"]]
    assert row["task"] == record["task"] and row["answer"] == record["expected_action"]
    assert expected_final(row) == record["expected_final"]
    trace = record["action_trace"]
    action = parse_action(trace)
    assert action == record["parsed_action"]
    correct_action = bool(trace["eos"] and trace["raw"] == row["answer"])
    answer = None
    if action["status"] in ("direct", "ask"):
        answer = action["content"]
    elif action["status"] == "tool":
        runtime = calculator_runtime(action, row["available"])
        assert record["runtime"] == runtime
        if runtime["status"] == "ok":
            parsed_final = parse_action(record["final_trace"])
            if parsed_final["status"] == "direct": answer = parsed_final["content"]
    assert answer == record["answer"]
    correct_final = correct_action and answer == expected_final(row)
    assert correct_action == record["action_correct"] and correct_final == record["end_to_end_correct"]
    result = group[row["task"]]
    result["count"] += 1
    result["action_correct"] += int(correct_action)
    result["end_to_end_correct"] += int(correct_final)
    if row["task"] == "calculator" and correct_action and not correct_final:
        wrong_calculator.append({"id":row["id"],"first":trace["raw"],"runtime":record["runtime"],"final":record["final_trace"]["raw"],"expected_final":expected_final(row)})
assert dict(group) == test["by_task"]
assert sum(g["count"] for g in group.values()) == test["count"]
assert sum(g["action_correct"] for g in group.values()) == test["action_correct"] == 80
assert sum(g["end_to_end_correct"] for g in group.values()) == test["end_to_end_correct"] == 78
assert {k:group[k] for k in ("calculator","image_color","image_shape","audio")} == {
    "calculator":{"count":12,"action_correct":12,"end_to_end_correct":10},
    "image_color":{"count":9,"action_correct":9,"end_to_end_correct":9},
    "image_shape":{"count":9,"action_correct":0,"end_to_end_correct":0},
    "audio":{"count":6,"action_correct":6,"end_to_end_correct":6}}
colors = Counter(rows[r["id"]]["image"]["color"] for r in test["records"] if r["task"] == "image_color")
shape_inputs = Counter(rows[r["id"]]["image"]["shape"] for r in test["records"] if r["task"] == "image_shape")
shape_outputs = Counter(r["action_trace"]["raw"] for r in test["records"] if r["task"] == "image_shape")
pitch = Counter(rows[r["id"]]["audio"]["pitch"] for r in test["records"] if r["task"] == "audio")
assert colors == {"green":9} and shape_inputs == {"square":9} and shape_outputs == {"DIRECT:circle":9} and pitch == {"low":3,"high":3}
print("JOINT_RECALC", json.dumps({"by_task":dict(group),"image_colors":dict(colors),"shape_inputs":dict(shape_inputs),"shape_outputs":dict(shape_outputs),"pitch_counts":dict(pitch),"calculator_failed_final":wrong_calculator}, ensure_ascii=False))

# Handcrafted traces exercise the original score implementation, not a trained model.
row = {"id":"manual-calculator", "family":"manual", "task":"calculator", "answer":"TOOL:calculator:1+2", "available":True}
def scored(first, second=None):
    calls = [[{"raw":first[0],"eos":first[1]}]]
    calls.append([] if second is None else [{"raw":second[0],"eos":second[1]}])
    with patch("tiny_perceptron.capstone.generate_traces", side_effect=calls):
        return evaluate_rows(None, [row])["records"][0]
cases = []
for title,first,second,expected in [
    ("request-and-final-match",("TOOL:calculator:1+2",True),("DIRECT:3",True),(True,True)),
    ("right-request-wrong-final",("TOOL:calculator:1+2",True),("DIRECT:8",True),(True,False)),
    ("swapped-ordered-arguments",("TOOL:calculator:2+1",True),("DIRECT:3",True),(False,False)),
    ("wrong-tool-name",("TOOL:other:1+2",True),None,(False,False)),
    ("missing-first-EOS",("TOOL:calculator:1+2",False),None,(False,False)),
    ("missing-final-EOS",("TOOL:calculator:1+2",True),("DIRECT:3",False),(True,False))]:
    r = scored(first,second)
    actual = (r["action_correct"], r["end_to_end_correct"])
    assert actual == expected
    cases.append({"case":title,"expected":expected,"observed":actual})
direct_row = {"id":"manual-direct","family":"manual","task":"image_shape","answer":"DIRECT:square","available":True}
with patch("tiny_perceptron.capstone.generate_traces",side_effect=[[{"raw":"DIRECT:circle","eos":True}],[]]):
    wrong = evaluate_rows(None,[direct_row])["records"][0]
assert wrong["parsed_action"]["status"] == "direct" and not wrong["action_correct"] and not wrong["end_to_end_correct"]
print("HANDCRAFTED_SCORING",json.dumps(cases))

pooled = Fraction(12+0,12+6)
equal_tasks = (Fraction(12,12)+Fraction(0,6))/2
assert pooled == Fraction(2,3) and equal_tasks == Fraction(1,2)
per_task_weights = [Fraction(12,18),Fraction(6,18)]
assert sum(w*a for w,a in zip(per_task_weights,[Fraction(1),Fraction(0)])) == pooled
print("HYPOTHETICAL_AVERAGES",json.dumps({"pooled":str(pooled),"equal_tasks":str(equal_tasks),"pooled_task_weights":[str(w) for w in per_task_weights],"equal_task_weights":["1/2","1/2"],"each_question_weight_pooled":"1/18","each_question_weight_equal_tasks":["1/24","1/12"],"scope":"Human hypothetical scores, not model measurement"}))

selection = inspect("docs/course-experiments/capstone-selection.json", ["/selected_stage","/selected_before_test_generation","/selected_at_utc","/criterion","/candidates"])
assert selection["selected_stage"] == "joint" and selection["selected_before_test_generation"] is True
for stage, filename in [("joint","capstone_joint.json"),("dpo","capstone_preference.json")]:
    path = "docs/course-experiments/results/"+filename
    candidate = inspect(path,["/revision","/seed","/results/validation_summary","/results/test_evaluated","/results/inference_export","/results/data_manifest/sha256","/modal/run_id"])
    recorded = selection["candidates"][stage]
    assert inspection[path]["sha256"] == recorded["evidence_sha256"]
    assert candidate["results"]["validation_summary"]["count"] == recorded["validation_count"] == 84
    assert candidate["results"]["validation_summary"]["end_to_end_correct"] == recorded["validation_end_to_end_correct"]
    assert candidate["results"]["test_evaluated"] is False
    assert candidate["results"]["inference_export"]["sha256"] == recorded["checkpoint_sha256"]
    assert candidate["results"]["data_manifest"]["sha256"] == manifest["sha256"]
assert selection["candidates"]["joint"]["checkpoint_sha256"] == test["checkpoint_sha256"]
assert selection["candidates"]["joint"]["validation_end_to_end_correct"] == 75 > 71 == selection["candidates"]["dpo"]["validation_end_to_end_correct"]
print("SELECTION",json.dumps({"joint_validation":"75/84","dpo_validation":"71/84","both_test_evaluated":False,"selected_stage":"joint","joint_checkpoint":test["checkpoint_sha256"]}))

student = inspect("docs/course-experiments/results/capstone_student.json", ["/revision","/device","/seed","/torch_version","/python_version","/results/teacher_checkpoint_sha256","/results/recipe_frozen_before_test","/results/steps_per_branch","/results/steps","/results/requested_steps","/results/schedule_completed","/results/branches/kd/parameters","/results/branches/kd/teacher_checkpoint_sha256","/results/evaluations/test_kd","/results/evaluations/test_kd_ptq4"])
assert student["results"]["teacher_checkpoint_sha256"] == selection["candidates"]["dpo"]["checkpoint_sha256"] != test["checkpoint_sha256"]
assert student["results"]["recipe_frozen_before_test"] is True
assert student["results"]["branches"]["kd"]["parameters"]["config"]["experts"] == 0
student_test = inspect("docs/course-experiments/capstone-evidence/student/test-kd.json",["/count","/action_correct","/end_to_end_correct","/by_task","/records"])
quant_test = inspect("docs/course-experiments/capstone-evidence/student/test-kd-ptq4.json",["/count","/action_correct","/end_to_end_correct","/by_task","/records"])
assert student_test["count"] == quant_test["count"] == 90
assert student_test["end_to_end_correct"] == quant_test["end_to_end_correct"] == 61
quant_by_id = {r["id"]:r for r in quant_test["records"]}
assert set(quant_by_id) == {r["id"] for r in student_test["records"]} == set(rows)
differences = []
for r in student_test["records"]:
    q = quant_by_id[r["id"]]
    if r["action_trace"]["generated_ids"] != q["action_trace"]["generated_ids"]:
        differences.append({"id":r["id"],"task":r["task"],"fp32_raw":r["action_trace"]["raw"],"int4_raw":q["action_trace"]["raw"],"fp32_generated_ids":r["action_trace"]["generated_ids"],"int4_generated_ids":q["action_trace"]["generated_ids"]})
assert differences
print("STUDENT",json.dumps({"teacher_checkpoint":student["results"]["teacher_checkpoint_sha256"],"dense_experts":0,"test_kd":"61/90","test_kd_int4":"61/90","first_trace_token_differences":len(differences),"first_difference":differences[0]},ensure_ascii=False))

# Inspect only necessary original method functions by AST, with exact source locations.
code_inspection = {}
for path, names in {"tiny_perceptron/capstone.py":["build_dataset","modality_tensors","evaluate_rows","parse_action","calculator_runtime","expected_final"],"scripts/course_experiments/capstone_student.py":["_train_student","run"],"scripts/course_experiments/capstone_deployment.py":["benchmark_generation"],"scripts/course_experiments/capstone.py":["run_deployment"]}.items():
    p = OUT / "original" / path
    tree = ast.parse(p.read_bytes())
    code_inspection[path] = {"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"functions":{n.name:{"first_line":n.lineno,"end_line":n.end_lineno} for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}}
(OUT / "inspection.json").write_text(json.dumps({"json_sources":inspection,"ast_source_methods":code_inspection},ensure_ascii=False,indent=2)+"\n")
print("AUDIT_PASS: existing records and denominator arithmetic verified; no GPU, model load, training, or dataset download.")
