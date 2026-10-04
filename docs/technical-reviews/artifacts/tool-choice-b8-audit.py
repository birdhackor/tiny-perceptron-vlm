"""B.8 reviewer audit; deterministic examples and recorded-trace replay, no training."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import platform
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments import applications, tool_choice
from tiny_perceptron.retrieval import call_tool


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


raw = (ROOT / "course/chapters/0B.md").read_text()
section = raw[raw.index("## B.8 "):]
code = section.split("```python\n", 1)[1].split("```", 1)[0]


def execute(text):
    out, namespace = io.StringIO(), {}
    with contextlib.redirect_stdout(out):
        exec(compile(text, "B.8:python-fence", "exec"), namespace)
    return out.getvalue(), namespace


original_out, ns = execute(code)
assert original_out.splitlines() == [
    "直接回答平均損失 1.0", "工具流程平均損失 0.12", "本例選擇 TOOL"
]
exercise_code = code.replace("direct_accuracy = 0.90", "direct_accuracy = 0.999")
assert exercise_code != code
exercise_out, ex = execute(exercise_code)
assert exercise_out.splitlines() == [
    "直接回答平均損失 0.01", "工具流程平均損失 0.12", "本例選擇 DIRECT"
]
for value, exact in [(ns["direct_loss"], Decimal("1")),
                     (ns["tool_loss"], Decimal("0.12")),
                     (ex["direct_loss"], Decimal("0.01"))]:
    assert abs(Decimal(str(value)) - exact) < Decimal("1e-12")
cost = Decimal("10")
tool_extra = Decimal("0.02")
threshold = Decimal("0.99") - tool_extra / cost
assert threshold == Decimal("0.988")
before_round = ns["direct_loss"]
rounded = round(before_round, 3)
assert ns["direct_loss"] == before_round

rows = tool_choice.build_records()
splits = tool_choice.split_records(rows)
labels = sorted({r["messages"][-1]["content"] for r in rows})
assert labels == ["ASK", "DIRECT", "TOOL"]
assert all(r["messages"][-1]["content"] == r["expected_action"] for r in rows)
families = {name: {r["family"] for r in data} for name, data in splits.items()}
assert not (families["train"] & families["validation"])
assert not (families["train"] & families["test"])
assert not (families["validation"] & families["test"])
router_tree = ast.parse((ROOT / "scripts/course_experiments/tool_choice.py").read_text())
assert not any(isinstance(n, ast.Name) and n.id == "call_tool" for n in ast.walk(router_tree))
report_path = ROOT / "docs/course-experiments/results/tool_choice.json"
router_result = json.loads(report_path.read_text())["results"]
assert router_result["reproducibility"]["script_sha256"] == sha(ROOT / "scripts/course_experiments/tool_choice.py")
assert "direct_accuracy" not in router_result and "tool_accuracy" not in router_result

tools_path = ROOT / "docs/course-experiments/results/tools.json"
recorded_tools = json.loads(tools_path.read_text())
for dependency in ("scripts/course_experiments/applications.py", "tiny_perceptron/retrieval.py"):
    assert recorded_tools["code_sha256"][dependency] == sha(ROOT / dependency)
stored = next(s for s in recorded_tools["results"]["samples"]
              if s["question"] == "CALC:multiply(9,9)")
assert stored["trace"][0]["tool_result"] == call_tool(stored["trace"][0]["parsed"]) == 81
assert stored["trace"][1]["generation"]["samples"][0]["generated"] == '{"done":true,"answer":8}}'
record = next(r for r in applications._tool_records()
              if r["messages"][1]["content"] == stored["question"])


def replay(generations):
    iterator = iter(copy.deepcopy(generations))
    original_sample = applications._sample
    applications._sample = lambda *args, **kwargs: next(iterator)
    try:
        return applications._tool_episode(None, record, None, max_steps=3)
    finally:
        applications._sample = original_sample


generations = [ev["generation"] for ev in stored["trace"]]
replayed = replay(generations)
assert replayed["trace"][0]["tool_result"] == 81
assert replayed["actual_tool_calls"] == 1
assert replayed["status"] == "invalid_request" and replayed["correct"] is False
injected_generations = copy.deepcopy(generations)
injected_generations[1]["samples"][0]["generated"] = '{"done":true,"answer":8}'
injected = replay(injected_generations)
assert injected["status"] == "done" and injected["answer"] == 8
assert injected["answer_correct"] is False and injected["correct"] is False


def compact_episode(ep):
    result = {key: ep[key] for key in ("question", "expected", "status", "answer",
        "answer_correct", "required_tool_behavior", "correct", "actual_tool_calls")}
    result["trace"] = [
        {**{key: ev[key] for key in ("step", "executed", "parsed", "tool_result",
             "request_matches_question", "error") if key in ev},
         "generated": ev["generation"]["samples"][0]["generated"]}
        for ev in ep["trace"]
    ]
    return result


result = {
    "reviewer_task": "/root/technical_tool_b8",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": "cpu", "cwd": str(ROOT)},
    "section_sha256": hashlib.sha256(section.encode()).hexdigest(),
    "source_code": code,
    "original_stdout": original_out,
    "original_raw_losses": {key: ns[key] for key in ("direct_loss", "tool_loss")},
    "exercise_change": "Only direct_accuracy 0.90 -> 0.999; all other source lines unchanged",
    "exercise_stdout": exercise_out,
    "exercise_raw_losses": {key: ex[key] for key in ("direct_loss", "tool_loss")},
    "numeric_tolerance": "absolute difference < 1e-12; printed lines match exactly",
    "derived_equal_loss_direct_accuracy": str(threshold),
    "round_probe": {"original_float": before_round, "rounded_return": rounded,
                    "original_variable_unchanged": ns["direct_loss"] == before_round},
    "router_audit": {"labels": labels, "records": len(rows),
       "split_counts": {k: len(v) for k, v in splits.items()},
       "family_sets_disjoint": True, "calls_calculator": False,
       "reported_direct_accuracy_or_tool_accuracy": False,
       "policy": tool_choice.POLICY,
       "scope": "Rebuilt fixed routing labels and inspected source. No model training or confidence estimation."},
    "stored_formal_episode": compact_episode(stored),
    "stored_formal_dependencies_match_current_hashes": True,
    "recorded_trace_replay": compact_episode(replayed),
    "injected_well_formed_wrong_final": compact_episode(injected),
    "replay_scope": "Recorded formal completions replayed through current parser and real local multiply."
                    " The second replay injects a well-formed wrong final. Neither run generates new model tokens,"
                    " retrains a model, or estimates a new model success rate.",
    "file_sha256": {p: sha(ROOT / p) for p in [
        "scripts/course_experiments/tool_choice.py", "scripts/course_experiments/applications.py",
        "tiny_perceptron/retrieval.py", "docs/course-experiments/results/tool_choice.json",
        "docs/course-experiments/results/tools.json"]},
    "checks_completed": "original fence; exercise-only edit; Decimal verification; fixed router labels;"
                        " disjoint family split; recorded calculator/final mismatch; current parser/pipeline replay",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
