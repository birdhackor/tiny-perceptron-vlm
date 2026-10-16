"""C.7 independent numeric, finite-policy identity and fixed-report checks; no training."""

import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices
from scripts.course_experiments.applications import (
    _FinitePolicy,
    _POLICY_ACTIONS,
    _policy_features,
    _policy_reward,
    _reasoning_records,
)
from scripts.course_experiments.common import Context, records_sha256, split_records


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


body = lesson_slices((ROOT / "course/chapters/0C.md").read_text())["C.7"]
freeze = json.loads((ROOT / "docs/course-revision-20261016/freeze-08.json").read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "C.7")
assert hashlib.sha256(body.encode()).hexdigest() == page["source_sha256"]
assert body == (ROOT / page["snapshot"]).read_text()
program = re.search(r"```python\n(.*?)```", body, re.S)[1]
cases = []
for reward in [1.0, 0.0]:
    source = program.replace("reward, baseline = 1.0, 0.5", f"reward, baseline = {reward}, 0.5")
    namespace = {}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(source, "frozen-C.7.py", "exec"), namespace)
    advantage = reward - 0.5
    expected_grad = [advantage / 2, -advantage / 2]
    expected_logits = [-0.1 * value for value in expected_grad]
    expected_p1 = 1 / (1 + math.exp(expected_logits[0] - expected_logits[1]))
    assert namespace["logits"].tolist() == [0.0, 0.0]
    assert namespace["before"].tolist() == [0.5, 0.5]
    assert namespace["logits"].grad.tolist() == expected_grad
    assert torch.allclose(namespace["updated"], torch.tensor(expected_logits), atol=1e-8, rtol=0)
    assert abs(float(namespace["after"][1]) - expected_p1) < 1e-7
    assert not namespace["updated"].requires_grad
    assert not namespace["before"].requires_grad
    cases.append({"reward": reward, "baseline": 0.5, "advantage": advantage,
                  "stdout": printed.getvalue(), "expected_probability_action_1_float64": expected_p1,
                  "observed_probability_action_1": float(namespace["after"][1]),
                  "original_logits_remain_unchanged": True, "updated_requires_grad": False})

report_path = ROOT / "docs/course-experiments/results/reasoning.json"
historical = json.loads(report_path.read_text())
reinforce = historical["results"]["reinforce"]
splits = split_records(_reasoning_records(), 42)
for split, rows in splits.items():
    assert len(rows) == historical["results"]["split"][split]["records"]
    assert records_sha256(rows) == historical["results"]["split"][split]["sha256"]
assert {r["family"] for r in splits["train"]}.isdisjoint({r["family"] for r in splits["test"]})
assert reinforce["strict"]["before"] == reinforce["weak_proxy"]["before"]
checked = {}
for label in ["strict", "weak_proxy"]:
    branch = reinforce[label]
    rows = branch["after"]["samples"]
    assert len(rows) == 24 and all(len(r["samples"]) == 16 for r in rows)
    assert {r["question"] for r in rows} == {r["question"] for r in splits["test"]}
    samples = [s for r in rows for s in r["samples"]]
    assert all(len(r["probabilities"]) == 17 for r in rows)
    assert all(r["greedy_action"] == _POLICY_ACTIONS[max(range(17), key=lambda j:r["probabilities"][j])] for r in rows)
    counts = {"greedy_accuracy": sum(r["greedy_strict_correct"] for r in rows),
              "sample_accuracy": sum(s["strict_reward"] for s in samples),
              "sample_proxy_reward": sum(s["proxy_reward"] for s in samples),
              "enumeration_action_rate": sum(s["action_id"] == 16 for s in samples)}
    for metric, numerator in counts.items():
        denominator = 24 if metric == "greedy_accuracy" else 384
        assert branch["after"][metric] == {"numerator": numerator, "denominator": denominator, "rate": numerator/denominator}
    assert branch["training"]["steps"] == 1200
    assert branch["training"]["sampled_actions"] == 1200 * 64
    assert branch["training"]["effective_tokens"] == 0
    assert counts["greedy_accuracy"] == 0
    if label == "weak_proxy":
        assert counts["sample_proxy_reward"] == 384 and counts["sample_accuracy"] == 0
        assert counts["enumeration_action_rate"] == 384
    checked[label] = {"recomputed_counts": counts, "distinct_test_questions": len(rows),
                      "draws_per_question": 16, "total_draws": len(samples),
                      "steps": 1200, "training_action_draws": 76800, "autoregressive_tokens": 0,
                      "first_question": {k:rows[0][k] for k in ["question", "truth", "greedy_action"]}}

ctx = Context("cpu", ROOT / "work/C.7-unused", ROOT / "work", ROOT / "assets/training", seed=42)
features = _policy_features([{"a":2,"b":4,"c":0}], ctx)
policy = _FinitePolicy()
assert features.shape == (1,18) and features.nonzero().tolist() == [[0,2],[0,10],[0,12]]
with torch.no_grad():
    scores = policy(features)
assert scores.shape == (1,17)
assert _POLICY_ACTIONS == [str(i) for i in range(16)] + [" ".join(str(i) for i in range(16))]
for truth in range(16):
    assert _policy_reward(16, truth) == 0 and _policy_reward(16, truth, proxy=True) == 1
    assert _policy_reward(truth, truth) == 1

commands = []
for args in [[sys.executable,"scripts/course_experiments/run.py","--help"],
             [sys.executable,"scripts/course_experiments/run.py","--list-assets","reasoning"],
             ["git","lfs","version"], ["git","lfs","pull","--help"]]:
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0
    commands.append({"command": " ".join(args), "returncode": r.returncode,
                     "stdout": r.stdout if len(r.stdout)<2300 else r.stdout[:900]+"\n[truncated; help-only check]"})
assert commands[1]["stdout"].strip() == "assets/training/gsm8k-v1.tar.gz"

source_hashes = {}
historical_function_checks = {}
for file in ["scripts/course_experiments/applications.py","scripts/course_experiments/common.py","scripts/course_experiments/run.py"]:
    current_hash = digest(ROOT/file)
    item = {"current_sha256":current_hash,"original_report_recorded_sha256":historical["code_sha256"][file],
            "current_matches_historical":current_hash==historical["code_sha256"][file]}
    original = subprocess.run(["git","show", historical["revision"]+":"+file],cwd=ROOT,capture_output=True)
    if original.returncode == 0:
        item["historical_commit_blob_sha256"] = hashlib.sha256(original.stdout).hexdigest()
        assert item["historical_commit_blob_sha256"] == historical["code_sha256"][file]
        if file.endswith("applications.py"):
            old_text, new_text = original.stdout.decode(), (ROOT/file).read_text()
            old_nodes = {n.name: ast.get_source_segment(old_text,n) for n in ast.parse(old_text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            new_nodes = {n.name: ast.get_source_segment(new_text,n) for n in ast.parse(new_text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            for name in ["_FinitePolicy","_policy_features","_policy_reward","_evaluate_policy","_reinforce","_reasoning_records","run_reasoning"]:
                assert old_nodes[name] == new_nodes[name]
                historical_function_checks[name] = {"source_identical":True,"source_sha256":hashlib.sha256(old_nodes[name].encode()).hexdigest()}
    source_hashes[file]=item

out = {"reviewer_task":"/root/tech_20261016_c_7", "scope":"No training. Original frozen Python block executed twice; historical L4 result JSON inspected and its action-level denominators recomputed locally.",
       "command":".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/C.7-verify.py",
       "environment":{"python":platform.python_version(),"torch":str(torch.__version__),"device":"cpu","platform":platform.platform()},
       "frozen_source_sha256":page["source_sha256"],"numeric_cases":cases,
       "historical_report":{"path":str(report_path.relative_to(ROOT)),"sha256":digest(report_path),
                            "revision":historical["revision"],"device":historical["device"],"gpu":historical["gpu"],
                            "torch":historical["torch_version"],"python":historical["python_version"],"seed":historical["seed"],
                            "status":historical["status"],"step_scale":historical["step_scale"],
                            "split":historical["results"]["split"],"checked":checked,"same_before_records":True},
       "finite_policy_forward":{"input_shape":list(features.shape),"nonzero_indices":features.nonzero().tolist(),"output_shape":list(scores.shape),"actions":_POLICY_ACTIONS,"tested_truths":list(range(16))},
       "read_only_cli_commands":commands,"source_hashes":source_hashes,"historical_function_checks":historical_function_checks,
       "result":"All assertions passed. Historical metrics were checked, not retrained; CPU calculations do not measure GPU training effects."}
target = ROOT / "docs/technical-reviews/artifacts/revision-20261016/C.7-execution.json"
target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
print(out["result"])
print(json.dumps({"numeric_cases":cases,"historical_counts":checked,"code_hashes":source_hashes},ensure_ascii=False,indent=2))
