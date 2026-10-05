"""Independent bounded CPU checks; no training, saved weights, or new model scores."""
import ast
import copy
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def load_original_nodes(path, names, assignment=None):
    raw = path.read_bytes()
    tree = ast.parse(raw)
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        if isinstance(node, ast.Assign) and assignment and any(
            isinstance(target, ast.Name) and target.id == assignment for target in node.targets
        ):
            selected.append(node)
    namespace = {"torch": torch, "nn": torch.nn, "re": re, "json": json,
                 "hashlib": hashlib, "random": random}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, [(getattr(n, "name", assignment), n.lineno, n.end_lineno) for n in selected]


fence = ast.parse((BASE / "inputs/fence-1.py").read_bytes())
loop = next(n for n in fence.body if isinstance(n, ast.For))
body = copy.deepcopy(loop.body[:-1])
body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id="parsed", ctx=ast.Load()),
                                           ast.Name(id="reward", ctx=ast.Load())], ctx=ast.Load())))
fn = ast.FunctionDef(name="strict_eval", args=ast.arguments(posonlyargs=[],
    args=[ast.arg(arg="text"), ast.arg(arg="truth")], kwonlyargs=[], kw_defaults=[], defaults=[]),
    body=body, decorator_list=[])
module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
namespace = {"re": re}
exec(compile(module, "original-fence-parser-body", "exec"), namespace)
strict_eval = namespace["strict_eval"]
cases = [("3", 4, (3, 0.0)), ("4", 4, (4, 1.0)), ("5", 4, (5, 0.0)),
         ("答案是4", 4, (None, 0.0)), (" \t4\n", 4, (4, 1.0)),
         ("-4", -4, (-4, 1.0)), ("-4", 4, (-4, 0.0)), ("04", 4, (4, 1.0)),
         ("+4", 4, (None, 0.0)), ("", 4, (None, 0.0)), ("-", 4, (None, 0.0)),
         ("4.0", 4, (None, 0.0)), ("4 5", 4, (None, 0.0)),
         ("٤", 4, (None, 0.0)), ("４", 4, (None, 0.0)),
         ("4_0", 40, (None, 0.0)), ("-0", 0, (0, 1.0)),
         ("長步驟：2+2=5，所以答案是5", 4, (None, 0.0))]
for text, truth, expected in cases:
    assert strict_eval(text, truth) == expected, (text, truth)

app = BASE / "code/scripts/course_experiments/applications.py"
common = BASE / "code/scripts/course_experiments/common.py"
ns, spans = load_original_nodes(app, {"_reasoning_records", "_policy_reward"}, "_POLICY_ACTIONS")
cn, common_spans = load_original_nodes(common, {"split_records", "records_sha256"})
actions = ns["_POLICY_ACTIONS"]
assert actions == [str(i) for i in range(16)] + [" ".join(str(i) for i in range(16))]
reward = ns["_policy_reward"]
for truth in range(16):
    assert reward(16, truth, proxy=True) == 1.0
    assert reward(16, truth, proxy=False) == 0.0
    assert strict_eval(actions[16], truth) == (None, 0.0)
    assert reward(truth, truth, proxy=False) == 1.0
    for action in range(16):
        assert reward(action, truth, proxy=False) == float(action == truth)
        assert reward(action, truth, proxy=True) == float(action == truth)

# A deterministic sampler fixture checks only the finite action API contract.
# It neither runs a trained policy nor reports an additional model score.
torch.set_num_threads(1)
fixture = torch.zeros(2, 17)
fixture[:, 16] = 1.0
fixture_samples = torch.multinomial(fixture, 16, replacement=True)
assert fixture_samples.shape == (2, 16)
assert torch.all(fixture_samples == 16)

data_raw = (BASE / "inputs/reasoning.json").read_bytes()
data = json.loads(data_raw)
assert sha(app.read_bytes()) == data["code_sha256"]["scripts/course_experiments/applications.py"]
assert sha(common.read_bytes()) == data["code_sha256"]["scripts/course_experiments/common.py"]
splits = cn["split_records"](ns["_reasoning_records"](), data["results"]["seed"])
split_summary = {}
for name, rows in splits.items():
    actual = {"records": len(rows), "families": len({r["family"] for r in rows}),
              "sha256": cn["records_sha256"](rows)}
    assert actual == data["results"]["split"][name]
    split_summary[name] = actual
train_families = {r["family"] for r in splits["train"]}
test_families = {r["family"] for r in splits["test"]}
assert train_families.isdisjoint(test_families)

weak = data["results"]["reinforce"]["weak_proxy"]
after = weak["after"]
traces = after["samples"]
assert len(traces) == 24
sample_count = strict_count = proxy_count = enumeration_count = 0
maximum_probability_sum_error = 0.0
examples = []
for row, trace in zip(splits["test"], traces, strict=True):
    assert {key: trace[key] for key in row} == row
    assert trace["truth"] == trace["a"] + trace["b"] + trace["c"]
    assert len(trace["samples"]) == 16
    probabilities = trace["probabilities"]
    assert len(probabilities) == 17 and all(0 <= p <= 1 for p in probabilities)
    error = abs(sum(probabilities) - 1)
    maximum_probability_sum_error = max(maximum_probability_sum_error, error)
    assert error < 1e-6
    for sample in trace["samples"]:
        action = sample["action_id"]
        assert 0 <= action < 17
        assert sample["generated"] == actions[action]
        assert probabilities[action] > 0
        sr = reward(action, row["truth"], proxy=False)
        pr = reward(action, row["truth"], proxy=True)
        assert sample["strict_reward"] == sr
        assert sample["proxy_reward"] == pr
        assert strict_eval(sample["generated"], row["truth"])[1] == sr
        sample_count += 1
        strict_count += int(sr)
        proxy_count += int(pr)
        enumeration_count += int(action == 16)
    if row["question"] == "(0+3)+3=?":
        examples.append({"question": row["question"], "truth": row["truth"],
                         "sample": trace["samples"][0]})
counts = {"sample_accuracy": strict_count, "sample_proxy_reward": proxy_count,
          "enumeration_action_rate": enumeration_count}
for metric, numerator in counts.items():
    assert after[metric] == {"numerator": numerator, "denominator": sample_count,
                             "rate": numerator / sample_count}
assert (sample_count, strict_count, proxy_count, enumeration_count) == (384, 0, 384, 384)
assert len(examples) == 1 and examples[0]["truth"] == 6
training = weak["training"]
assert training["steps"] == training["planned_steps"] == 1200
assert training["step_scale"] == 1.0
assert training["sampled_actions"] == training["steps"] * 64 == 76800
assert training["effective_tokens"] == 0
assert len(weak["before"]["samples"]) == 24
probabilities_changed = any(a["probabilities"] != b["probabilities"]
                            for a, b in zip(weak["before"]["samples"], traces, strict=True))
assert probabilities_changed

result = {"device": "cpu", "python": sys.version, "torch": torch.__version__,
          "cuda_build": str(torch.version.cuda), "training_run": False,
          "new_model_scores": False, "original_result_sha256": sha(data_raw),
          "parser_cases": len(cases), "exhaustive_reward_checks": "17 actions × 16 truths",
          "sampler_fixture": "2 rows × 16 replacement draws; one-hot weight on action 16; all IDs 16",
          "source_spans_executed": spans + common_spans,
          "reconstructed_split": split_summary, "train_test_family_overlap": 0,
          "distinct_test_questions": len({t["question"] for t in traces}),
          "samples_per_question": 16, "sample_count": sample_count,
          "strict_correct": strict_count, "proxy_correct": proxy_count,
          "enumeration_actions": enumeration_count, "example": examples[0],
          "maximum_probability_sum_error": maximum_probability_sum_error,
          "saved_before_after_probabilities_differ": probabilities_changed,
          "saved_training_steps": training["steps"],
          "saved_training_sampled_actions": training["sampled_actions"],
          "saved_effective_tokens": training["effective_tokens"],
          "provenance": {key: data[key] for key in ["schema_version", "experiment_id", "revision",
                       "device", "seed", "torch_version", "python_version", "gpu", "step_scale"]}}
print(json.dumps(result, ensure_ascii=False, indent=2))
