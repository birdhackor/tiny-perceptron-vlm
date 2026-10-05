"""Personal bounded CPU checks for course/chapters/13.md#13.4.

No checkpoint loading, model training, data fetching, or GPU operations.
JSON access is restricted to the named measurement/provenance pointers below.
"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.model import ModelConfig, TinyLM

A = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.manual_seed(134)
environment = {
    "python": sys.version,
    "executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_git_revision": str(torch.version.git_version),
    "device": "cpu",
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "torch_threads": str(torch.get_num_threads()),
    "seed": "134",
    "cwd": str(Path.cwd()),
}
(A / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
raw = (A / "inputs/fence-1.python").read_bytes()
namespace = {"__name__": "__main__"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(raw, "13.4:original-fence-1", "exec"), namespace)
original_stdout = buf.getvalue()
(A / "original-fence.stdout.txt").write_text(original_stdout)
assert original_stdout == "參考未跟著改 True\n參考不記梯度 False\n"
policy, reference = namespace["policy"], namespace["reference"]
assert policy is not reference
assert policy.embedding.weight.data_ptr() != reference.embedding.weight.data_ptr()
assert all(not p.requires_grad for p in reference.parameters())
assert all(p.requires_grad for p in policy.parameters())
assert all(not m.training for m in reference.modules())
assert all(p.grad is None for p in policy.parameters())
assert all(p.grad is None for p in reference.parameters())
print("ORIGINAL_FENCE", repr(original_stdout))
print("ORIGINAL_FENCE assertions: independent storage; frozen reference; unfrozen policy; eval flags; no backward/update")

alias_namespace = {"__name__": "__main__"}
alias_code = raw.replace(b"copy.deepcopy(policy)", b"policy")
(A / "alias-fence.py").write_bytes(alias_code)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(alias_code, "13.4:alias-exercise", "exec"), alias_namespace)
alias_stdout = buf.getvalue()
(A / "alias-fence.stdout.txt").write_text(alias_stdout)
assert alias_stdout == "參考未跟著改 False\n參考不記梯度 False\n"
assert alias_namespace["policy"] is alias_namespace["reference"]
assert all(not p.requires_grad for p in alias_namespace["policy"].parameters())
print("ALIAS_EXERCISE", repr(alias_stdout))
print("ALIAS assertions: same object; policy also frozen; manual no_grad edit changes both names")

unfrozen = TinyLM(ModelConfig(width=8)).eval()
assert not unfrozen.training
assert unfrozen(torch.tensor([[1, 2]]))["logits"].requires_grad
print("EVAL_ONLY: evaluation mode by itself retains autograd=True for trainable parameters")

reference_gap = -4 - (-3)
policy_gap = -3 - (-3)
relative_gap = policy_gap - reference_gap
assert (reference_gap, policy_gap, relative_gap) == (-1, 0, 1)
assert math.exp(-3) == math.exp(-3)
print("MANUAL_GAPS", reference_gap, policy_gap, relative_gap, "policy candidate probabilities tie")
loss = dpo_loss(*[torch.tensor([x], dtype=torch.float64) for x in (-3., -3., -4., -3.)], beta=0.1)
expected_loss = math.log1p(math.exp(-0.1 * relative_gap))
assert math.isclose(loss.item(), expected_loss, abs_tol=1e-14)
initial_loss = dpo_loss(*[torch.tensor([x], dtype=torch.float64) for x in (-4., -3., -4., -3.)], beta=0.1)
assert math.isclose(initial_loss.item(), math.log(2), abs_tol=1e-14)
print("DPO_MARGIN_FORMULA", {"beta": .1, "relative_gap": relative_gap, "loss": loss.item(), "initial_relative_gap": 0, "initial_loss": initial_loss.item()})

def pure_functions(filename, names):
    tree = ast.parse((A / "original" / filename).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert set(names) == {node.name for node in nodes}
    namespace = {"json": json, "hashlib": hashlib, "random": random}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename + ":selected-pure-functions", "exec"), namespace)
    return namespace

revision = "8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d"
text_funcs = pure_functions("git-" + revision + "-text.py", ["_json_bytes", "_digest", "arithmetic_records"])
common_funcs = pure_functions("git-" + revision + "-common.py", ["split_records"])
behavior_funcs = pure_functions("git-" + revision + "-behavior.py", ["_preference_parts"])
arithmetic = common_funcs["split_records"](text_funcs["arithmetic_records"](), seed=42)
pairs = behavior_funcs["_preference_parts"](arithmetic)
dpo = json.loads((A / "original/docs/course-experiments/results/dpo.json").read_text())
style = json.loads((A / "original/docs/course-experiments/results/style.json").read_text())
assert dpo["revision"] == revision
for filename in ["behavior.py", "common.py", "text.py"]:
    path = "scripts/course_experiments/" + filename
    assert hashlib.sha256((A / "original" / ("git-" + revision + "-" + filename)).read_bytes()).hexdigest() == dpo["code_sha256"][path]
print("IMMUTABLE_SOURCE_HASHES match result /code_sha256 for behavior.py, common.py, text.py")
measurements = {"pointers": [], "splits": {}}
def saved_split_digest(rows):
    # Original _save_splits, lines 53 and 57: hash JSONL file bytes,
    # not the separate compact/sorted _digest convention used elsewhere.
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

for split in ["train", "validation", "test"]:
    recorded = dpo["results"]["data"][split]
    rows = pairs[split]
    assert len(rows) == recorded["records"]
    assert len({row["family"] for row in rows}) == recorded["families"]
    assert saved_split_digest(rows) == recorded["sha256"]
    style_data = style["results"]["arithmetic_data"][split]
    assert style_data["records"] == len(arithmetic[split])
    assert saved_split_digest(arithmetic[split]) == style_data["sha256"]
    measurements["splits"][split] = {k: recorded[k] for k in ["records", "families", "sha256"]}
    measurements["pointers"].extend(["dpo:/results/data/" + split, "style:/results/arithmetic_data/" + split])
assert not ({r["family"] for r in pairs["train"]} & {r["family"] for r in pairs["validation"]})
print("RECONSTRUCTED_SPLITS", json.dumps(measurements["splits"], sort_keys=True))

content = style["results"]["content_evaluation"]["test"]
samples = content["samples"]
assert len(samples) == content["records"] == content["examples"] == 7
assert sum(row["exact"] for row in samples) == content["matches"] == 0
for row in samples:
    question = row["messages"][0]["content"].removesuffix("=?")
    assert row["expected"] == str(sum(int(x) for x in question.split("+")))
    ids = row["generated_ids"]
    answer_ids = ids[:ids.index(2)] if 2 in ids else ids
    expected_ids = [byte + 8 for byte in row["expected"].encode("utf-8")]
    assert row["exact"] == (answer_ids == expected_ids)
    assert row["generated"] != row["expected"]
assert content["exact_match"] == content["matches"] / content["records"] == 0
assert style["results"]["content_checkpoint"] == style["results"]["content_training"]["checkpoint"] == "content.pt"
measurements["content_test"] = {"records": 7, "matches": 0, "exact_match": 0, "samples": [{k: row[k] for k in ["messages", "expected", "generated", "exact"]} for row in samples]}
measurements["pointers"].extend(["style:/results/content_evaluation/test/records", "style:/results/content_evaluation/test/examples", "style:/results/content_evaluation/test/matches", "style:/results/content_evaluation/test/exact_match", "style:/results/content_evaluation/test/samples", "style:/results/content_checkpoint", "style:/results/content_training/checkpoint"])
print("CONTENT_BASELINE_TEST: raw exact strings and flags agree; 0/7 correct")

results = dpo["results"]
before_rows = results["before"]["validation"]["samples"]
after_rows = results["runs"]["model"]["preference"]["validation"]["samples"]
assert len(before_rows) == len(after_rows) == len(pairs["validation"]) == 8
for row in before_rows:
    assert row["policy_margin"] == row["policy_chosen_logp"] - row["policy_rejected_logp"]
    assert row["relative_margin"] == row["policy_margin"] - row["reference_margin"] == 0
before = before_rows[2]
after = after_rows[2]
assert before["prompt"] == after["prompt"] == "3+5=?"
assert before["chosen"] == after["chosen"] == "8"
assert before["rejected"] == after["rejected"] == "9"
assert after["reference_margin"] == before["reference_margin"]
assert after["policy_margin"] == after["policy_chosen_logp"] - after["policy_rejected_logp"]
assert after["relative_margin"] == after["policy_margin"] - after["reference_margin"]
assert all(round(x, 5) == y for x, y in [(before["reference_margin"], -10.89346), (after["policy_margin"], -8.44789), (after["relative_margin"], 2.44557)])
assert after["relative_margin"] > 0 and after["policy_margin"] < 0
assert after["policy_rejected_logp"] > after["policy_chosen_logp"]
assert after["chosen_answer_tokens"] == after["rejected_answer_tokens"] == 2
measurements["validation_3_plus_5"] = {"before_sample_index": 2, "after_sample_index": 2, "before": before, "after": after, "rejected_to_chosen_probability_ratio": math.exp(-after["policy_margin"])}
measurements["pointers"].extend(["dpo:/results/before/validation/samples/2", "dpo:/results/runs/model/preference/validation/samples/2"])
for name, beta in [("model", .1), ("beta1", 1.)]:
    assert results["runs"][name]["beta"] == beta
    assert results["runs"][name]["training"]["steps"] == 250
    measurements["pointers"].extend(["dpo:/results/runs/" + name + "/beta", "dpo:/results/runs/" + name + "/training/steps"])
assert results["reference_unchanged"] is True
assert len(results["reference_sha256"]) == 64
measurements["reference_recorded_state_digest"] = results["reference_sha256"]
measurements["reference_recorded_unchanged"] = results["reference_unchanged"]
measurements["pointers"].extend(["dpo:/results/reference_sha256", "dpo:/results/reference_unchanged"])
print("RAW_3_PLUS_5_MARGINS", json.dumps(measurements["validation_3_plus_5"], sort_keys=True))
print("RECORDED_REFERENCE_DIGEST", results["reference_sha256"], "unchanged=True; original run_dpo verifies before/after state hashes; no weights loaded")
print("BETA_RUNS: .1 and 1.0; each 250 updates recorded; source copies the same content.pt base")
(A / "measurements.json").write_text(json.dumps(measurements, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print("PASS: all bounded check assertions completed")
