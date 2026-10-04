"""Fresh B.6 review: short deterministic probes; no optimization or retraining."""

import copy
import hashlib
import json
import platform
import runpy
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch

from scripts.course_experiments import tool_choice
from scripts.course_experiments.common import Context, records_sha256
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.training import load_checkpoint

torch.set_num_threads(2)
torch.manual_seed(42)
tok = ByteTokenizer()
messages = [
    {"role": "system", "content": "精確計算用工具，解釋或照抄直接回答；缺資訊或工具先求助。計算器可用。"},
    {"role": "user", "content": "1加2等於多少？"},
    {"role": "assistant", "content": "TOOL"},
]
out = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                       "device": "cpu", "threads": str(torch.get_num_threads())}}

def inspect_chat(msgs):
    x, y = render_chat(msgs, tok)
    positions = (y != -100).nonzero().flatten().tolist()
    answer = y[y != -100].tolist()
    user_start = 1 + 1 + len(tok.encode(msgs[0]["content"])) + 1 + 1
    user_end = user_start + len(tok.encode(msgs[1]["content"]))
    return {"input_length": len(x), "label_length": len(y), "input_dtype": str(x.dtype),
            "label_dtype": str(y.dtype), "supervised_positions": positions,
            "supervised_ids": answer, "decoded": tok.decode(answer),
            "eos_last": answer[-1] == tok.eos_id,
            "first_supervised_input_is_assistant_marker": int(x[positions[0]]) == tok.assistant_id,
            "user_ids_retained_in_input": tok.encode(msgs[1]["content"]) == x[user_start:user_end].tolist()}

x, y = render_chat(messages, tok)
out["original"] = inspect_chat(messages)
assert out["original"]["supervised_ids"] == [92, 87, 87, 84, 2]
assert out["original"]["supervised_positions"] == [127, 128, 129, 130, 131]
assert out["original"]["decoded"] == "TOOL" and out["original"]["eos_last"]
edited = copy.deepcopy(messages)
edited[2]["content"] = "ASK"
out["exercise"] = inspect_chat(edited)
assert out["exercise"]["supervised_ids"] == [73, 91, 83, 2]
assert edited[:2] == messages[:2]
out["exercise"]["system_and_user_unchanged"] = True
out["exercise"]["policy_expected_action"] = tool_choice._expected_action("numerical_addition", True)
assert out["exercise"]["policy_expected_action"] == "TOOL"

model = TinyLM(ModelConfig(width=8, layers=1, heads=2, max_length=160))
logits = model(x.unsqueeze(0))["logits"]
logits.retain_grad()
loss = masked_loss(logits, y.unsqueeze(0))
loss.backward()
ignored = y == -100
out["mask"] = {"ignored_logit_gradient_max": float(logits.grad[0, ignored].abs().max()),
               "supervised_logit_gradient_max": float(logits.grad[0, ~ignored].abs().max()),
               "user_input_byte_length": len(tok.encode(messages[1]["content"]))}
assert out["mask"]["ignored_logit_gradient_max"] == 0
assert out["mask"]["supervised_logit_gradient_max"] > 0
user_start = 1 + 1 + len(tok.encode(messages[0]["content"])) + 1 + 1
user_end = user_start + len(tok.encode(messages[1]["content"]))
out["original"]["user_ids_retained_in_input"] = x[user_start:user_end].tolist() == tok.encode(messages[1]["content"])
assert out["original"]["user_ids_retained_in_input"]
out["original"]["user_input_positions"] = [user_start, user_end - 1]
altered_x = x.clone()
altered_x[user_start] = tok.encode("2")[0]
with torch.no_grad():
    changed = model(altered_x.unsqueeze(0))["logits"]
out["mask"]["changed_context_active_logits_max_difference"] = float((changed[0, ~ignored] - logits.detach()[0, ~ignored]).abs().max())
assert out["mask"]["changed_context_active_logits_max_difference"] > 0
try:
    render_chat([{"role": "user", "content": "1+2"}], tok)
except ValueError as error:
    out["no_assistant"] = str(error)
try:
    render_chat([{"role": "tool", "content": "3"}], tok)
except ValueError as error:
    out["tool_role"] = str(error)

splits = tool_choice.split_records(tool_choice.build_records(), 42)
dataset_path = ROOT / "outputs/tool-choice-development/fixed-900/dataset.json"
saved_dataset = json.loads(dataset_path.read_text())
test_families = {r["family"] for r in splits["test"]}
diagnostic = [r for r in tool_choice.build_records(paraphrase=True) if r["family"] in test_families]
rebuilt = {**splits, "paraphrase_diagnostic": diagnostic}
assert rebuilt == saved_dataset
out["dataset"] = {}
for name, rows in rebuilt.items():
    assert all(r["messages"][2]["content"] in tool_choice.ACTIONS for r in rows)
    assert all(not r["messages"][1]["content"].startswith(("CALC", "COPY")) for r in rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["family"], row["template"]), []).append(row)
    assert all({r["calculator_available"] for r in pair} == {True, False} for pair in grouped.values())
    out["dataset"][name] = {"records": len(rows), "families": len({r["family"] for r in rows}),
                            "action_counts": dict(Counter(r["expected_action"] for r in rows)),
                            "records_sha256": records_sha256(rows),
                            "availability_paired": True, "no_calc_copy_prefixes": True,
                            "action_only_answers": True}
assert all({r["family"] for r in splits[a]}.isdisjoint({r["family"] for r in splits[b]})
           for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
out["dataset"]["numeric_family_split_disjoint"] = True
assert {r["messages"][1]["content"] for r in diagnostic}.isdisjoint(
    {r["messages"][1]["content"] for r in tool_choice.build_records()})
out["dataset"]["diagnostic_question_wordings_unseen"] = True

report_path = ROOT / "docs/course-experiments/results/tool_choice.json"
report = json.loads(report_path.read_text())["results"]
raw_result_path = ROOT / "outputs/tool-choice-development/fixed-900/results.json"
assert json.loads(raw_result_path.read_text()) == report
checkpoint_path = ROOT / "outputs/tool-choice-development/fixed-900/tool-choice.pt"
checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
assert checkpoint_hash == report["frozen_evaluation"]["checkpoint_sha256"]
assert hashlib.sha256((ROOT / "scripts/course_experiments/tool_choice.py").read_bytes()).hexdigest() == report["reproducibility"]["script_sha256"]
trained, payload = load_checkpoint(checkpoint_path, "cpu")
assert payload["step"] == 900 and payload["metadata"]["mode"] == "sft"
assert payload["training_state"]["batch_size"] == 24
assert payload["metadata"]["records_sha256"] == records_sha256(splits["train"])
assert tool_choice._state_sha256(trained) == report["frozen_evaluation"]["weights_sha256"]
out["saved_run"] = {"checkpoint_path": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": checkpoint_hash,
                    "step": payload["step"], "metadata": payload["metadata"],
                    "model_config": payload["config"], "batch_size": payload["training_state"]["batch_size"],
                    "planned_steps": payload["training_state"]["planned_steps"],
                    "packaged_results_equal_original_results": True, "checkpoint_hash_matches": True,
                    "current_tool_choice_code_matches_saved_run": True}
before = tool_choice._state_sha256(trained)
ctx = Context("cpu", ROOT / "outputs/reviewer-tools/tool-choice-b6-replay", ROOT / "outputs", ROOT / "assets", 42)
out["replay"] = {}
for name in ("validation", "test", "paraphrase_diagnostic"):
    current = tool_choice.evaluate(trained, rebuilt[name], ctx)
    assert current["records_sha256"] == report[name]["records_sha256"]
    assert current["metrics"] == report[name]["metrics"]
    assert [r["generation"]["samples"][0]["generated_ids"] for r in current["samples"]] == [r["generation"]["samples"][0]["generated_ids"] for r in report[name]["samples"]]
    out["replay"][name] = {"samples": len(current["samples"]), "identical_generated_ids": True,
                            "metrics_match": True, "accuracy": current["metrics"]["accuracy"]}
assert tool_choice._state_sha256(trained) == before
out["replay"]["weights_unchanged"] = True

class TrainingBlockedForReview(RuntimeError):
    pass

def capture_fit(model, records, ctx, **kwargs):
    out["cli_probe"] = {"argv": sys.argv.copy(), "device": ctx.device, "seed": ctx.seed,
                        "records": len(records), "training_kwargs": kwargs,
                        "initial_model_config": model.description()["config"],
                        "initial_state_sha256": tool_choice._state_sha256(model),
                        "initialized_before_fit": True, "optimizer_updates": 0,
                        "scope": "Real documented CLI dispatch, with output override; fit_lm intercepted before optimization. Not a retraining."}
    assert kwargs == {"mode": "sft", "steps": 900, "batch_size": 24, "lr": 0.003, "name": "tool-choice"}
    assert tool_choice._state_sha256(model) != before
    raise TrainingBlockedForReview("Review deliberately intercepted fit_lm before any update")

original_fit = tool_choice.fit_lm
original_argv = sys.argv.copy()
tool_choice.fit_lm = capture_fit
try:
    sys.argv = ["scripts/course_experiments/run.py", "--experiment", "tool_choice", "--device", "cpu", "--output", "outputs/reviewer-tools/tool-choice-b6-cli"]
    runpy.run_path(str(ROOT / sys.argv[0]), run_name="__main__")
except TrainingBlockedForReview:
    out["cli_probe"]["interception_caught"] = True
finally:
    sys.argv = original_argv
    tool_choice.fit_lm = original_fit

out["execution_result"] = "All assertions passed; original example and ASK exercise checked, saved checkpoint replayed, CLI training prevented before updates."
print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
