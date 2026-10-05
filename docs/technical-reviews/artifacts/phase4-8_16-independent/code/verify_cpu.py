"""Bounded review checks; constructed examples are not model measurements."""
import ast
import collections
import json
import io
import math
from contextlib import redirect_stdout
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO))
from tiny_perceptron.data import ByteTokenizer, IGNORE, pad_batch, render_chat

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)

def selected_classes(path, names, namespace):
    """Execute unchanged upstream class nodes, without unrelated package imports."""
    tree = ast.parse(path.read_bytes())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace

def no_decorator(*args, **kwargs):
    return lambda function: function

def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate field")
        obj[key] = value
    return obj

def local_format(text):
    try:
        obj = json.loads(text, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return False
    return isinstance(obj, dict) and set(obj) == {"text"} and isinstance(obj["text"], list) and all(isinstance(x, str) for x in obj["text"])

original = (ROOT / "code/original-fence-1.json").read_bytes().decode("utf-8")
chinese = ["今日休館", "明日開放"]
english = ["Closed today", "Open tomorrow"]
assert json.loads(original) == {"text": chinese}
assert local_format(original)

ifeval = selected_classes(ROOT / "sources/ifeval-instructions.py", {"Instruction", "JsonFormat"}, {"json": json})
official_json = ifeval["JsonFormat"]("detectable_format:json_format")
official_json.build_description()
format_cases = {
    "original": (original, True, True),
    "four_entries": (json.dumps({"text": [chinese[0], english[0], chinese[1], english[1]]}, ensure_ascii=False), True, True),
    "plain_two_lines": ("\n".join(chinese), False, False),
    "markdown_wrapper": ("```json\n" + original + "```", False, True),
    "extra_key": (json.dumps({"text": chinese, "note": "中文"}, ensure_ascii=False), False, True),
    "numeric_entry": ('{"text":["今日休館",3]}', False, True),
    "scalar_json": ('"今日休館"', False, True),
    "empty_array": ('{"text":[]}', True, True),
    "duplicate_key": ('{"text":[],"text":["今日休館","明日開放"]}', False, True),
    "outside_prefix": ("以下是答案：" + original, False, False),
    "outside_suffix": (original + "已完成", False, False),
}
formats = {}
for name, (text, want_local, want_official) in format_cases.items():
    got = (local_format(text), official_json.check_following(text))
    assert got == (want_local, want_official), (name, got)
    formats[name] = {"local_schema": got[0], "ifeval_json_rule": got[1]}

# Independent semantic annotations for these five hand-written examples only.
# Row identity denotes selected source rows even when a row has a transcription typo.
manual_cases = [
    ("correct", {1: chinese[0], 3: chinese[1]}, (1, 3), original, True),
    ("typo", {1: "今日開館", 3: chinese[1]}, (1, 3), '{"text":["今日開館","明日開放"]}', True),
    ("all_four", {1: chinese[0], 2: english[0], 3: chinese[1], 4: english[1]}, (1, 2, 3, 4), format_cases["four_entries"][0], True),
    ("plain", {1: chinese[0], 3: chinese[1]}, (1, 3), "\n".join(chinese), True),
    ("limit_no_eos", {1: chinese[0], 3: chinese[1]}, (1, 3), original, False),
]
rows = []
for name, mapping, row_ids, text, eos in manual_cases:
    content = [mapping.get(1), mapping.get(3)] == chinese and [r for r in row_ids if r in (1, 3)] == [1, 3]
    scope = row_ids == (1, 3)
    checks = (content, scope, local_format(text), eos)
    rows.append({"name": name, "checks": checks, "joint": all(checks)})
assert [r["checks"] for r in rows] == [(True, True, True, True), (False, True, True, True), (True, False, True, True), (True, True, False, True), (True, True, True, False)]
denominator = len(rows)
counts = [sum(r["checks"][i] for r in rows) for i in range(4)]
joint = sum(r["joint"] for r in rows)
assert denominator == 5 and counts == [4, 4, 4, 4] and joint == 1
assert sum(sum(r["checks"]) for r in rows) / 20 == 0.8
assert joint / denominator == 0.2
assert joint / sum(r["checks"][3] for r in rows) == 0.25

# Run unchanged official aggregation and strict/loose wrappers on synthetic input.
evaluation_namespace = {"collections": collections, "OutputExample": types.SimpleNamespace, "instructions_registry": types.SimpleNamespace(INSTRUCTION_DICT={"detectable_format:json_format": ifeval["JsonFormat"]})}
evaluation_tree = ast.parse((ROOT / "sources/ifeval-evaluation_lib.py").read_bytes())
evaluation_names = {"print_report", "test_instruction_following_strict", "test_instruction_following_loose"}
evaluation_nodes = [node for node in evaluation_tree.body if isinstance(node, ast.FunctionDef) and node.name in evaluation_names]
assert len(evaluation_nodes) == 3
exec(compile(ast.Module(body=evaluation_nodes, type_ignores=[]), str(ROOT / "sources/ifeval-evaluation_lib.py"), "exec"), evaluation_namespace)
official_aggregate_stdout = io.StringIO()
with redirect_stdout(official_aggregate_stdout):
    evaluation_namespace["print_report"]([types.SimpleNamespace(follow_instruction_list=r["checks"], instruction_id_list=["review:content", "review:scope", "review:format", "review:eos"]) for r in rows])
assert "prompt-level: 0.2" in official_aggregate_stdout.getvalue()
assert "instruction-level: 0.8" in official_aggregate_stdout.getvalue()
inp = types.SimpleNamespace(prompt="review synthetic prompt", instruction_id_list=["detectable_format:json_format"], kwargs=[{}])
introduced_response = "以下是答案：\n" + original
strict_introduced = evaluation_namespace["test_instruction_following_strict"](inp, {inp.prompt: introduced_response}).follow_all_instructions
loose_introduced = evaluation_namespace["test_instruction_following_loose"](inp, {inp.prompt: introduced_response}).follow_all_instructions
assert (strict_introduced, loose_introduced) == (False, True)

tok = ByteTokenizer()
prompt = '只抄兩行中文，依原順序，回只有text欄位的JSON。' + "\n".join([chinese[0], english[0], chinese[1], english[1]])
x, y = render_chat([{"role": "user", "content": prompt}, {"role": "assistant", "content": original.strip()}], tok)
labels = y[y != IGNORE].tolist()
assert labels == tok.encode(original.strip()) + [tok.eos_id]
first = (y != IGNORE).nonzero()[0].item()
assert x[first].item() == tok.assistant_id and y[first].item() == tok.encode(original)[0]
assert y[-1].item() == tok.eos_id
_, cropped_y, _ = pad_batch([(x, y)], max_length=len(x) - 1)
assert (cropped_y != IGNORE).any() and cropped_y[0, -1].item() != tok.eos_id

# Execute upstream stopping/forcing class bodies at the fixed reference version.
namespace = {"torch": torch, "math": math, "Union": __import__("typing").Union, "Optional": __import__("typing").Optional, "add_start_docstrings": no_decorator, "STOPPING_CRITERIA_INPUTS_DOCSTRING": "", "LOGITS_PROCESSOR_INPUTS_DOCSTRING": "", "StoppingCriteria": object, "LogitsProcessor": object, "isin_mps_friendly": torch.isin}
selected_classes(ROOT / "sources/hf-stopping_criteria.py", {"MaxLengthCriteria", "EosTokenCriteria"}, namespace)
selected_classes(ROOT / "sources/hf-logits_process.py", {"ForcedEOSTokenLogitsProcessor"}, namespace)
EOS = namespace["EosTokenCriteria"](2)
LENGTH = namespace["MaxLengthCriteria"](max_length=4)
FORCED = namespace["ForcedEOSTokenLogitsProcessor"](max_length=4, eos_token_id=2)
native_scores = torch.tensor([[0., 7., -8., 0.]])
assert native_scores.argmax(-1).item() == 1
assert FORCED(torch.tensor([[1, 3]]), native_scores).argmax(-1).item() == 1
forced_scores = FORCED(torch.tensor([[1, 3, 1]]), native_scores)
assert forced_scores.argmax(-1).item() == 2
stop_checks = {
    "native_eos": (EOS(torch.tensor([[1, 3, 2]]), native_scores).item(), LENGTH(torch.tensor([[1, 3, 2]]), native_scores).item()),
    "length_without_eos": (EOS(torch.tensor([[1, 3, 1, 1]]), native_scores).item(), LENGTH(torch.tensor([[1, 3, 1, 1]]), native_scores).item()),
    "forced_eos_at_limit": (EOS(torch.tensor([[1, 3, 1, 2]]), forced_scores).item(), LENGTH(torch.tensor([[1, 3, 1, 2]]), forced_scores).item()),
}
assert stop_checks == {"native_eos": (True, False), "length_without_eos": (False, True), "forced_eos_at_limit": (True, True)}
assert tok.decode(tok.encode(original.strip())) == tok.decode(tok.encode(original.strip()) + [2])

# Revised English request: unchanged format, different expected rows and content.
old_under_english = (chinese == english, (1, 3) == (2, 4), local_format(original), True)
english_target = json.dumps({"text": english}, ensure_ascii=False)
assert old_under_english == (False, False, True, True)
assert local_format(english_target)

result = {"scope": "Constructed review cases, not model inference/training/accuracy measurements; semantic rows and stop conditions are explicitly assigned for the handwritten table.", "original_json": json.loads(original), "format_variants": formats, "manual_rows": rows, "table_arithmetic": {"fixed_denominator": denominator, "per_requirement_pass_counts": counts, "joint_pass_count": joint, "average_of_requirement_rates": 0.8, "joint_fraction": 0.2, "incorrect_eos_filtered_joint_fraction": 0.25}, "official_aggregation_stdout": official_aggregate_stdout.getvalue(), "official_intro_json_strict_loose": [strict_introduced, loose_introduced], "render_chat": {"first_answer_prediction_position": first, "first_input_id": x[first].item(), "last_effective_target": labels[-1], "effective_target_count": len(labels), "answer_utf8_bytes": len(original.strip().encode()), "truncation_retains_answer_but_loses_eos": True}, "fixed_upstream_stopping_classes": stop_checks, "forced_score_native_argmax": 1, "forced_score_processed_argmax": 2, "old_chinese_target_under_english_request": old_under_english, "english_target": english_target, "official_execution_scope": "AST extracted exact classes/functions; unchanged methods executed, unrelated imports and decorators supplied by bounded local namespace; full Transformers/IFEval packages not installed or run."}
(ROOT / "execution/cpu-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
