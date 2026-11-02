"""Reviewer-owned CPU checks for canonical 16.15; no model weights or training."""

import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import platform
import re
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v7.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(code):
    namespace = {"__name__": "__review_inline__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "<frozen-16.15>", "exec"), namespace)
    return namespace, output.getvalue()


def softmax(values):
    exp = [math.exp(v - max(values)) for v in values]
    return [v / sum(exp) for v in exp]


def scalar_encode(rows):
    result = []
    for row in rows:
        weights = softmax([sum(a * b for a, b in zip(row, key)) / math.sqrt(2) for key in rows])
        result.append([row[j] + sum(w * key[j] for w, key in zip(weights, rows)) for j in range(2)])
    return result


def scalar_answer(memory, question):
    candidates = [[1.0, 0.0], [0.0, 1.0]]
    weights, hidden = [], []
    for candidate in candidates:
        query = [a + b for a, b in zip(question, candidate)]
        w = softmax([sum(a * b for a, b in zip(query, key)) / math.sqrt(2) for key in memory])
        weights.append(w)
        hidden.append([candidate[j] + sum(p * key[j] for p, key in zip(w, memory)) for j in range(2)])
    scores = [sum(a * b for a, b in zip(row, question)) / math.sqrt(2) for row in hidden]
    scores.append(1.0 - max(scores))
    return weights, hidden, scores, softmax(scores)


def max_diff(a, b):
    return float((torch.as_tensor(a, dtype=torch.float64) - torch.as_tensor(b, dtype=torch.float64)).abs().max())


freeze = json.loads(FREEZE.read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "16.15")
snapshot = ROOT / page["snapshot"]
body = snapshot.read_text()
chapter = (ROOT / page["source"]).read_text()
current = re.search(r"^## 16\.15 .*?(?=^## |\Z)", chapter, re.M | re.S)[0]
assert current == body
assert digest(snapshot) == page["source_sha256"]
inline = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(inline) == 1
code = inline[0]
nbpath = ROOT / page["notebook"]
assert digest(nbpath) == page["notebook_sha256"]
notebook = json.loads(nbpath.read_text())
code_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
assert len(code_cells) == 3 and code_cells[-1].rstrip() == code.rstrip()

baseline, output = run(code)
expected_output = """材料編碼次數：重算／共享 3 1
M、權重、特徵、分數、機率的最大差 0.0
Q1/Q2/Q1機率 [[0.9216, 0.0769, 0.0015], [0.1746, 0.8254, 0.0], [0.9216, 0.0769, 0.0015]]
錯誤寫回後Q1機率差 0.1079
雙向混合示意的材料特徵差 0.9325
各分支從prefix起步 [4.5, 5.5, 4.5]
誤用前題結尾接下一題 [4.5, 6.55, 6.675]
"""
assert output == expected_output
memory = baseline["memory"]
assert tuple(memory.shape) == (2, 2)
assert memory.dtype == torch.float32 and memory.device.type == "cpu"
assert torch.equal(memory, baseline["saved"])
assert torch.equal(baseline["shared"][0][-1], baseline["shared"][2][-1])
assert 0 < float(baseline["shared"][1][-1][-1]) < 0.00005
scalar_memory = scalar_encode([[1.0, 0.0], [0.0, 2.0]])
scalar_results = [scalar_answer(scalar_memory, q) for q in [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]]
scalar_errors = [max_diff(memory.tolist(), scalar_memory)]
for tensor_result, scalar_result in zip(baseline["shared"], scalar_results):
    scalar_errors += [max_diff(t.tolist(), s) for t, s in zip(tensor_result[1:], scalar_result)]
assert max(scalar_errors) < 2e-6

# Independent API check: the explicitly coded cross-attention agrees with SDPA.
sdpa_errors = []
for q in baseline["questions"]:
    query = q[None, :] + baseline["candidates"]
    attn = F.scaled_dot_product_attention(query, memory, memory, dropout_p=0.0)
    expected = baseline["answer"](memory, q)[1] - baseline["candidates"]
    sdpa_errors.append(float((attn - expected).abs().max()))
assert max(sdpa_errors) < 1e-6
clone = memory.clone()
clone[0, 0] += 1
assert not torch.equal(clone, memory) and torch.equal(memory, baseline["saved"])
dropout = torch.nn.Dropout(0.75).eval()
assert torch.equal(dropout(memory), memory)

# Each exercise is restarted from the unmodified inline program.
no_write_code = code.replace("    m[0].add_(q)  # 錯誤的寫回；正確answer沒有這一行\n", "")
assert no_write_code != code
no_write, no_write_output = run(no_write_code)
no_write_delta = float((no_write["first"][-1] - no_write["last"][-1]).abs().max())
assert no_write_delta == 0.0
no_copy_code = code.replace(
    'clean = [advance(copy.deepcopy(prefix), [x]) for x in [3.0, 4.0, 3.0]]',
    'clean = [advance(prefix, [x]) for x in [3.0, 4.0, 3.0]]',
)
assert no_copy_code != code
no_copy, no_copy_output = run(no_copy_code)
assert no_copy["clean"] == [4.5, 6.55, 6.675]
assert no_copy["prefix"]["r"] == 6.675
assert baseline["prefix"]["r"] == 2.6
assert baseline["prefix"]["conv"] == [2.0]
rollback = copy.deepcopy(baseline["prefix"])
baseline["advance"](rollback, [3.0])
del rollback["kv"][2:]
assert rollback["r"] == 4.5 and rollback["conv"] == [3.0]
assert rollback["kv"] == baseline["prefix"]["kv"]

# Run the exact local Notebook setup, SVG display and lesson cells. No Colab branch runs.
notebook_namespace = {"__name__": "__review_notebook__"}
notebook_stdout = io.StringIO()
with contextlib.redirect_stdout(notebook_stdout):
    for i, cell in enumerate(code_cells):
        exec(compile(cell, f"<notebook-code-{i}>", "exec"), notebook_namespace)
assert notebook_namespace["in_colab"] is False
assert notebook_namespace["root"] == ROOT
assert notebook_namespace["error"] == 0.0
assert notebook_stdout.getvalue().endswith(expected_output)

# Inspect the actual project implementation without running any formal model forward.
modelpath = ROOT / "tiny_perceptron/lesson_products/model.py"
tree = ast.parse(modelpath.read_text())
classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
scorer = ast.get_source_segment(modelpath.read_text(), classes["CandidateScorer"])
block = ast.get_source_segment(modelpath.read_text(), classes["Block"])
tokenizer = ast.get_source_segment(modelpath.read_text(), classes["Tokenizer"])
assert 'parts = [state] if question is None else [state, QUESTIONS[question], MEANINGS[candidate]]' in tokenizer
assert 'block(hidden, tokens != 0)' in scorer
assert 'attn_mask=valid[:, None, None, :], dropout_p=0.0' in block
assert 'is_causal' not in block

record = {
    "reviewer_task": "/root/omni_p5_technical_16_15",
    "scope": "Frozen inline CPU demonstration, two specified exercises, exact local notebook code, static project model contract. No pretrained/formal neural weights or final-family inference.",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU", "threads": str(torch.get_num_threads())},
    "inputs": {
        "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE)},
        "snapshot": {"path": str(snapshot.relative_to(ROOT)), "sha256": digest(snapshot)},
        "notebook": {"path": str(nbpath.relative_to(ROOT)), "sha256": digest(nbpath)},
        "project_model": {"path": str(modelpath.relative_to(ROOT)), "sha256": digest(modelpath)},
        "inline_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    },
    "stdout": output,
    "baseline": {
        "memory": memory.tolist(),
        "branches": [{"weights": v[1].tolist(), "hidden": v[2].tolist(), "scores": v[3].tolist(), "probabilities": v[4].tolist()} for v in baseline["shared"]],
        "intermediate_max_difference": baseline["error"],
        "memory_unchanged": torch.equal(memory, baseline["saved"]),
        "dirty_probability_difference": float((baseline["first"][-1] - baseline["last"][-1]).abs().max()),
        "mixed_memory_difference": float((baseline["old1"] - baseline["old2"]).abs().max()),
        "clean_recurrence": baseline["clean"],
        "polluted_recurrence": baseline["polluted"],
        "prefix": baseline["prefix"],
    },
    "independent_reference": {"formula": "softmax(row dot key / sqrt(2)); residual sum; same explicit candidate and none rules; Python math float64 without torch matmul", "memory": scalar_memory, "branches": scalar_results, "max_absolute_difference_from_FP32": max(scalar_errors)},
    "api_checks": {"sdpa_absolute_differences": sdpa_errors, "clone_mutation_isolated": True, "eval_dropout_is_identity": True},
    "exercise_1": {"removed_only_writeback": True, "probability_difference": no_write_delta, "stdout": no_write_output},
    "exercise_2": {"removed_only_deepcopy_from_clean": True, "clean": no_copy["clean"], "prefix_after_clean": no_copy["prefix"], "polluted_after_mutated_prefix": no_copy["polluted"], "stdout": no_copy_output},
    "kv_crop_counterexample": {"restored_kv_only": rollback["kv"], "r_still_changed": rollback["r"], "conv_still_changed": rollback["conv"]},
    "notebook": {"executed_code_cell_count": len(code_cells), "local_setup_only": True, "stdout": notebook_stdout.getvalue()},
    "project_static_contract": {"separate_state_question_candidate_parts": True, "only_valid_key_mask": True, "no_causal_flag_in_Block": True, "no_project_model_forward_executed": True},
    "assertions_passed": True,
}
target = OUT / "inline-execution-v1.json"
with target.open("x") as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"result": "pass", "artifact": str(target.relative_to(ROOT)), "sha256": digest(target), "scalar_max_error": max(scalar_errors), "sdpa_max_error": max(sdpa_errors)}, ensure_ascii=False))
