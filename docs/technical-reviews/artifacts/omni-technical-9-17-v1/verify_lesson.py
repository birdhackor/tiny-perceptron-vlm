"""9.17 reviewer CPU receipt: execute frozen inline code and original Kev pure nodes.

No backbone, tokenizer download, training, GPU, or network is used by this script.
The small ASCII tokenizer below tests token layout only, not language capability.
"""

import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
CACHE = Path("/workspace/work/reference/omni-technical-9-17-v1")
OUT = Path(__file__).with_name("execution.json")
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix08.json"
freeze = json.loads(FREEZE.read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "9.17")
snapshot = ROOT / page["snapshot"]
raw = snapshot.read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
code_blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
assert len(code_blocks) == 1
code = code_blocks[0]
torch.set_num_threads(1)


def run_inline(program, filename):
    """Execute exact supplied program; trace its return locals without editing it."""
    records = []

    def trace(frame, event, arg):
        if event == "return" and frame.f_code.co_filename == filename and frame.f_code.co_name == "score":
            v = frame.f_locals
            records.append({
                "order": list(v["order"]),
                "question": v["h_question"].tolist(),
                "causal": v["causal"],
                "q": v["q"].tolist(),
                "keys": torch.stack(v["keys"]).tolist(),
                "real_scores": v["real_scores"].tolist(),
                "none_score": float(v["none_score"]),
                "probability_by_id": arg[0],
                "chosen": arg[1],
            })
        return trace

    ns = {}
    stdout = io.StringIO()
    old_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        with contextlib.redirect_stdout(stdout):
            exec(compile(program, filename, "exec"), ns)
    finally:
        sys.settrace(old_trace)
    return ns, stdout.getvalue().splitlines(), records


ns, stdout, rows = run_inline(code, "frozen-9.17-inline.py")
assert len(stdout) == len(rows) == 8
expected_choices = ["card", "card", "address", "address", "card", "card", "address", "address"]
assert [row["chosen"] for row in rows] == expected_choices
for order_pair in ((0, 1), (2, 3), (6, 7)):
    for cid in ("card", "address", "none"):
        assert math.isclose(rows[order_pair[0]]["probability_by_id"][cid], rows[order_pair[1]]["probability_by_id"][cid], abs_tol=1e-7)
for cid, expected in zip(("card", "address", "none"), (0.7124, 0.1732, 0.1145)):
    assert round(rows[0]["probability_by_id"][cid], 4) == expected
assert round(rows[4]["probability_by_id"]["address"], 4) == 0.2982
assert round(rows[5]["probability_by_id"]["address"], 4) == 0.1732
assert rows[0]["probability_by_id"]["card"] == rows[2]["probability_by_id"]["address"]

# A separate executed scalar reference; these evaluations are not called hand calculations.
a = math.sqrt(2)
scalar_logits = [a, 0.0, 1.0 - a]
scalar_exp = [math.exp(x) for x in scalar_logits]
scalar_prob = [x / sum(scalar_exp) for x in scalar_exp]
assert max(abs(rows[0]["probability_by_id"][cid] - p) for cid, p in zip(("card", "address", "none"), scalar_prob)) < 1e-7
shape = {
    "Wq": list(ns["Wq"].shape), "Wk": list(ns["Wk"].shape),
    "q": [len(rows[0]["q"])], "stacked_keys": [len(rows[0]["keys"]), len(rows[0]["keys"][0])],
    "real_scores": [len(rows[0]["real_scores"])],
    "none_scalar": list(torch.tensor(rows[0]["none_score"]).shape),
    "none_with_axis": list(torch.tensor(rows[0]["none_score"])[None].shape),
    "concatenated_scores": [len(rows[0]["probability_by_id"])],
    "q_numel": int((ns["questions"]["現在要辦"] @ ns["Wq"]).numel()),
}
assert shape["stacked_keys"] == [2, 2] and shape["concatenated_scores"] == [3] and shape["q_numel"] == 2
assert torch.equal(ns["Wq"], torch.tensor([[1., 0.], [0., 1.]]))
assert all(not t.requires_grad and t.device.type == "cpu" and t.dtype == torch.float32
           for t in [*ns["features"].values(), *ns["questions"].values(), ns["Wq"], ns["Wk"]])

old = '"card": torch.tensor([1., 0.])'
new = '"card": torch.tensor([0., 1.])'
assert code.count(old) == 1
exercise_code = code.replace(old, new)
exercise_ns, exercise_stdout, exercise_rows = run_inline(exercise_code, "frozen-9.17-exercise.py")
for row in exercise_rows:
    if not row["causal"]:
        assert row["real_scores"][0] == row["real_scores"][1]
        assert row["probability_by_id"]["card"] == row["probability_by_id"]["address"]
assert [r["chosen"] for r in exercise_rows[:4]] == ["none", "none", "address", "address"]

# Parse and execute unmodified AST nodes from the pinned official source, without importing its backbone dependencies.
kev_path = CACHE / "kev-model-2074cc4.py"
kev_raw = kev_path.read_bytes()
assert hashlib.sha256(kev_raw).hexdigest() == "2ff3b2e67c048e3bc5a0d1a32aafe50804ed315a5564fce06bb4b95697c5271c"
tree = ast.parse(kev_raw)
assignment_lines = {11, 14, 75, 84}
node_names = {"ContextOverflow", "user_tokens", "encode", "branch_mask_batch", "rows_of", "PointerHead"}
nodes = [n for n in tree.body if getattr(n, "lineno", None) in assignment_lines or getattr(n, "name", None) in node_names]
model = next(n for n in tree.body if getattr(n, "name", None) == "DecisionModel")
nodes += [next(n for n in model.body if getattr(n, "name", None) == "_readout")]
kev = {"math": math, "re": re, "torch": torch, "nn": torch.nn}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(kev_path), "exec"), kev)


class LayoutTokenizer:
    """Single-character ASCII content; official special-token IDs are distinct."""

    def convert_tokens_to_ids(self, token):
        return 1000 + kev["SPECIAL"].index(token)

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return SimpleNamespace(input_ids=[ord(c) for c in text])


record = {"state": "S", "questions": [
    {"instr": "Q1", "options": ["A", "B"], "label": 0},
    {"instr": "Q2", "options": ["A", "B"], "label": 1},
]}
enc = kev["encode"](LayoutTokenizer(), record)
mask = kev["branch_mask_batch"]([enc["seg"]], "cpu")[0, 0]
d1, d2 = enc["decide_idx"]
o1, o2 = enc["opt_idx"]
mask_checks = {
    "default_option_isolation_false": enc["option_isolation"] is False,
    "candidate_indices_are_opt_end_tokens": all(enc["ids"][i] == 1003 for oi in enc["opt_idx"] for i in oi),
    "decide_indices_are_decide_tokens": all(enc["ids"][i] == 1004 for i in enc["decide_idx"]),
    "decide_reads_all_own_candidate_endpoints": all(float(mask[d, i]) == 0 for d, oi in zip(enc["decide_idx"], enc["opt_idx"]) for i in oi),
    "later_candidate_reads_earlier_candidate": float(mask[o1[1], o1[0]]) == 0,
    "earlier_candidate_cannot_read_future_candidate": float(mask[o1[0], o1[1]]) < -1e30,
    "questions_are_isolated": all(float(mask[d1, i]) < -1e30 for i in o2) and all(float(mask[d2, i]) < -1e30 for i in o1),
    "ordinary_branch_positions_are_sequential": enc["pos"][:d1 + 1] == list(range(d1 + 1)),
}
assert all(mask_checks.values())
head = kev["PointerHead"](2, dp=2)
with torch.no_grad():
    for linear in (head.q, head.k):
        linear.weight.copy_(torch.eye(2)); linear.bias.zero_()
    hidden = torch.arange(len(enc["ids"]) * 2, dtype=torch.float32).reshape(-1, 2) / 100
    output = kev["_readout"](SimpleNamespace(head=head, device="cpu"), hidden, enc)
    for d, oi, result in zip(enc["decide_idx"], enc["opt_idx"], output):
        torch.testing.assert_close(result, (hidden[oi] @ hidden[d]) / math.sqrt(2))
state_ids, state_pos, question_rows = kev["rows_of"](enc)
assert len(question_rows) == 2 and all(r["ids"][r["decide"]] == 1004 for r in question_rows)

nb = ROOT / page["notebook"]
notebook = json.loads(nb.read_text())
notebook_codes = ["".join(c["source"]).strip() for c in notebook["cells"] if c["cell_type"] == "code"]
assert hashlib.sha256(nb.read_bytes()).hexdigest() == page["notebook_sha256"]
assert code.strip() in notebook_codes
official_source_match = {}
for local, original in ((Path(torch.__file__).parent / "_torch_docs.py", CACHE / "torch-docs-5c4886.py"),
                        (Path(torch.__file__).parent / "nn/functional.py", CACHE / "torch-functional-5c4886.py")):
    official_source_match[original.name] = hashlib.sha256(local.read_bytes()).hexdigest() == hashlib.sha256(original.read_bytes()).hexdigest()
assert all(official_source_match.values())

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-9-17-v1/verify_lesson.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git_version": torch.version.git_version,
                    "device": "cpu", "dtype": "torch.float32", "threads": str(torch.get_num_threads()), "platform": platform.platform()},
    "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest()},
    "snapshot": {"path": page["snapshot"], "sha256": hashlib.sha256(raw).hexdigest()},
    "inline_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "notebook": {"path": page["notebook"], "sha256": page["notebook_sha256"], "contains_same_inline_program": True},
    "inputs": {"features": {k: v.tolist() for k, v in ns["features"].items()}, "questions": {k: v.tolist() for k, v in ns["questions"].items()},
               "Wq": ns["Wq"].tolist(), "Wk": ns["Wk"].tolist(), "orders": ns["orders"], "all_inputs_cpu_float32_no_grad": True},
    "shape_observation": shape,
    "original_stdout": stdout,
    "original_score_calls": rows,
    "executed_scalar_reference": {"logits": scalar_logits, "probabilities": scalar_prob, "max_abs_tolerance": 1e-7},
    "exercise": {"only_substitution": {"old": old, "new": new}, "code_sha256": hashlib.sha256(exercise_code.encode()).hexdigest(),
                 "stdout": exercise_stdout, "independent_choices": [r["chosen"] for r in exercise_rows[:4]],
                 "all_independent_real_candidate_scores_and_probabilities_equal": True},
    "kev_original_pure_nodes": {"source_sha256": hashlib.sha256(kev_raw).hexdigest(), "mask_checks": mask_checks,
                                "readout_indices": {"decide": enc["decide_idx"], "candidate_endpoints": enc["opt_idx"]},
                                "identity_projection_logits": [x.tolist() for x in output], "scale": head.scale,
                                "rows_of_question_count": len(question_rows),
                                "scope": "Unmodified selected original AST nodes; synthetic ASCII layout tokenizer and synthetic hidden states. No real tokenizer/backbone inference or quality measurement."},
    "installed_source_matches_downloaded_official_commit": official_source_match,
    "result": "All assertions passed; exact frozen inline program and stated exercise executed on CPU. No training or model-quality claim tested.",
}
with OUT.open("x") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"result": result["result"], "output": str(OUT.relative_to(ROOT)), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, ensure_ascii=False))
