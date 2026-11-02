"""9.11 technical review: execute the frozen inline program and its exact exercise.

CPU only. The optional product check performs a single random-weight forward;
it does not train, fetch weights, invoke Jev, or execute an application action.
"""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
freeze = json.loads(FREEZE.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "9.11")
snapshot = ROOT / page["snapshot"]
raw = snapshot.read_bytes()
digest = lambda b: hashlib.sha256(b).hexdigest()
assert digest(raw) == page["source_sha256"]
chapter = (ROOT / page["source"]).read_text()
start = chapter.index("## 9.11 ")
end = chapter.index("\n## ", start + 1) + 1
assert chapter[start:end].encode() == raw
assert page["figures_sha256"] == {}
inline_blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
assert len(inline_blocks) == 1
inline = inline_blocks[0]


def execute(code, name):
    stdout = io.StringIO()
    namespace = {"__name__": "__main__"}
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, name, "exec"), namespace)
    return namespace, stdout.getvalue()


original, original_stdout = execute(inline, "frozen-9.11-inline")
expected_stdout = (
    "查卡片進度 0.844\n"
    "更改地址 0.042\n"
    "以上皆不適合 0.114\n"
    "程式依規則採取的動作： 開啟卡片進度頁\n"
)
assert original_stdout == expected_stdout
assert original["scores"].shape == (3,)
assert original["scores"].dtype == torch.float32
assert isinstance(original["probability"].tolist(), list)
assert original["best"] == "查卡片進度"
assert int(original["probability"].argmax()) == 0
assert original["actions"][original["best"]] == "開啟卡片進度頁"
assert original["probability"].argsort(descending=True).tolist() == [0, 2, 1]

scores = [2.0, -1.0, 0.0]
denominator = sum(math.exp(s) for s in scores)
reference = [math.exp(s) / denominator for s in scores]
observed = original["probability"].tolist()
max_abs_error = max(abs(a - b) for a, b in zip(reference, observed, strict=True))
assert max_abs_error < 1e-7
rounded = [round(p, 3) for p in observed]
assert rounded == [0.844, 0.042, 0.114]
assert sum(Decimal(str(p)) for p in rounded) == Decimal("1.000")
assert abs(sum(observed) - 1.0) < 1e-7

old = '"question": "下一步服務？"'
new = '"question": "是否已有訂單編號？"'
assert inline.count(old) == 1
exercise = inline.replace(old, new)
changed, exercise_stdout = execute(exercise, "frozen-9.11-question-only-exercise")
assert changed["judgment"]["question"] == "是否已有訂單編號？"
assert changed["judgment"]["state"] == original["judgment"]["state"]
assert changed["judgment"]["candidates"] == original["judgment"]["candidates"]
assert exercise_stdout == original_stdout
assert torch.equal(changed["probability"], original["probability"])

notebook_path = ROOT / page["notebook"]
notebook = json.loads(notebook_path.read_bytes())
assert digest(notebook_path.read_bytes()) == page["notebook_sha256"]
code_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
assert len(code_cells) == 2
assert code_cells[1].rstrip() == inline.rstrip()
notebook_namespace, setup_stdout = execute(code_cells[0], "9.11-notebook-local-setup")
assert notebook_namespace["in_colab"] is False
nb_stdout = io.StringIO()
with contextlib.redirect_stdout(nb_stdout):
    exec(compile(code_cells[1], "9.11-notebook-inline", "exec"), notebook_namespace)
assert nb_stdout.getvalue() == original_stdout

# Directly inspect the current course implementation by a forward on 3 rows.
# No optimizer, backward, parameter update, pretrained weight, or dataset run.
from tiny_perceptron.lesson_products.data import render
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, Tokenizer, padded

torch.set_num_threads(1)
torch.manual_seed(911)
tokenizer = Tokenizer()
state = render({"task": "card", "application": "1234", "device": "手機"})
candidate_ids = ["card", "address", "none"]
sequences = [tokenizer.encode(state, "choice", c) for c in candidate_ids]
model = CandidateScorer(Config(len(tokenizer), width=8, heads=2, ff=16, c_layers=1)).cpu().eval()
tokens = padded(sequences, 128, "cpu")
with torch.no_grad():
    product_scores = model(tokens)
assert product_scores.shape == (3,)
assert product_scores.isfinite().all()
assert product_scores.device.type == "cpu"

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-9-11-v1/verify_inline.py",
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "platform": platform.platform(),
        "device": "cpu",
    },
    "freeze": str(FREEZE.relative_to(ROOT)),
    "freeze_sha256": digest(FREEZE.read_bytes()),
    "source_snapshot": page["snapshot"],
    "source_sha256": digest(raw),
    "current_slice_equal_snapshot": True,
    "inline_code_sha256": digest(inline.encode()),
    "figure_sha256": {},
    "original_stdout": original_stdout,
    "input_scores": scores,
    "scores_shape": list(original["scores"].shape),
    "scores_dtype": str(original["scores"].dtype),
    "reference_method": "Python binary64 exp(score)/sum(exp(scores)); independently compared with torch float32",
    "reference_probabilities": reference,
    "observed_probabilities": observed,
    "maximum_absolute_error": max_abs_error,
    "tolerance": 1e-7,
    "rounded_probabilities": rounded,
    "rounded_decimal_sum": str(sum(Decimal(str(p)) for p in rounded)),
    "unrounded_python_sum": sum(observed),
    "unrounded_torch_sum": float(original["probability"].sum()),
    "argmax_index": int(original["probability"].argmax()),
    "selected_candidate": original["best"],
    "second_candidate": original["judgment"]["candidates"][2],
    "second_probability": observed[2],
    "exercise_changed_only_question": True,
    "exercise_stdout": exercise_stdout,
    "exercise_stdout_identical": True,
    "notebook": page["notebook"],
    "notebook_sha256": digest(notebook_path.read_bytes()),
    "notebook_inline_exact_match": True,
    "notebook_local_setup_executed": True,
    "notebook_inline_executed": True,
    "notebook_stdout_identical": True,
    "course_model_forward": {
        "state": state,
        "question_id": "choice",
        "candidates": candidate_ids,
        "token_shape": list(tokens.shape),
        "score_shape": list(product_scores.shape),
        "scores": product_scores.tolist(),
        "seed": 911,
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameter_updates": 0,
        "training_steps": 0,
        "quality_claim": "none; random-weight CPU forward verifies the input/scoring interface only",
    },
    "jev_service_calls": 0,
    "kev_large_model_executions": 0,
    "result": "all frozen-inline, numeric, exercise, notebook-local, and random-forward assertions passed",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
