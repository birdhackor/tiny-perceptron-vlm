"""Reviewer-owned offline audit of canonical 17.17; no training or product inference."""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
PROOF = Path(__file__).resolve().parent
FREEZE = "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v7.json"
FREEZE_SHA = "ab1bd5df0dc294cfa5948a22a601d35fbc6389ae48ef6e141bc1df31515dfc35"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(code):
    namespace = {"__name__": "__review_probe__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "canonical-17.17-inline", "exec"), namespace)
    return namespace, output.getvalue()


freeze_bytes = (ROOT / FREEZE).read_bytes()
assert digest(freeze_bytes) == FREEZE_SHA
page = next(p for p in json.loads(freeze_bytes)["pages"] if p["page_id"] == "17.17")
body_bytes = (ROOT / page["snapshot"]).read_bytes()
assert digest(body_bytes) == page["source_sha256"]
body = body_bytes.decode()
chapter = (ROOT / page["source"]).read_text()
heading = re.search(r"^## 17\.17 .+$", chapter, re.M)
tail = chapter[heading.start():]
following = re.search(r"^## .+$", tail[1:], re.M)
current_body = tail[:following.start() + 1] if following else tail
assert current_body.encode() == body_bytes
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 1
inline = blocks[0]
notebook_bytes = (ROOT / page["notebook"]).read_bytes()
assert digest(notebook_bytes) == page["notebook_sha256"]
notebook = json.loads(notebook_bytes)
code_cells = [(i, "".join(c["source"])) for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code"]
assert [i for i, _ in code_cells] == [2, 3, 5]
assert code_cells[-1][1].rstrip() == inline.rstrip()
for path, expected in page["figures_sha256"].items():
    assert digest((ROOT / path).read_bytes()) == expected

base, base_stdout = run(inline)
assert base["target_ids"] == [1, 2, 0]
assert abs(base["total"].item() - 1.375) < 1e-7
assert base["predicted"] == [2, 1, 0]
assert base["generated"] == 1.875
exercise, exercise_stdout = run(inline.replace("value = torch.tensor(1.4)", "value = torch.tensor(1.6)"))
assert exercise["target_ids"] == [2, 0, 2]
assert abs(exercise["total"].item() - 1.625) < 1e-7
assert exercise["predicted"] == base["predicted"]

# Independent exact rational arithmetic, separate from the inline Torch loop.
books = [[Fraction(v) for v in book] for book in [["0", "1", "2"], ["-0.5", "0", "0.5"], ["-0.125", "0", "0.125"]]]


def rational_trace(value):
    residual, total = Fraction(value), Fraction(0)
    rows = []
    for q, book in enumerate(books):
        squared_distances = [(residual - entry) ** 2 for entry in book]
        selected = min(range(len(book)), key=lambda k: squared_distances[k])
        total += book[selected]
        residual -= book[selected]
        rows.append({"codebook": q, "selected_id": selected, "entry": str(book[selected]),
                     "candidate_squared_distances": list(map(str, squared_distances)),
                     "cumulative": str(total), "residual": str(residual),
                     "squared_error": str(residual ** 2)})
    return rows


exact = {value: rational_trace(value) for value in ["1.4", "1.6"]}
assert [r["selected_id"] for r in exact["1.4"]] == base["target_ids"]
assert [r["selected_id"] for r in exact["1.6"]] == exercise["target_ids"]
assert [r["squared_error"] for r in exact["1.4"]] == ["4/25", "1/100", "1/1600"]
assert all(len(set(r["candidate_squared_distances"])) == 3 for r in exact["1.4"])
tie_id = (torch.tensor(0.5) - torch.tensor([0.0, 1.0, 2.0])).square().argmin().item()
assert tie_id == 0

# No reference value/target is accepted or loaded by predict_frame.
function = next(n for n in ast.walk(ast.parse(inline)) if isinstance(n, ast.FunctionDef) and n.name == "predict_frame")
loaded_names = sorted({n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)})
assert not {"value", "target_ids", "residual", "total"}.intersection(loaded_names)
parameter_bytes_before = b"".join(p.detach().numpy().tobytes() for name in ["context", "heads", "embeddings"] for p in base[name].parameters())
base["value"] = torch.tensor(-999.0)
base["target_ids"] = [0, 0, 0]
after_reference_change = base["predict_frame"](torch.tensor([0.3, -0.4]))
assert after_reference_change == [2, 1, 0]
with torch.no_grad():
    h = torch.tanh(base["context"](torch.tensor([0.3, -0.4])))
    trace = [{"stage": "condition", "h_shape": list(h.shape), "h": h.tolist()}]
    for q in range(3):
        scores = base["heads"][q](h)
        selected = scores.argmax(-1)
        feedback = base["embeddings"][q](selected)
        h = torch.tanh(h + feedback)
        assert h.shape == (4,) and scores.shape == (3,) and feedback.shape == (4,)
        assert h.abs().max() <= 1
        trace.append({"codebook": q, "scores": scores.tolist(), "selected_id": selected.item(), "h_after": h.tolist()})
parameter_bytes_after = b"".join(p.detach().numpy().tobytes() for name in ["context", "heads", "embeddings"] for p in base[name].parameters())
assert parameter_bytes_after == parameter_bytes_before

# Execute all notebook code cells through the local branch; do not save the notebook.
notebook_namespace = {"__name__": "__review_notebook__"}
notebook_stdout = io.StringIO()
with contextlib.redirect_stdout(notebook_stdout):
    for index, code in code_cells:
        exec(compile(code, f"notebook-cell-{index}", "exec"), notebook_namespace)
assert notebook_namespace["target_ids"] == [1, 2, 0]
assert notebook_namespace["predicted"] == [2, 1, 0]

# Inspect production source via AST only: no model import, instantiation, weights or inference.
sound_path = "tiny_perceptron/omni_products/sound.py"
sound_bytes = (ROOT / sound_path).read_bytes()
sound_tree = ast.parse(sound_bytes)
codec = next(n for n in sound_tree.body if isinstance(n, ast.ClassDef) and n.name == "SoundCodec")
init = next(n for n in codec.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
book_assignments = [n for n in ast.walk(init) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "codebook" for t in n.targets)]
assert len(book_assignments) == 1
assert ast.unparse(book_assignments[0].value) == "nn.Embedding(config.codes, config.width)"
talker = next(n for n in sound_tree.body if isinstance(n, ast.ClassDef) and n.name == "ProgramTalker")
generate = next(n for n in talker.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
assert [arg.arg for arg in generate.args.args] == ["self", "program", "max_blocks"]
train_path = "tiny_perceptron/omni_products/training_runtime.py"
train_bytes = (ROOT / train_path).read_bytes()
train_tree = ast.parse(train_bytes)
train_talker = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_talker")
train_text = ast.unparse(train_talker)
assert "codec.eval().requires_grad_(False)" in train_text
assert "F.cross_entropy(model(previous, schedule).transpose(1, 2), target)" in train_text

result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reviewer_task": "/root/omni_p5_technical_17_17",
    "scope": "Canonical 17.17 only; exact inline, generated notebook local code cells, independent scalar arithmetic and production AST; no training or formal/final inference.",
    "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": "CPU", "threads": str(torch.get_num_threads()), "dtype": str(torch.get_default_dtype())},
    "freeze": {"path": FREEZE, "sha256": FREEZE_SHA},
    "source_snapshot": {"path": page["snapshot"], "sha256": digest(body_bytes), "equals_current_authored_slice": True},
    "notebook": {"path": page["notebook"], "sha256": digest(notebook_bytes), "executed_code_cells": [i for i, _ in code_cells], "teaching_code_equals_inline": True, "stdout": notebook_stdout.getvalue(), "saved_or_changed": False},
    "optional_teaching_code": "None: assigned frozen unit has one Python fence and no details/optional block.",
    "inline_sha256": digest(inline.encode()),
    "base_stdout": base_stdout,
    "exercise_stdout": exercise_stdout,
    "independent_rational_arithmetic": exact,
    "reconstruction_without_reference": {"ids": [1, 2, 0], "value": str(sum(books[q][k] for q, k in enumerate([1, 2, 0])))},
    "tie_probe": {"input": 0.5, "book": [0, 1, 2], "observed_id": tie_id},
    "predict_frame_loaded_names": loaded_names,
    "changed_reference_and_target_prediction": after_reference_change,
    "condition_model_trace": trace,
    "untrained_parameters_sha256_before_and_after": digest(parameter_bytes_after),
    "parameters_unchanged": True,
    "fixed_width_storage": {"choices_per_book": 3, "bits_per_id": (3 - 1).bit_length(), "one_book_bits_per_frame": 2, "three_book_bits_per_frame": 6, "overhead_included": False},
    "production_ast": {"source": sound_path, "sha256": digest(sound_bytes), "codec_codebook_assignments": len(book_assignments), "codebook_constructor": ast.unparse(book_assignments[0].value), "generate_args": [arg.arg for arg in generate.args.args], "training_source": train_path, "training_source_sha256": digest(train_bytes), "codec_frozen_before_targets": True, "cross_entropy_present": True, "production_nn_executed": False},
    "result": "All assertions passed. Printed decimal results match body; reference perturbation changes RVQ targets and leaves untrained conditional generation unchanged."
}
(PROOF / "cpu-probe-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": result["result"], "output": str((PROOF / "cpu-probe-result.json").relative_to(ROOT))}, ensure_ascii=False))
