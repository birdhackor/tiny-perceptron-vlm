"""獨立執行 1.2 的人工 CPU 算例；不訓練或執行神經模型。"""

import ast
import contextlib
import hashlib
import inspect
import io
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v8.json"
freeze = json.loads(FREEZE.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "1.2")
snapshot = ROOT / page["snapshot"]
raw = snapshot.read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
body = raw.decode("utf-8")
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 3

scope = {}
outputs = []
for index, code in enumerate(blocks, start=1):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, f"frozen-1.2-block-{index}", "exec"), scope)
    outputs.append({"block": index, "stdout": stream.getvalue()})
    if index == 1:
        imported_function = scope["split_documents"]
        original_parts = {name: list(documents) for name, documents in scope["parts"].items()}

assert ast.dump(ast.parse(blocks[1]), include_attributes=False) == ast.dump(
    ast.parse(inspect.getsource(imported_function)), include_attributes=False
)
assert len(inspect.getsource(imported_function).splitlines()) == 11
assert scope["split_documents"] is not imported_function
expected_parts = {
    "train": ["鳥看魚。", "狗看貓。"],
    "validation": ["魚看鳥。"],
    "test": ["貓看狗。"],
}
assert original_parts == expected_parts
assert scope["unique"] == ["鳥看魚。", "狗看貓。", "魚看鳥。", "貓看狗。"]
assert scope["split_documents"](scope["docs"]) == expected_parts

skipped_scope = {}
with contextlib.redirect_stdout(io.StringIO()) as skipped_stream:
    exec(blocks[0], skipped_scope)
    exec(blocks[2], skipped_scope)
assert skipped_scope["split_documents"] is imported_function
assert skipped_stream.getvalue().splitlines()[-1] == "True"

original_docs = list(scope["docs"])
duplicate_parts = imported_function(original_docs + ["貓看狗。"])
new_parts = imported_function(original_docs + ["鳥看貓。"])
assert duplicate_parts == original_parts
assert {key: len(value) for key, value in new_parts.items()} == {
    "train": 3, "validation": 1, "test": 1
}

boundary_cases = []
for n in (3, 4, 5, 12, 20):
    documents = [f"document-{i}" for i in range(n)]
    parts = imported_function(documents)
    a, b = max(1, int(n * 0.8)), max(2, int(n * 0.9))
    b = min(b, n - 1)
    a = min(a, b - 1)
    boundary_cases.append({"n": n, "a": a, "b": b,
                           "counts": {key: len(value) for key, value in parts.items()}})
assert boundary_cases[3]["counts"] == {"train": 9, "validation": 1, "test": 2}

from tiny_perceptron.data import toy_documents

toy_docs = toy_documents()
toy_parts = imported_function(toy_docs, seed=42)
toy_counts = {key: len(value) for key, value in toy_parts.items()}
assert len(toy_docs) == len(set(toy_docs)) == 12
assert toy_counts == {"train": 9, "validation": 1, "test": 2}
saved_result_path = ROOT / "docs/course-experiments/results/simple_models.json"
saved_result = json.loads(saved_result_path.read_bytes())
saved_counts = {key: value["records"] for key, value in saved_result["results"]["data"].items()
                if isinstance(value, dict) and "records" in value}
assert saved_counts == toy_counts
assert {"1.12", "1.14", "2.5"}.issubset(saved_result["results"]["sections"])

for n in range(3, 1001):
    documents = [f"document-{i}" for i in range(n)]
    parts = imported_function(documents)
    train, validation, test = (set(parts[key]) for key in ("train", "validation", "test"))
    assert train and validation and test
    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == set(documents)
    assert parts == imported_function(documents, seed=42)

errors = []
for documents in ([], ["one"], ["one", "two"], ["one"] * 4):
    try:
        imported_function(documents)
    except ValueError as error:
        assert str(error) == "至少三份不同文件，才能示範 train/validation/test"
        errors.append({"input": documents, "exception": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError("Expected ValueError for fewer than three unique documents")

document = "貓看狗。狗看貓。"
windows = [document[i:i + 4] for i in range(len(document) - 3)]
assert windows[:2] == ["貓看狗。", "看狗。狗"]
assert windows[0][1:] == windows[1][:-1] == "看狗。"

chapter = (ROOT / page["source"]).read_text()
current_section = re.search(r"^## 1\.2 .*?(?=^## |\Z)", chapter, re.M | re.S).group()
assert current_section.encode("utf-8") == raw
notebook_raw = (ROOT / page["notebook"]).read_bytes()
assert hashlib.sha256(notebook_raw).hexdigest() == page["notebook_sha256"]
notebook = json.loads(notebook_raw)
notebook_codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
assert all(any(code.strip() == cell.strip() for cell in notebook_codes) for code in blocks)

result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-1.2/recompute.py",
    "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                    "device": "CPU; string/list arithmetic only; no tensor computation or neural forward"},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest()},
    "source": {"path": page["snapshot"], "sha256": hashlib.sha256(raw).hexdigest()},
    "repository_code": {"path": "tiny_perceptron/data.py", "sha256": hashlib.sha256((ROOT / "tiny_perceptron/data.py").read_bytes()).hexdigest(),
                        "function_lines": "89-99", "inline_ast_matches_imported": True, "function_line_count": 11},
    "inline_blocks": outputs,
    "original_parts": original_parts,
    "skip_redefinition_final_stdout": skipped_stream.getvalue().splitlines()[-1],
    "duplicate_exercise": {"same_as_original": duplicate_parts == original_parts, "parts": duplicate_parts},
    "new_document_exercise": new_parts,
    "boundary_cases": boundary_cases,
    "linked_toy_experiment": {"documents": toy_docs, "counts": toy_counts,
                              "saved_result": {"path": str(saved_result_path.relative_to(ROOT)),
                                               "sha256": hashlib.sha256(saved_result_path.read_bytes()).hexdigest(),
                                               "counts": saved_counts,
                                               "sections": saved_result["results"]["sections"]},
                              "scope": "Recomputed only document generation/split counts and compared existing result metadata; no experiment run or model invocation"},
    "property_check": "All n=3..1000: nonempty pairwise-disjoint/exhaustive groups; repeated seed=42 gives identical groups",
    "small_input_rejections": errors,
    "overlapping_windows": {"source": document, "windows": windows, "first_pair_overlap": "看狗。"},
    "current_section_equals_frozen": True,
    "notebook": {"path": page["notebook"], "sha256": hashlib.sha256(notebook_raw).hexdigest(), "all_three_teaching_blocks_present": True,
                 "executed": "Only the three frozen-body Python examples above; notebook setup cell not executed"},
    "limitations": ["No training, generalization, final-family inference, teacher forward, model-quality evaluation, or cross-platform execution.",
                    "Seed reproducibility checked only for this Python environment and identical initial document order.",
                    "The lesson's original assertion checks only exact train/validation overlap; expanded reviewer checks are not attributed to the textbook."],
    "result": "All assertions passed"
}
print(json.dumps(result, ensure_ascii=False, indent=2))
