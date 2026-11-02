"""Bounded CPU review of the exact frozen W.5 code; no project helpers or training."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
from datetime import UTC, datetime

import torch


ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v4.json"
EXPECTED = "ab8c7f5ad3c5e5f87ac6fcb95527a9d1f38ba4b7c407528a2d0313dbf5f592a5"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(name, data):
    with (DEST / name).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run_block(code, namespace, filename):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, filename, "exec"), namespace)
    return output.getvalue()


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert torch.get_default_device() == torch.device("cpu")
freeze = json.loads(FREEZE.read_text())
page = next(item for item in freeze["pages"] if item["page_id"] == "W.5")
snapshot = ROOT / page["snapshot"]
assert digest(snapshot) == EXPECTED == page["source_sha256"]
assert page["figures_sha256"] == {}
raw = snapshot.read_text()
blocks = re.findall(r"^```python\n(.*?)^```", raw, flags=re.M | re.S)
assert len(blocks) == 2

canonical_ref = freeze["canonical_source_freeze"]
canonical_path = ROOT / canonical_ref["path"]
assert digest(canonical_path) == canonical_ref["sha256"]
canonical = json.loads(canonical_path.read_text())
canonical_page = next(item for item in canonical["pages"] if item["page_id"] == "first-steps")
canonical_snapshot = ROOT / canonical_page["snapshot"]
assert digest(canonical_snapshot) == canonical_page["source_sha256"]
canonical_section = re.search(
    r"^## W\.5\b.*?(?=^## |\Z)", canonical_snapshot.read_text(), flags=re.M | re.S
).group(0)
live_section = re.search(
    r"^## W\.5\b.*?(?=^## |\Z)", (ROOT / page["source"]).read_text(), flags=re.M | re.S
).group(0)
assert canonical_section == live_section == raw

namespace = {}
stdout_1 = run_block(blocks[0], namespace, f"{page['snapshot']}:block1")
result = namespace["result"]
recipes = namespace["recipes"]
weights = namespace["weights"]
assert result.tolist() == [[40.0, 1.0], [80.0, 6.0]]
assert tuple(recipes.shape) == (2, 3)
assert tuple(weights.shape) == (3, 2)
assert tuple(result.shape) == (2, 2)
assert recipes.dtype == weights.dtype == result.dtype == torch.float32
assert result.device.type == "cpu"

stdout_2 = run_block(blocks[1], namespace, f"{page['snapshot']}:block2")
layer = namespace["layer"]
baseline = layer(recipes).tolist()
assert baseline == [[45.0, 1.0], [85.0, 6.0]]
assert layer.weight.tolist() == [[10.0, 20.0, 30.0], [0.0, 1.0, 4.0]]
assert layer.bias.tolist() == [5.0, 0.0]
assert tuple(layer.weight.shape) == (2, 3)
assert tuple(weights.T.shape) == (2, 3)
assert isinstance(layer(recipes).tolist(), list)
assert all(isinstance(row, list) for row in layer(recipes).tolist())
assert layer.weight.requires_grad and layer.bias.requires_grad
assert torch.is_grad_enabled()
assert layer.weight.grad_fn is None and layer.bias.grad_fn is None

exercise_code = blocks[1].replace("[5.0, 0.0]", "[8.0, 0.0]")
assert exercise_code != blocks[1]
stdout_exercise = run_block(exercise_code, namespace, f"{page['snapshot']}:block2-exercise")
exercise_layer = namespace["layer"]
exercise = exercise_layer(recipes).tolist()
assert exercise == [[48.0, 1.0], [88.0, 6.0]]
delta = (exercise_layer(recipes) - layer(recipes)).tolist()
assert delta == [[3.0, 0.0], [3.0, 0.0]]

try:
    recipes @ weights[:2]
except RuntimeError as error:
    mismatch = str(error)
else:
    raise AssertionError("incompatible inner dimensions unexpectedly succeeded")

grad_before = torch.is_grad_enabled()
with torch.no_grad():
    grad_during = torch.is_grad_enabled()
    copied = exercise_layer.weight.copy_(weights.T)
    copy_grad_fn = str(copied.grad_fn)
grad_after = torch.is_grad_enabled()
assert grad_before and not grad_during and grad_after
assert copy_grad_fn == "None"

prerequisites = []
for item in freeze["pages"]:
    if item["page_id"] == "W.3":
        p = ROOT / item["snapshot"]
        assert digest(p) == item["source_sha256"]
        prerequisites.append({
            "page_id": "W.3", "path": item["snapshot"], "sha256": digest(p),
            "inspection_scope": "Read text for axis, shape and tolist prerequisites only; no figure render or visual verification claimed.",
        })
for item in canonical["pages"]:
    if item["page_id"] in ("1.6", "2.1"):
        p = ROOT / item["snapshot"]
        assert digest(p) == item["source_sha256"]
        prerequisites.append({
            "page_id": item["page_id"], "path": item["snapshot"], "sha256": digest(p),
            "inspection_scope": "Read linked text and inline code to confirm nn import and first copy_ use; no figure verification claimed.",
        })

torch_root = Path(torch.__file__).parent
selected = {
    "nn/__init__.py": [(2, 17)],
    "nn/modules/linear.py": [(53, 54), (60, 77), (108, 114), (130, 131)],
    "autograd/grad_mode.py": [(22, 36), (81, 86)],
    "_tensor_docs.py": [(1147, 1156), (5446, 5454), (6770, 6781)],
    "_torch_docs.py": [(7832, 7842), (7909, 7918), (7934, 7938)],
}
official_sources = []
for suffix, spans in selected.items():
    p = torch_root / suffix
    lines = p.read_text().splitlines()
    official_sources.append({
        "file": f"torch/{suffix}", "installed_path": str(p), "sha256": digest(p),
        "url": f"https://github.com/pytorch/pytorch/blob/{torch.version.git_version}/torch/{suffix}",
        "access_method": "Read installed PyTorch official source and API docstrings; pinned URL identifies package-reported git revision, not a separate remote fetch.",
        "excerpts": [{"lines": f"{lo}-{hi}", "text": "\n".join(lines[lo-1:hi])} for lo, hi in spans],
    })

chapter_matches = {}
for chapter in ("01", "02"):
    p = ROOT / f"course/chapters/{chapter}.md"
    found = [{"line": i, "text": line} for i, line in enumerate(p.read_text().splitlines(), 1)
             if re.search(r"\bnn\b|@|Linear|copy_", line)]
    chapter_matches[p.relative_to(ROOT).as_posix()] = {"sha256": digest(p), "matches": found}
assert not any(re.search(r"@|Linear|copy_", entry["text"])
               for entry in chapter_matches["course/chapters/01.md"]["matches"])

save_new("source-inspection.json", {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_w_5",
    "freeze": {"path": FREEZE.relative_to(ROOT).as_posix(), "sha256": digest(FREEZE)},
    "canonical_freeze": canonical_ref,
    "section": page,
    "canonical_and_live_section_match": True,
    "official_package_version": torch.__version__,
    "official_package_git_revision": torch.version.git_version,
    "original_sources": official_sources,
    "prerequisites": prerequisites,
    "chapter_route_search": chapter_matches,
    "figure_scope": "W.5 has no figure references; figure_sha256 is exactly {}. Prerequisite figures were not rendered or claimed reviewed.",
})
save_new("execution.json", {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_w_5",
    "source_snapshot": page["snapshot"], "source_sha256": EXPECTED,
    "command": "timeout 30s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-w-5-v1/inspect_and_replay.py",
    "environment": {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_git_revision": torch.version.git_version, "platform": platform.platform(),
        "device": "cpu", "dtype": str(result.dtype), "threads": str(torch.get_num_threads()),
        "interop_threads": str(torch.get_num_interop_threads()),
    },
    "method": "Extract and exec both exact inline python blocks sequentially in the same namespace; exercise changes only literal [5.0, 0.0] to [8.0, 0.0]. Additional checks inspect dimensions, copying, gradient context, and incompatible matrix shapes. No training or project model helper.",
    "blocks": [{"index": i + 1, "sha256": hashlib.sha256(code.encode()).hexdigest()}
               for i, code in enumerate(blocks)],
    "block1": {"stdout": stdout_1, "recipes_shape": list(recipes.shape),
               "weights_shape": list(weights.shape), "result": result.tolist(),
               "result_shape": list(result.shape)},
    "block2": {"stdout": stdout_2, "weight": layer.weight.tolist(), "bias": layer.bias.tolist(),
               "result": baseline, "weight_shape": list(layer.weight.shape),
               "weights_transpose_shape": list(weights.T.shape),
               "parameter_requires_grad_retained": True, "parameter_grad_fn": None,
               "tolist_types": "list[list[float]]"},
    "exercise": {"stdout": stdout_exercise, "bias": exercise_layer.bias.tolist(),
                 "result": exercise, "delta_from_baseline": delta},
    "invalid_inner_dimensions": {"left_shape": [2, 3], "right_shape": [2, 2],
                                 "exception_type": "RuntimeError", "message": mismatch},
    "no_grad_probe": {"before": grad_before, "during": grad_during, "after": grad_after,
                      "copy_result_grad_fn": copy_grad_fn},
    "result": "All assertions passed; all example and exercise values exactly match."
})
print("W.5 exact frozen blocks and exercise: all assertions passed")
