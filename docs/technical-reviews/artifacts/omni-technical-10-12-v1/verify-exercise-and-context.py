"""重跑正文改一行的練習，並有界核對10.12明確連結的三個前置。"""
import contextlib
import hashlib
import io
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from tiny_perceptron.data import IGNORE
from tiny_perceptron.multimodal import VisionEncoder, expand_modalities, patchify, unpatchify

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
snapshot = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-reader-fix02/sources/10.12.md"
code = re.findall(r"```python\n(.*?)```", snapshot.read_text(), re.S)[0]
original_line = 'times_b = torch.tensor([0.0, 500.0, 1000.0, 1500.0])'
exercise_line = 'times_b = torch.tensor([0.0, 500.0, 1750.0, 3000.0])'
assert code.count(original_line) == 1
exercise_code = code.replace(original_line, exercise_line)
torch.set_num_threads(1)
torch.manual_seed(1012)
exercise_stdout = io.StringIO()
with contextlib.redirect_stdout(exercise_stdout):
    exec(compile(exercise_code, "10.12-exercise-modified-times-b", "exec"), {})
assert exercise_stdout.getvalue().splitlines()[1] == "B 1500 ms 採影格 1"
invalid_code = exercise_code.replace(exercise_line, 'times_b = torch.tensor([100.0, 500.0, 1750.0, 3000.0])')
invalid_stdout = io.StringIO()
try:
    with contextlib.redirect_stdout(invalid_stdout):
        exec(compile(invalid_code, "10.12-exercise-invalid-start", "exec"), {})
except ValueError as error:
    invalid_error = str(error)
    assert invalid_error == "從 0 ms 有畫面，時間戳嚴格遞增"
else:
    raise AssertionError("invalid first timestamp accepted")
grid = torch.arange(16.0).reshape(1, 1, 4, 4)
patches = patchify(grid, 2)
assert patches[0, 1].tolist() == [2.0, 3.0, 6.0, 7.0]
assert torch.equal(unpatchify(patches, 1, 4, 4, 2), grid)
vision = VisionEncoder(width=16)
assert tuple(vision.position.shape) == (1, 16, 16)
ids = torch.tensor([1, 5, 4, 73, 2])
labels = torch.tensor([IGNORE, IGNORE, IGNORE, 73, 2])
x, y = expand_modalities(ids, labels, nn.Embedding(264, 8), {5: torch.randn(3, 8)}, {5})
assert tuple(x.shape) == (1, 6, 8)
assert y.tolist() == [[IGNORE, IGNORE, IGNORE, IGNORE, 73, 2]]
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-10-12-v1/verify-exercise-and-context.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "threads": "1", "seed": "1012"},
    "inline_source_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    "exact_substitution": {"before": original_line, "after": exercise_line},
    "exercise_stdout": exercise_stdout.getvalue().splitlines(),
    "invalid_first_time": {"value_ms": 100, "error": invalid_error, "stdout_before_error": invalid_stdout.getvalue().splitlines()},
    "linked_context_spot_checks": {"10.2_patch_position_1": patches[0, 1].tolist(), "10.2_roundtrip": True, "10.4_existing_vision_position_shape": list(vision.position.shape), "10.7_expanded_shape": list(x.shape), "10.7_next_targets": y.tolist()},
    "repository_code": {"path": "tiny_perceptron/multimodal.py", "sha256": hashlib.sha256((ROOT / 'tiny_perceptron/multimodal.py').read_bytes()).hexdigest()},
    "result": "All assertions passed. Exercise rerun uses the literal inline code with only the specified line changed. Linked context checks are bounded mechanism checks, not a new full review of prerequisite lessons; no training or model-quality measurement.",
}
with (OUT / "exercise-context-summary.json").open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
