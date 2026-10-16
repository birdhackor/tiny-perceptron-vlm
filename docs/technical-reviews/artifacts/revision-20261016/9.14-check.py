"""9.14 的 CPU 算例核對；沒有訓練，也不測量 GPU 或校準效果。"""

from contextlib import redirect_stdout
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch

from scripts.reading_time import lesson_slices


root = Path(__file__).resolve().parents[4]
source = root / "course/chapters/09.md"
body = lesson_slices(source.read_text(encoding="utf-8"))["9.14"]
snippet = re.findall(r"```python\n(.*?)```", body, re.S)[0]
captured = io.StringIO()
with redirect_stdout(captured):
    exec(compile(snippet, "course/chapters/09.md#9.14", "exec"), {})

y = torch.tensor([1.0, 0.0, 0.0])
rows = []
for p, expected in [
    (torch.tensor([0.8, 0.1, 0.1]), 0.06),
    (torch.tensor([0.2, 0.7, 0.1]), 1.14),
]:
    ce = float(-p[0].log())
    brier = float(((p - y) ** 2).sum())
    assert p.shape == y.shape == (3,)
    assert p.dtype == y.dtype == torch.float32
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert abs(brier - expected) < 1e-6
    rows.append({"p": p.tolist(), "target": y.tolist(), "CE": ce,
                 "Brier_sum": brier, "expected_Brier_sum": expected})

result = {
    "command": "PYTHONPATH=. uv run python docs/technical-reviews/artifacts/revision-20261016/9.14-check.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": "cpu", "dtype": "torch.float32", "platform": platform.platform()},
    "source_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    "original_snippet_stdout": captured.getvalue(),
    "cases": rows,
    "checks": "Original snippet executed verbatim; shape, dtype, probability normalization, and both Brier sums passed.",
    "tolerance": "absolute error < 1e-6 for displayed float32 Brier values",
    "scope": "Only the two deterministic CPU arithmetic examples; no training, calibration-quality or GPU claim.",
}
destination = root / "docs/technical-reviews/artifacts/revision-20261016/9.14-execution.json"
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
