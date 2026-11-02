"""14.19 獨立 technical reviewer 的人工 CPU 核對；不載入任何模型。"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v7.json"
freeze = json.loads(FREEZE.read_text())
page = next(page for page in freeze["pages"] if page["page_id"] == "14.19")
source = ROOT / page["snapshot"]
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
blocks = re.findall(r"```python\n(.*?)\n```", raw.decode(), re.S)
assert len(blocks) == 1
code = blocks[0]
capture = io.StringIO()
namespace = {}
with contextlib.redirect_stdout(capture):
    exec(compile(code, str(source) + ":inline-python", "exec"), namespace)

expected_lines = [
    "F5／A1 原時／A1 移時的 t [5.0, 5.2, 9.2]",
    "原時／移時對 F5 的點積 [2.9877, 1.0123]",
    "移時只改時間對 True",
    "序號版兩個分數相同 True",
    "A1 可讀 F5／F15 True True",
    "A1 可讀下一個聲音塊 False",
    "只改高度後三對 [[-0.5878, -0.809], [0.7071, 0.7071], [1.0, 0.0]]",
]
assert capture.getvalue().splitlines() == expected_lines

# 獨立的 Python scalar 旋轉，不複用 axis_rope 或 torch 的三角函數。
def scalar_rotation(values, coordinates, axes, speeds):
    result = []
    for pair, axis in enumerate(axes):
        a, b = values[2 * pair : 2 * pair + 2]
        angle = coordinates[axis] * speeds[pair]
        result.extend([a * math.cos(angle) - b * math.sin(angle), a * math.sin(angle) + b * math.cos(angle)])
    return result


def close(actual, expected, atol=1e-12):
    assert len(actual) == len(expected)
    assert max(abs(a - b) for a, b in zip(actual, expected, strict=True)) <= atol


vector = [1.0, 0.0] * 3
axes = [0, 1, 2]
speeds = [math.pi / 4] * 3
actual_scores = [float(namespace[name] @ namespace["key"]) for name in ["q_a", "q_b"]]
expected_scores = [2 + math.cos(delta * math.pi / 4) for delta in [0.2, 4.2]]
close(actual_scores, expected_scores)
for coordinate in [[5.0, 0.0, 0.0], [5.2, 0.0, 0.0], [9.2, 0.0, 0.0], [5.2, 1.0, 0.0]]:
    actual = namespace["axis_rope"](namespace["content"], torch.tensor(coordinate, dtype=torch.float64))
    close(actual.tolist(), scalar_rotation(vector, coordinate, axes, speeds))

width = namespace["axis_rope"](namespace["content"], namespace["coordinates"](1300.0, w=1.0))
assert torch.equal(width[:4], namespace["q_a"][:4])
close(width[-2:].tolist(), [math.sqrt(0.5), math.sqrt(0.5)])
assert torch.equal(namespace["moved_h"][:2], namespace["q_a"][:2])
assert torch.equal(namespace["moved_h"][4:], namespace["q_a"][4:])

tags = namespace["tags"]
assert len(tags) == 24
assert [tags[i] for i in [0, 6, 16, 17, 18, 20, 23]] == [
    "video_begin", "frame", "frame", "video_end", "audio_begin", "audio", "audio_end"
]
assert namespace["causal_mask"][20].nonzero().flatten().tolist() == list(range(21))
assert 15 * 250 == 3750

# unit_ms=500 的明示練習：重新產生 key、Q；不能沿用舊座標。
namespace["unit_ms"] = 500.0
unit500_coordinates = [namespace["coordinates"](t) for t in [1250.0, 1300.0, 2300.0]]
unit500_rotated = [namespace["axis_rope"](namespace["content"], coord) for coord in unit500_coordinates]
unit500_scores = [float(q @ unit500_rotated[0]) for q in unit500_rotated[1:]]
close(unit500_scores, [2 + math.cos(delta * math.pi / 4) for delta in [0.1, 2.1]])

# 有意義的小變化：任意內容時，改 t/h/w 各自只改被分配的一對；共同平移保持點積。
nonuniform = torch.tensor([0.2, -0.7, 1.3, 0.1, -0.4, 0.9], dtype=torch.float64)
base = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float64)
base_rotated = namespace["axis_rope"](nonuniform, base).reshape(3, 2)
axis_effects = []
for axis in range(3):
    changed = base.clone()
    changed[axis] += 0.7
    rotated = namespace["axis_rope"](nonuniform, changed).reshape(3, 2)
    effect = [not torch.equal(rotated[i], base_rotated[i]) for i in range(3)]
    assert effect == [i == axis for i in range(3)]
    close(rotated.flatten().tolist(), scalar_rotation(nonuniform.tolist(), changed.tolist(), axes, speeds))
    axis_effects.append(effect)

other = torch.tensor([-0.4, 0.3, 0.2, -1.1, 1.0, -0.8], dtype=torch.float64)
other_coord = torch.tensor([2.3, 1.8, -0.5], dtype=torch.float64)
shift = torch.tensor([9.0, -2.0, 4.0], dtype=torch.float64)
before = float(namespace["axis_rope"](nonuniform, base) @ namespace["axis_rope"](other, other_coord))
after = float(namespace["axis_rope"](nonuniform, base + shift) @ namespace["axis_rope"](other, other_coord + shift))
assert abs(before - after) <= 1e-12

# 文字所列的對數檢查確實會拒絕兩對、四對的向量。
error_cases = []
for size in [4, 8]:
    try:
        namespace["axis_rope"](torch.ones(size, dtype=torch.float64), base)
    except ValueError as error:
        assert str(error) == "每一對都要有軸分配"
        error_cases.append({"vector_size": size, "message": str(error)})
    else:
        raise AssertionError("expected a ValueError")

# 對照 Notebook：教材程式逐 byte 包含在凍結 Notebook 的 code cell。
notebook_path = ROOT / page["notebook"]
notebook_raw = notebook_path.read_bytes()
assert hashlib.sha256(notebook_raw).hexdigest() == page["notebook_sha256"]
notebook = json.loads(notebook_raw)
notebook_code = "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
assert code in notebook_code

report = {
    "scope": "artificial inline CPU demonstration; no model loading, training, inference or workflow",
    "environment": {"python": sys.version, "torch": torch.__version__, "device": "CPU", "platform": platform.platform()},
    "source": {"path": str(source.relative_to(ROOT)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
    "freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest()},
    "inline_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "inline_stdout": capture.getvalue(),
    "score_comparison": {"torch": actual_scores, "scalar_cosine": expected_scores, "atol": 1e-12},
    "width_exercise": width.reshape(3, 2).tolist(),
    "sequence": {"length": len(tags), "F5": 6, "F15": 16, "A1": 20, "A1_visible_indices": list(range(21)), "F15_time_ms": 3750},
    "unit500_exercise": {"coordinates": [coord.tolist() for coord in unit500_coordinates], "scores": unit500_scores},
    "nonuniform_axis_effects": axis_effects,
    "common_shift_dot_product": {"before": before, "after": after, "absolute_error": abs(before - after)},
    "invalid_pair_count": error_cases,
    "notebook": {"path": page["notebook"], "bytes": len(notebook_raw), "sha256": hashlib.sha256(notebook_raw).hexdigest(), "contains_exact_inline_code": True},
    "result": "PASS: all assertions satisfied",
}
print(json.dumps(report, ensure_ascii=False, indent=2))
