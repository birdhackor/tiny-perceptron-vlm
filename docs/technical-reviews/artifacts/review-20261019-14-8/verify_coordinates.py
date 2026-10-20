"""核對位置縮放算例、分數位置旋轉與圖中的全部座標。"""

import ast
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import xml.etree.ElementTree as ET

import torch


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.modern import rope as library_rope


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


chapter_path = ROOT / "course/chapters/14.md"
chapter = chapter_path.read_text()
rope_block = next(
    block for block in re.findall(r"```python\n(.*?)```", chapter, re.S) if "def rope(" in block
)
tree = ast.parse(rope_block)
definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rope")
namespace = {"torch": torch}
exec(compile(ast.Module(body=[definition], type_ignores=[]), str(chapter_path), "exec"), namespace)
lesson_rope = namespace["rope"]

length, extended = 8, 16
coordinates = [m * length / extended for m in range(extended)]
assert coordinates == [m / 2 for m in range(16)]
assert len(coordinates) == 16
assert max(coordinates) == 7.5 and max(coordinates) < 8
assert coordinates[-1] != 7
assert coordinates[15] - coordinates[0] == 7.5
assert all(coordinates[m + 1] - coordinates[m] == 0.5 for m in range(15))

exercise = [m * length / 24 for m in range(24)]
assert len(exercise) == 24
assert math.isclose(exercise[5], 5 / 3, rel_tol=0, abs_tol=1e-15)
assert math.isclose(exercise[23], 23 / 3, rel_tol=0, abs_tol=1e-15)
assert round(exercise[5], 4) == 1.6667
assert round(exercise[23], 4) == 7.6667

positions = torch.arange(16) * length / extended
x = torch.linspace(-1, 1, 128).reshape(1, 1, 16, 8)
before = x.clone()
actual = library_rope(x, positions)
lesson_actual = lesson_rope(x, positions)
expected = torch.empty_like(x)
for m in range(16):
    for pair in range(4):
        angle = (m / 2) * 10000 ** (-(2 * pair) / 8)
        a, b = x[0, 0, m, 2 * pair : 2 * pair + 2].tolist()
        expected[0, 0, m, 2 * pair] = a * math.cos(angle) - b * math.sin(angle)
        expected[0, 0, m, 2 * pair + 1] = a * math.sin(angle) + b * math.cos(angle)
max_error = (actual - expected).abs().max().item()
assert max_error < 1e-6
assert torch.equal(actual, lesson_actual)
assert actual.shape == x.shape and torch.equal(x, before)
norm_error = (actual.square().sum(-1) - x.square().sum(-1)).abs().max().item()
assert norm_error < 2e-6

svg_path = ROOT / "course/figures/rewrite-14-8-position-interpolation.svg"
svg = ET.fromstring(svg_path.read_bytes())
ns = {"svg": "http://www.w3.org/2000/svg"}
texts = [(int(node.attrib["x"]), int(node.attrib["y"]), node.text) for node in svg.findall(".//svg:text", ns)]
cards = []
for m in range(16):
    column, row = m % 8, m // 8
    x_coord, y_offset = 55 + 74 * column, 204 * row
    character = next(t for x_pos, y_pos, t in texts if x_pos == x_coord and y_pos == 147 + y_offset)
    original = next(t for x_pos, y_pos, t in texts if x_pos == x_coord and y_pos == 187 + y_offset)
    scaled = next(t for x_pos, y_pos, t in texts if x_pos == x_coord and y_pos == 263 + y_offset)
    assert int(original) == m
    assert float(scaled) == coordinates[m]
    cards.append({"character": character, "position": m, "coordinate": float(scaled)})
assert "".join(card["character"] for card in cards) == "校慶週六上午開始請在操場入口集合"
arrows = svg.findall(".//svg:line", ns)
assert len(arrows) == 16
assert all(float(line.attrib["y2"]) > float(line.attrib["y1"]) for line in arrows)

print(json.dumps({
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "dtype": str(x.dtype), "positions_dtype": str(positions.dtype)},
    "input_configuration": {"x_shape": list(x.shape), "x_values": "linspace(-1, 1, 128)", "positions": positions.tolist(), "mask": "none; coordinate and RoPE calculation only", "base": 10000},
    "source_files": {"course/chapters/14.md": sha(chapter_path), "tiny_perceptron/modern.py": sha(ROOT / "tiny_perceptron/modern.py"), "course/figures/rewrite-14-8-position-interpolation.svg": sha(svg_path)},
    "lesson_rope_definition_sha256": hashlib.sha256(ast.get_source_segment(rope_block, definition).encode()).hexdigest(),
    "coordinates": coordinates,
    "relative_distance": {"before": 15, "after": coordinates[-1] - coordinates[0]},
    "adjacent_distance": {"before": 1, "after": 0.5},
    "hypothetical_degrees_per_position": 60,
    "position_two_degrees": {"before": 2 * 60, "after": coordinates[2] * 60},
    "exercise": {"count": len(exercise), "position_five": exercise[5], "position_last": exercise[23]},
    "rope": {"shape": list(actual.shape), "max_absolute_error_against_scalar_formula": max_error, "max_squared_norm_error": norm_error, "lesson_and_library_equal": torch.equal(actual, lesson_actual), "input_unchanged": torch.equal(x, before)},
    "figure_cards": cards,
    "downward_arrows": len(arrows),
    "result": "全部核對通過"
}, ensure_ascii=False, indent=2))
