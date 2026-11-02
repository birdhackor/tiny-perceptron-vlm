"""10.1 technical review: execute the frozen inline code and independent expectations."""

import ast
import contextlib
import hashlib
import inspect
import io
import json
import platform
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import torch

from tiny_perceptron.multimodal import scene as repository_scene


ROOT = Path(__file__).resolve().parents[4]
SNAPSHOT = ROOT / "docs/course-review-20261019/freeze-01/sources/10.1.md"
OUTPUT = Path(__file__).with_name("verification.json")
if OUTPUT.exists():
    raise SystemExit("Refuse to overwrite preserved verification.json")

raw = SNAPSHOT.read_text()
blocks = re.findall(r"```python\n(.*?)```", raw, re.S)
assert len(blocks) == 2
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    imported_namespace = {}
    exec(compile(blocks[0], str(SNAPSHOT) + ":block1", "exec"), imported_namespace)
expected_stdout = (
    "形狀 (3, 16, 16)\n"
    "中央RGB [1.0, 0.0, 0.0]\n"
    "左上RGB [0.0, 0.0, 0.0]\n"
    "加入筆數軸 (1, 3, 16, 16)\n"
)
assert printed.getvalue() == expected_stdout

# W.3 explicitly teaches import torch; provide that prerequisite to the excerpt.
inline_namespace = {"torch": torch}
exec(compile(blocks[1], str(SNAPSHOT) + ":block2", "exec"), inline_namespace)
inline_scene = inline_namespace["scene"]
inline_ast = ast.dump(ast.parse(blocks[1]).body[0], include_attributes=False)
repository_ast = ast.dump(ast.parse(inspect.getsource(repository_scene)).body[0], include_attributes=False)
assert inline_ast == repository_ast

# Independent integer-coordinate construction, not a call to the repository mask.
square_coordinates = [(r, c) for r in range(16) for c in range(16) if 4 <= r <= 12 and 4 <= c <= 12]
circle_coordinates = [(r, c) for r in range(16) for c in range(16) if (r - 8) ** 2 + (c - 8) ** 2 <= 16]
variant_checks = []
for color, values in {"red": [1.0, 0.0, 0.0], "green": [0.0, 1.0, 0.0], "blue": [0.0, 0.0, 1.0]}.items():
    for shape, coords in [("square", square_coordinates), ("circle", circle_coordinates)]:
        expected = torch.zeros(3, 16, 16)
        for r, c in coords:
            expected[:, r, c] = torch.tensor(values)
        image = inline_scene(color, shape)
        assert torch.equal(image, expected)
        assert torch.equal(image, repository_scene(color, shape))
        variant_checks.append({"color": color, "shape": shape, "colored_pixels": len(coords), "exact_match": True})

image = inline_scene("red", "square")
before = image.clone()
central = image[:, 8, 8].tolist()
central[0] = 99.0
assert torch.equal(image, before)
batch = image.unsqueeze(0)
assert batch.shape == (1, 3, 16, 16)
assert batch.numel() == image.numel() == 768
assert batch.data_ptr() == image.data_ptr()
assert torch.equal(inline_scene(), image)

yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
assert yy[4, 12].item() == 4 and xx[4, 12].item() == 12
region = (xx - 8).abs() <= 4
region &= (yy - 8).abs() <= 4
assert region.dtype == torch.bool and region.sum().item() == 81
rhs = torch.tensor([1.0, 0.0, 0.0])[:, None]
assert rhs.shape == (3, 1) and image[:, region].shape == (3, 81)
assert torch.equal(rhs, torch.tensor([1.0, 0.0, 0.0]).unsqueeze(1))
assert image.sum(dim=(1, 2)).tolist() == [81.0, 0.0, 0.0]

offset_checks = []
for offset in [-2, 0, 2]:
    shifted = inline_scene(offset=offset)
    coords = shifted[0].nonzero()
    observed_bounds = [coords[:, 0].min().item(), coords[:, 0].max().item(), coords[:, 1].min().item(), coords[:, 1].max().item()]
    assert observed_bounds == [4, 12, 4 + offset, 12 + offset]
    offset_checks.append({"offset": offset, "row_min_max_col_min_max": observed_bounds})

try:
    inline_scene(shape="triangle")
except ValueError as error:
    invalid_shape = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("Invalid shape did not raise ValueError")

circle = inline_scene("red", "circle")
assert image[:, 4, 4].tolist() == [1.0, 0.0, 0.0]
assert circle[:, 4, 4].tolist() == [0.0, 0.0, 0.0]
assert circle[:, 8, 8].tolist() == [1.0, 0.0, 0.0]
assert circle.shape == image.shape

hwc = image.permute(1, 2, 0).contiguous()
assert torch.equal(hwc.permute(2, 0, 1), image)
assert not torch.equal(hwc.reshape(3, 16, 16), image)
normalization = torch.tensor([0, 128, 255], dtype=torch.uint8).float() / 255
assert torch.allclose(normalization, torch.tensor([0.0, 128 / 255, 1.0]), atol=1e-7, rtol=0)
assert image.max().item() == 1.0

svg_path = ROOT / "course/figures/rewrite-10-pixels.svg"
svg = ET.parse(svg_path).getroot()
cells = [element for element in svg if element.tag.endswith("rect") and element.get("fill") in {"#000000", "#ff0000"}]
assert len(cells) == 256
svg_red = []
for cell in cells:
    assert cell.get("width") == cell.get("height") == "14"
    row = (int(cell.get("y")) - 82) // 14
    column = (int(cell.get("x")) - 64) // 14
    assert (int(cell.get("y")) - 82) % 14 == (int(cell.get("x")) - 64) % 14 == 0
    expected_color = "#ff0000" if (row, column) in square_coordinates else "#000000"
    assert cell.get("fill") == expected_color
    if cell.get("fill") == "#ff0000":
        svg_red.append((row, column))
assert sorted(svg_red) == square_coordinates
highlights = [element for element in svg if element.tag.endswith("rect") and element.get("stroke") == "#ffc857"]
assert [(int(e.get("y")) - 82, int(e.get("x")) - 64) for e in highlights] == [(0, 0), (112, 112)]

result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_10_1",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_10_1-v1/verify.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu", "default_dtype": str(torch.get_default_dtype())},
    "snapshot_sha256": hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
    "inline_block_sha256": [hashlib.sha256(block.encode()).hexdigest() for block in blocks],
    "inline_dependency": "The implementation excerpt is executed with torch imported, as taught in linked prerequisite W.3; no repository helper substitutes for this execution.",
    "stdout": printed.getvalue(),
    "inline_function_ast_matches_repository": True,
    "variants": variant_checks,
    "square": {"shape": list(image.shape), "dtype": str(image.dtype), "center": image[:, 8, 8].tolist(), "top_left": image[:, 0, 0].tolist(), "corner_4_4": image[:, 4, 4].tolist(), "rows_and_columns": "4..12 inclusive", "pixels": 81, "channel_sums": image.sum(dim=(1, 2)).tolist()},
    "batch": {"shape": list(batch.shape), "elements_before_and_after": [image.numel(), batch.numel()], "same_storage": True},
    "tolist_leaves_image_unchanged": True,
    "default_arguments_equal_explicit_default_call": True,
    "meshgrid": {"shape": list(yy.shape), "dtype": str(yy.dtype), "yy_4_12": yy[4, 12].item(), "xx_4_12": xx[4, 12].item(), "region_dtype": str(region.dtype), "selected_shape": list(image[:, region].shape), "color_rhs_shape": list(rhs.shape)},
    "offsets": offset_checks,
    "invalid_shape": invalid_shape,
    "exercise": {"circle_corner_4_4": circle[:, 4, 4].tolist(), "circle_center": circle[:, 8, 8].tolist(), "shape": list(circle.shape), "squared_corner_distance": 32, "radius_squared": 16, "circle_pixels": len(circle_coordinates)},
    "layout": {"hwc_shape": list(hwc.shape), "permute_roundtrip_exact": True, "reshape_is_not_axis_conversion": True},
    "uint8_divide_255": {"input": [0, 128, 255], "output": normalization.tolist(), "tolerance": "atol=1e-7, rtol=0"},
    "svg": {"sha256": hashlib.sha256(svg_path.read_bytes()).hexdigest(), "cells": len(cells), "red_cells": len(svg_red), "every_cell_matches_scene": True, "highlight_coordinates": [[0, 0], [8, 8]]},
    "training_performed": False,
    "result": "All assertions passed; deterministic CPU image generation only.",
}
OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": result["result"], "output": str(OUTPUT.relative_to(ROOT)), "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}, ensure_ascii=False))
