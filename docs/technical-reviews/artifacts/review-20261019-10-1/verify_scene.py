"""核對合成圖的座標、廣播、軸順序與 SVG 像素。"""

import ast
import hashlib
import inspect
import json
import platform
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.multimodal import scene

torch.set_default_device("cpu")
torch.set_num_threads(1)
frozen_path = ROOT / "docs/course-review-20261019/freeze-01/sources/10.1.md"
frozen = frozen_path.read_bytes()
chapter = (ROOT / "course/chapters/10.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", chapter))
start = next(i for i, item in enumerate(headers) if item[0].startswith(b"## 10.1 "))
current = chapter[headers[start].start():headers[start + 1].start()]
assert frozen == current
fences = re.findall(r"```python\n(.*?)```", frozen.decode(), re.S)
local_namespace = {"torch": torch}
exec(compile(fences[1], str(frozen_path), "exec"), local_namespace)
inline_scene = local_namespace["scene"]
same_implementation = ast.dump(ast.parse(fences[1]), include_attributes=False) == ast.dump(
    ast.parse(inspect.getsource(scene)), include_attributes=False
)
assert same_implementation

image = scene("red", "square")
circle = scene("red", "circle")
assert torch.equal(image, scene())
assert torch.equal(image, inline_scene())
assert torch.equal(circle, inline_scene(shape="circle"))
before = image.clone()
center_list = image[:, 8, 8].tolist()
assert torch.equal(before, image)
batch = image.unsqueeze(0)
assert batch.numel() == image.numel()
assert batch.untyped_storage().data_ptr() == image.untyped_storage().data_ptr()
assert image.shape == circle.shape == (3, 16, 16)
assert image.dtype == torch.float32
assert image.min().item() == 0.0 and image.max().item() == 1.0
zero_image = torch.zeros(3, 16, 16)
assert zero_image.shape == (3, 16, 16) and not zero_image.count_nonzero().item()
color_checks = {}
for color, expected_rgb in {"red": [1.0, 0.0, 0.0], "green": [0.0, 1.0, 0.0], "blue": [0.0, 0.0, 1.0]}.items():
    actual_rgb = scene(color=color)[:, 8, 8].tolist()
    assert actual_rgb == expected_rgb
    color_checks[color] = actual_rgb
assert center_list == circle[:, 8, 8].tolist() == [1.0, 0.0, 0.0]
assert image[:, 0, 0].tolist() == [0.0, 0.0, 0.0]
assert image[:, 4, 4].tolist() == [1.0, 0.0, 0.0]
assert circle[:, 4, 4].tolist() == [0.0, 0.0, 0.0]

yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
assert torch.equal(circle[0] != 0, (xx - 8).square() + (yy - 8).square() <= 16)
assert not circle[1:].count_nonzero().item()
assert torch.equal(yy[:, 0], torch.arange(16))
assert torch.equal(xx[0, :], torch.arange(16))
assert torch.equal(yy, yy[:, :1].expand(16, 16))
assert torch.equal(xx, xx[:1, :].expand(16, 16))
horizontal = (xx - 8).abs() <= 4
region = horizontal & ((yy - 8).abs() <= 4)
red = torch.tensor([1.0, 0.0, 0.0])[:, None]
assert region.dtype == torch.bool and region.sum().item() == 81
assert image[:, region].shape == (3, 81) and red.shape == (3, 1)
assert torch.equal(image[:, region], red.expand(3, 81))
assert not image[1:].count_nonzero().item()
assert not image[:, ~region].count_nonzero().item()
square_rows = torch.where(region.any(dim=1))[0].tolist()
square_columns = torch.where(region.any(dim=0))[0].tolist()
assert square_rows == square_columns == list(range(4, 13))
for offset in (-2, 2):
    shifted = scene(offset=offset)
    assert torch.equal(shifted[0] != 0, (xx - (8 + offset)).abs().le(4) & (yy - 8).abs().le(4))
try:
    scene(shape="triangle")
except ValueError as error:
    invalid_shape = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("未拒絕不支援的形狀")

hwc = torch.arange(16 * 16 * 3).reshape(16, 16, 3)
chw = hwc.permute(2, 0, 1)
assert chw[:, 4, 7].tolist() == hwc[4, 7, :].tolist()
assert not torch.equal(chw, hwc.reshape(3, 16, 16))
normalized = torch.tensor([0, 128, 255], dtype=torch.uint8).float() / 255
assert normalized[0].item() == 0.0 and normalized[-1].item() == 1.0
chapter_scene_calls = []
for fence in re.findall(r"```python\n(.*?)```", chapter.decode(), re.S):
    for node in ast.walk(ast.parse(fence)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "scene":
            assert len(node.args) < 4 and all(item.arg != "offset" for item in node.keywords)
            chapter_scene_calls.append(ast.unparse(node))

svg_path = ROOT / "course/figures/rewrite-10-pixels.svg"
svg = ET.parse(svg_path).getroot()
pixels = [node for node in svg if node.tag.endswith("rect") and node.get("fill") in ("#000000", "#ff0000")]
xs = sorted({int(node.get("x")) for node in pixels})
ys = sorted({int(node.get("y")) for node in pixels})
assert len(pixels) == 256 and len(xs) == len(ys) == 16
assert all(xs[i + 1] - xs[i] == 14 and ys[i + 1] - ys[i] == 14 for i in range(15))
assert all(node.get("width") == node.get("height") == "14" for node in pixels)
seen = set()
red_coordinates = []
for node in pixels:
    row, column = ys.index(int(node.get("y"))), xs.index(int(node.get("x")))
    assert (row, column) not in seen
    seen.add((row, column))
    is_red = node.get("fill") == "#ff0000"
    assert is_red == bool(image[0, row, column].item())
    if is_red:
        red_coordinates.append([row, column])
highlights = [
    [ys.index(int(node.get("y"))), xs.index(int(node.get("x")))]
    for node in svg if node.tag.endswith("rect") and node.get("fill") == "none"
]
assert highlights == [[0, 0], [8, 8]]
labels = [node.text for node in svg if node.tag.endswith("text")]
png_path = ROOT / "outputs/course-review-20261019/figure-renders-04/1d95e51a6c2c5aaef0b08253f549c2c9d88bcc9425ebea0903d7920572efcc3b.png"
receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
figure_receipt = next(item for item in receipt["records"] if item["page_id"] == "10.1")
assert hashlib.sha256(png_path.read_bytes()).hexdigest() == figure_receipt["png"]["sha256"]

result = {
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "device": str(image.device), "platform": platform.platform()},
    "source_sha256": hashlib.sha256(frozen).hexdigest(),
    "source_matches_freeze": current == frozen,
    "inline_scene_matches_repository_ast": same_implementation,
    "default_equals_explicit_square": True,
    "docstring": scene.__doc__,
    "color_dictionary_results": color_checks,
    "initial_zeros": {"shape": list(zero_image.shape), "nonzero_count": int(zero_image.count_nonzero())},
    "square": {"shape": list(image.shape), "dtype": str(image.dtype), "center_rgb": center_list, "upper_left_rgb": image[:, 0, 0].tolist(), "corner_rgb": image[:, 4, 4].tolist(), "rows": square_rows, "columns": square_columns, "red_pixel_count": int(region.sum()), "green_blue_nonzero": int(image[1:].count_nonzero()), "red_value_sum": image[0].sum().item()},
    "batch": {"shape": list(batch.shape), "numel_before": image.numel(), "numel_after": batch.numel(), "same_storage": True},
    "tolist_preserves_image": True,
    "meshgrid": {"shape": list(yy.shape), "yy_first_column": yy[:, 0].tolist(), "xx_first_row": xx[0, :].tolist(), "region_dtype": str(region.dtype)},
    "broadcast": {"selection_shape": list(image[:, region].shape), "rhs_shape": list(red.shape), "all_selected_match_expansion": True},
    "circle": {"shape": list(circle.shape), "center_rgb": circle[:, 8, 8].tolist(), "corner_rgb": circle[:, 4, 4].tolist(), "corner_distance_squared": (4 - 8) ** 2 + (4 - 8) ** 2, "radius_squared": (16 // 4) ** 2, "red_pixel_count": int(circle[0].count_nonzero()), "all_pixels_match_distance_rule": True},
    "offset_checks": {"tested_offsets": [-2, 2], "match_coordinate_rule": True},
    "chapter_scene_calls": {"calls": chapter_scene_calls, "all_use_default_offset": True},
    "invalid_shape": invalid_shape,
    "axis_conversion": {"hwc_shape": list(hwc.shape), "chw_shape": list(chw.shape), "pixel_values_preserved": True, "plain_reshape_is_different": True},
    "uint8_to_float_scale": {"input": [0, 128, 255], "result": normalized.tolist(), "scene_divided_again_max": (image / 255).max().item()},
    "figure": {"svg_sha256": hashlib.sha256(svg_path.read_bytes()).hexdigest(), "pixel_count": len(pixels), "red_pixel_count": len(red_coordinates), "all_pixels_match_scene": True, "highlight_coordinates": highlights, "labels": labels, "inspected_png": str(png_path.relative_to(ROOT)), "png_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(), "png_matches_receipt": True},
    "result": "全部核對成立",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
