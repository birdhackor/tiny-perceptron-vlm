"""Independent CPU checks of authored examples; no OCR model is run."""

import ast
import hashlib
import json
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Load only the two original Unicode comparison functions identified by AST.
source = ROOT / "tiny_perceptron/natural_concepts.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"edit_distance", "text_error_report"}]
assert len(selected) == 2
ns = {}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), ns)

# Every input pixel has its own value: this tests coordinate selection, not OCR.
image = Image.new("L", (4, 2))
image.putdata(range(8))
left = image.crop((0, 0, 2, 2))
right = image.crop((2, 0, 4, 2))
assert list(left.getdata()) == [0, 1, 4, 5]
assert list(right.getdata()) == [2, 3, 6, 7]
assert right.size == (2, 2)
assert set(right.getdata()).isdisjoint(left.getdata())

svg = ROOT / "course/figures/new-11.18-two-regions.svg"
root = ET.parse(svg).getroot()
texts = [e.text for e in root.iter() if e.tag.endswith("}text")]
assert "入口" in texts and "出口" in texts
rects = [e.attrib for e in root.iter() if e.tag.endswith("}rect")]
chosen = next(r for r in rects if r.get("stroke") == "#168252")
assert [int(chosen[k]) for k in ("x", "y", "width", "height")] == [347, 120, 226, 130]
assert 'data:' not in svg.read_text() and '@font-face' not in svg.read_text()

# This contract is the course's prospective alphabet, not a trained tokenizer.
alphabet = set("大小上下左右開關入出人口")
assert len(alphabet) == 12
assert set("入口出口開關").issubset(alphabet)
cases = [
    {"left": "入口", "right": "出口", "selected": "right", "expected": "出口"},
    {"left": "入口", "right": "出口", "selected": "left", "expected": "入口"},
    {"left": "出口", "right": "入口", "selected": "right", "expected": "入口"},
    {"left": "入口", "right": "開關", "selected": "right", "expected": "開關"},
]
for case in cases:
    assert case[case["selected"]] == case["expected"]

reports = {p: ns["text_error_report"]("出口", p) for p in ["出口", "入口出口", "口出", "出入", "入口"]}
assert reports["出口"]["exact"] is True and reports["出口"]["edits"] == 0
assert reports["入口出口"]["exact"] is False and reports["入口出口"]["edits"] == 2
assert reports["口出"]["exact"] is False and reports["口出"]["edits"] == 2
assert reports["出入"]["edits"] == 1
assert reports["入口"]["edits"] == 1
for report in reports.values():
    assert report["reference_characters"] == 2
    assert report["cer"] == report["edits"] / 2

result = {
    "scope": "Pixel/target/Unicode contracts only. No trained model, dataset download, parameter update or OCR inference.",
    "coordinate_check": {"source_size": image.size, "left_pixels": list(left.getdata()), "right_pixels": list(right.getdata()), "right_size": right.size},
    "svg_selected_rectangle": chosen,
    "prospective_alphabet_count": len(alphabet),
    "target_contract_cases": cases,
    "unicode_comparisons": reports,
    "source_sha256": {"tiny_perceptron/natural_concepts.py": sha(source), "course/figures/new-11.18-two-regions.svg": sha(svg)},
    "environment": {"python": sys.version, "python_executable": sys.executable, "Pillow": PIL.__version__, "platform": platform.platform(), "device": "cpu"},
    "result": "all assertions passed",
}
(OUT / "bounded-checks-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
