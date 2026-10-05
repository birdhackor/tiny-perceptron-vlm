"""Bounded checks for this reviewer's corrected 10.5 source and new figure."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
ART = OUT.parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.multimodal import scene

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

source = OUT / "section.md"
figure = OUT / "shape-gradient.svg"
fence = OUT / "fence-1.py"
initial = ART / "initial-report.json"
old = json.loads(initial.read_bytes())  # Only this same reviewer's initial report.

reused = []
for item in old["artifacts"]:
    p = ROOT / item["path"]
    assert sha(p) == item["sha256"], item["id"]
    reused.append({"artifact_id": item["id"], "path": item["path"], "sha256": item["sha256"], "hash_unchanged": True})
assert sha(fence) == sha(ART / "original/fence-1.py")
for item in old["sources"]:
    if item["kind"] == "repository_code":
        assert sha(ROOT / item["path"]) == item["sha256"]
for name in ["section.md", "fence-1.py"]:
    assert (OUT / name).read_bytes() == (ROOT / "outputs/course-revision-20261005/phase4-10_5-independent-recheck" / name).read_bytes()
assert figure.read_bytes() == (ROOT / "course/figures/rewrite-10-shape-gradient.svg").read_bytes()

# The new paragraph requires no extra model score claim. Reinspect only the
# named raw fields that establish its six-class/width16/actual-update scope.
raw_path = ROOT / "docs/course-experiments/results/encoders.json"
assert sha(raw_path) == sha(ART / "inputs/docs/course-experiments/results/encoders.json")
data = json.loads(raw_path.read_bytes())
vision = data["results"]["vision"]
pointers = ["/revision", "/device", "/results/vision/config", "/results/vision/classes",
            "/results/vision/training/weights_changed", "/results/vision/training/steps",
            "/results/vision/training/trainable_parameters"]
selected = {"/revision": data["revision"], "/device": data["device"],
            "/results/vision/config": vision["config"], "/results/vision/classes": vision["classes"],
            "/results/vision/training/weights_changed": vision["training"]["weights_changed"],
            "/results/vision/training/steps": vision["training"]["steps"],
            "/results/vision/training/trainable_parameters": vision["training"]["trainable_parameters"]}
assert vision["config"] == {"width": 16, "image_size": 16, "patch_size": 4}
assert vision["classes"] == [color + " " + shape for color in ["red", "green", "blue"] for shape in ["circle", "square"]]
assert vision["training"]["weights_changed"] is True
assert vision["training"]["trainable_parameters"] == 1446

xml = ET.parse(figure).getroot()
ns = "{http://www.w3.org/2000/svg}"
rects = list(xml.iter(ns + "rect"))
matched = []
for x_base, color, shape in [(55, "red", "square"), (373, "blue", "circle")]:
    image = scene(color, shape)
    colored = 0
    count = 0
    for y in range(16):
        for x in range(16):
            candidates = [r for r in rects if r.get("x") == str(x_base + 13 * x) and r.get("y") == str(87 + 13 * y)
                          and r.get("width") == "13" and r.get("height") == "13"]
            assert len(candidates) == 1
            expected = "#" + "".join(f"{round(c * 255):02x}" for c in image[:, y, x].tolist())
            assert candidates[0].get("fill") == expected
            colored += int(expected != "#000000")
            count += 1
    matched.append({"sample": color + " " + shape, "matched_cells": count, "colored_cells": colored})
texts = [n.text for n in xml.iter(ns + "text")]
paths = [dict(n.attrib) for n in xml.iter(ns + "path") if n.get("stroke") is not None]
assert {"類別編號 0", "類別編號 1", "入口 → 平均 → 分類頭 → 分數", "交叉熵：分數對正確標籤", "梯度", "此處未做 optimizer.step()"} <= set(texts)
assert {p["d"] for p in paths if p["stroke"] == "#486885"} == {"M263,191 L315,191 L315,480", "M373,220 L335,220 L335,480", "M320,558 L320,632"}
assert {p["d"] for p in paths if p["stroke"] == "#a16f10"} == {"M159,423 L159,450 L32,450 L32,662 L55,662", "M477,423 L477,450 L608,450 L608,662 L585,662"}
assert {p["d"] for p in paths if p["stroke"] == "#b83b3b"} == {"M470,632 L470,558"}

result = {"source_sha256": sha(source), "figure_sha256": sha(figure), "fence_sha256": sha(fence),
          "fence_unchanged": True, "implementation_sources_unchanged": True,
          "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "cuda_build": str(torch.version.cuda)},
          "original_execution_reused": "Original unchanged fence and repository implementation have been hashed; this callback did not reexecute original model code or rerun historical inference/training.",
          "verified_reused_artifacts": reused,
          "historical_raw_json_sha256": sha(raw_path), "historical_inspection_pointers": pointers,
          "historical_selected_original_values": selected,
          "history_scope_check": "Actual raw fields confirm six color+shape classes, feature width16, and recorded weights_changed=True; current paragraph explicitly distinguishes width8 two-class gradient-only example.",
          "new_figure_sample_matches": matched, "new_figure_texts": texts, "new_figure_paths": paths,
          "scope": "Figure pixels/paths and unchanged execution inputs checked. Human visual inspection of fresh renders and original issue resolution are recorded separately. No original-summary or other-review content read."}
(OUT / "checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"source_sha256": result["source_sha256"], "figure_sha256": result["figure_sha256"],
                  "unchanged_fence": True, "verified_prior_artifacts": len(reused),
                  "new_figure_matched_pixels": sum(r["matched_cells"] for r in matched),
                  "historical_width": vision["config"]["width"], "historical_classes": len(vision["classes"]),
                  "historical_weights_changed": vision["training"]["weights_changed"], "exit": "passed"}, indent=2))
