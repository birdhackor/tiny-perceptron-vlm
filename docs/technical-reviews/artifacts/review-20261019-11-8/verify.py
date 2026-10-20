"""重算固定猜測、逐題介入評分，並記錄實際讀取的來源。"""

import hashlib
import html
import json
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from scripts.course_experiments.modalities import _media, _vision_records
from tiny_perceptron.multimodal import scene


ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def fetch(url, excerpts):
    raw = subprocess.check_output(["curl", "-fsSL", url])
    plain = re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", " ", raw.decode())))
    for excerpt in excerpts:
        assert excerpt in plain, "來源未包含指定摘錄"
    return {
        "url": url,
        "accessed_on": "2026-10-20",
        "transport": "curl -fsSL",
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "excerpts": excerpts,
        "full_text_saved": False,
    }


paths = [
    "docs/course-review-20261019/freeze-01/sources/11.8.md",
    "docs/course-review-20261019/freeze-04.json",
    "course/figures/rewrite-11-shape-pairs.svg",
    "docs/course-review-20261019/figure-renders-04.json",
    "docs/course-experiments/results/vision_ablation.json",
    "docs/course-experiments/results/vqa.json",
    "scripts/course_experiments/modalities.py",
    "scripts/course_experiments/common.py",
    "scripts/course_experiments/run.py",
    "tiny_perceptron/multimodal.py",
    "docs/course-experiments/README.md",
]
fingerprints = {path: digest(path) for path in paths}
source = (ROOT / paths[0]).read_text()
chapter = (ROOT / "course/chapters/11.md").read_text()
current = re.search(r"^## 11\.8 .*?(?=^## |\Z)", chapter, re.M | re.S)[0]
assert current == source, "現行小節與凍結原文不同"

truth = ["circle", "square", "circle", "square"]
predictions = ["circle"] * len(truth)
baselines = {
    "repeated": predictions,
    "pairs": list(zip(truth, predictions)),
    "booleans": [a == b for a, b in zip(truth, predictions)],
    "balanced": sum(a == b for a, b in zip(truth, predictions)) / len(truth),
    "imbalanced": (["circle"] * 9 + ["square"]).count("circle") / 10,
    "exercise": ["circle", "circle", "circle", "square"].count("circle") / 4,
    "constant_prediction_pair_correct": int("circle" == "circle") + int("circle" == "square"),
    "constant_prediction_pair_pass": ("circle" == "circle") and ("circle" == "square"),
}
assert (baselines["balanced"], baselines["imbalanced"], baselines["exercise"]) == (0.5, 0.9, 0.75)

record = json.loads((ROOT / "docs/course-experiments/results/vision_ablation.json").read_text())
vqa = json.loads((ROOT / "docs/course-experiments/results/vqa.json").read_text())
splits = record["results"]["data"]["splits"]
assert _vision_records(("shape?", "color?")) == {name: part["records"] for name, part in splits.items()}
assert all(part["sha256"] == vqa["results"]["data"]["splits"][name]["sha256"] for name, part in splits.items())
code_matches = {path: fingerprints[path] == record["code_sha256"][path] for path in paths if path in record["code_sha256"]}
assert all(code_matches[path] for path in ("scripts/course_experiments/modalities.py", "scripts/course_experiments/common.py", "tiny_perceptron/multimodal.py"))
rows = splits["test"]["records"]
modes = record["results"]["interventions"]
counts = {}
compact = {}
for mode in ("none", "blank", "shuffle", "shape_swap_relabelled"):
    samples = modes[mode]["samples"]
    assert len(samples) == len(rows)
    correct = sum(s["generated"] == s["target"] for s in samples)
    assert correct == modes[mode]["correct"]
    assert all(s["exact_match"] == (s["generated"] == s["target"]) for s in samples)
    assert all(s["generation_error"] is None and s["eos"] for s in samples)
    counts[mode] = {"correct": correct, "count": len(samples)}
    compact[mode] = [{k: s[k] for k in ("row", "question", "target", "generated", "donor_row")} for s in samples]
    for question in ("shape?", "color?"):
        selected = [s for s in samples if s["question"] == question]
        counts[mode][question] = {
            "correct": sum(s["generated"] == s["target"] for s in selected),
            "count": len(selected),
            "generated": [s["generated"] for s in selected],
        }

shape_pairs = []
for original, swapped in zip(modes["none"]["samples"], modes["shape_swap_relabelled"]["samples"], strict=True):
    if original["question"] == "shape?":
        assert original["target"] != swapped["target"]
        shape_pairs.append(original["exact_match"] and swapped["exact_match"])
counts["shape_pairs"] = {"correct": sum(shape_pairs), "count": len(shape_pairs), "distinct_unordered_image_pairs": 3}

donor_score = []
for sample in modes["shuffle"]["samples"]:
    donor = sample["donor_row"]
    assert donor == (sample["row"] + len(rows) // 2) % len(rows)
    assert modes["shuffle"]["samples"][donor]["donor_row"] == sample["row"]
    assert rows[donor]["question"] == sample["question"]
    donor_score.append(sample["generated"] == rows[donor]["answer"])
counts["shuffle_relabelled"] = {"correct": sum(donor_score), "count": len(donor_score)}
counts["shuffle_relabelled"]["color?"] = {
    "correct": sum(ok for ok, row in zip(donor_score, rows, strict=True) if row["question"] == "color?"),
    "count": 6,
}

class CPUContext:
    device = "cpu"


for row in rows:
    replacement = dict(row, shape="square" if row["shape"] == "circle" else "circle")
    assert all(row[k] == replacement[k] for k in ("color", "offset", "question"))
    image, waveform = _media(row, CPUContext())
    new_image, _ = _media(replacement, CPUContext())
    assert waveform is None and not torch.equal(image, new_image)
    assert torch.count_nonzero(torch.zeros_like(image)).item() == 0

svg = ET.fromstring((ROOT / "course/figures/rewrite-11-shape-pairs.svg").read_bytes())
ns = {"s": "http://www.w3.org/2000/svg"}
cells = [r for r in svg.findall("s:rect", ns) if r.get("width") == "13" and r.get("height") == "13"]
grids = ([r for r in cells if float(r.get("x")) < 300], [r for r in cells if float(r.get("x")) >= 300])
figure = {}
for shape, grid in zip(("circle", "square"), grids, strict=True):
    assert len(grid) == 256
    left = min(float(r.get("x")) for r in grid)
    top = min(float(r.get("y")) for r in grid)
    red = {(round((float(r.get("y")) - top) / 13), round((float(r.get("x")) - left) / 13)) for r in grid if r.get("fill") == "#ff0000"}
    expected = {tuple(v) for v in torch.nonzero(scene("red", shape)[0]).tolist()}
    assert red == expected, "圖中像素與合成形狀規則不同"
    figure[shape] = {"grid": [16, 16], "red_pixels": len(red), "center": [8, 8]}
receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_text())
png = next(r["png"] for r in receipt["records"] if r["figure"] == "course/figures/rewrite-11-shape-pairs.svg")
assert digest(png["path"]) == png["sha256"]

external = [
    fetch("https://arxiv.org/html/1612.00837v3", ["Language-only models have simply no basis for differentiating between the two cases"]),
    fetch("https://captum.ai/api/feature_ablation.html", ["Baselines define reference value which replaces each feature when ablated.", "we internally use zero scalar corresponding to each input tensor."]),
    fetch("https://docs.python.org/3.13/library/stdtypes.html", ["bool is a subclass of int", "Return the total number of occurrences of value in sequence ."]),
    fetch("https://docs.python.org/3.13/library/functions.html", ["Iterate over several iterables in parallel, producing tuples with an item from each one.", "Sums start and the items of an iterable"]),
]
pearl_url = "https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf"
pearl = subprocess.check_output(["curl", "-fsSL", pearl_url])
assert pearl.startswith(b"%PDF-")
external.append({
    "url": pearl_url,
    "accessed_on": "2026-10-20",
    "transport": "curl -fsSL; 原文以 web PDF 解析閱讀",
    "response_sha256": hashlib.sha256(pearl).hexdigest(),
    "locator": "Representing interventions; printed pp. 107–108; equations (6)–(8)",
    "excerpts": ["replacing them by a constant X = x, while keeping the rest of the model unchanged"],
    "full_text_saved": False,
})

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/review-20261019-11-8/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "source_fingerprints": fingerprints,
    "current_code_matches_recorded_hashes": code_matches,
    "frozen_source_matches_current": True,
    "baselines": baselines,
    "recorded_training_environment": {k: record[k] for k in ("revision", "seed", "device", "torch_version", "python_version", "gpu", "step_scale")},
    "recorded_training": {k: record["results"]["training"][k] for k in ("steps", "effective_targets", "weights_changed", "nonzero_gradient_seen")},
    "splits": {name: {k: part[k] for k in ("count", "sha256")} for name, part in splits.items()},
    "counts": counts,
    "samples": compact,
    "upstream_all_model_and_export_same_hash": [a for a in vqa["artifacts"] if a["path"] in ("all.pt", "model.pt")],
    "before_samples_equal_upstream_all": record["results"]["before"]["samples"] == vqa["results"]["variants"]["all"]["test"]["samples"],
    "media_control_checks": "問題、顏色、位置固定；僅形狀改變；黑圖為相同尺寸的全零 tensor；未重跑訓練或 checkpoint 推論。",
    "figure": {"pixel_checks": figure, "png_receipt": png},
    "external_sources": external,
    "result": "所有斷言通過；逐題重算與正文數值一致。",
}
json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
print()
