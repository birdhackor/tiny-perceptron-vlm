"""Bounded CPU checks and recomputation of existing photograph records; no inference."""
from pathlib import Path
import base64
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.natural_concepts import picture_order_report, swapped_picture

def read(path):
    return json.loads((ROOT / path).read_text())

def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

chapter = (ROOT / "course/chapters/20.md").read_text()
section = chapter[chapter.index("## 20.9 "):chapter.index("## 20.10 ")]
code = re.search(r"```python\n(.*?)```", section, re.S)[1]
stream = io.StringIO()
with redirect_stdout(stream):
    exec(compile(code, "course/chapters/20.md#20.9", "exec"), {})
expected = "卡 0 物件吻合 True\n活動符合本題 True\n卡 1 物件吻合 True\n活動符合本題 False\n"
assert stream.getvalue() == expected
assert {"人", "腳踏車"} == set(["腳踏車", "人"])
order = picture_order_report()
assert order == {"different_pixels": 128, "same_4_by_4_summary": True, "same_pixel_sequence": False}
first, second = swapped_picture()
color_positions = int((first != second).any(dim=0).sum())
assert color_positions == 64
# Explicitly hold text and history fixed while changing only the visual input.
question = "紅條在藍條的哪一邊？"
history = []
pair = [{"question": question, "history": history, "pixel_sha256":
         hashlib.sha256(image.numpy().tobytes()).hexdigest()}
        for image in (first, second)]
assert pair[0]["question"] == pair[1]["question"] and pair[0]["history"] == pair[1]["history"]
assert pair[0]["pixel_sha256"] != pair[1]["pixel_sha256"]
manifest_path = "docs/natural-assistant/v4/manifest.json"
manifest = read(manifest_path)
photos = [r for r in manifest["rows"] if r["task"] == "scene"]
split_sets = {}
for split in ("train", "validation", "test"):
    rows = [r for r in photos if r["split"] == split]
    split_sets[split] = {"image": {r["image"] for r in rows},
                         "family": {r["family"] for r in rows},
                         "sha": {r["source"]["image_sha256"] for r in rows}}
for a,b in (("train","validation"),("train","test"),("validation","test")):
    for field in ("image", "family", "sha"):
        assert not split_sets[a][field] & split_sets[b][field], (a,b,field)
assert len(split_sets["test"]["image"]) == 42
test = [r for r in photos if r["split"] == "test"]
assert Counter(r["references"]["qa_type"] for r in test)["scene"] == 42
assert len(test) == 126
for r in test:
    assert r["references"]["kind"] == "manual" and r["references"]["rubric"]
scores_path = "docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/scored/scores.json"
subsets_path = "docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/descriptive-subsets.json"
scores, subsets = read(scores_path), read(subsets_path)
assert subsets["scores_sha256"] == sha(scores_path)
assert subsets["manifest_sha256"] == sha(manifest_path)
decisions = {r["case_id"]: r for r in scores["case_decisions"]}
summary = Counter(); facts = Counter(); recomputed = {}
for row in test:
    d = decisions["scene:" + row["id"]]
    assert d["passed"] == (d["decoder_complete"] and d["rubric_passed"])
    category = row["references"]["qa_type"]
    (summary if category == "scene" else facts)["correct"] += int(d["passed"])
    (summary if category == "scene" else facts)["denominator"] += 1
for category in sorted({r["references"]["qa_type"] for r in test if r["references"]["qa_type"] != "scene"}):
    ids = ["scene:"+r["id"] for r in test if r["references"]["qa_type"] == category]
    observed = {"correct": sum(decisions[x]["passed"] for x in ids), "denominator": len(ids)}
    recorded = subsets["photo_fact_subsets"][category]
    assert {x["case_id"] for x in recorded["cases"]} == set(ids)
    assert all(x["passed"] == decisions[x["case_id"]]["passed"] for x in recorded["cases"])
    assert observed == {k: recorded[k] for k in ("correct", "denominator")}
    recomputed[category] = observed
assert dict(summary) == {"correct": 25, "denominator": 42}
assert dict(facts) == {"correct": 58, "denominator": 84}
cat_path = "course/figures/natural-v4-training-cat.svg"
embedded = base64.b64decode(re.search(r"data:image/[^;]+;base64,([^\"]+)", (ROOT/cat_path).read_text())[1])
original = (ROOT/"outputs/natural-v4/factual-research/20.9/cat-original.jpg").read_bytes()
assert embedded == original
cat = [r for r in photos if r["id"].startswith("vision-v4:docci/train_01827/")]
assert len(cat)==2 and all(r["split"]=="train" for r in cat)
caption_row = read("outputs/natural-v4/factual-research/20.9/cat-original-description.jsonl")
assert all(r["references"]["official_description"] == caption_row["description"] for r in cat)
print(json.dumps({"environment": {"python": platform.python_version(), "torch": torch.__version__,
                  "device": "cpu", "cuda_available": str(torch.cuda.is_available())},
    "section_sha256": hashlib.sha256(section.encode()).hexdigest(),
    "card_stdout": stream.getvalue(), "card_order_invariant": True,
    "prerequisite_picture_order": order, "different_color_positions": color_positions,
    "controlled_input_pair": pair, "counterfactual_scope": "Input-only construction; no model response generated",
    "photo_split_counts": {s:{"rows": sum(r["split"]==s for r in photos),
                              "images":len(split_sets[s]["image"]), "families":len(split_sets[s]["family"])} for s in split_sets},
    "pairwise_split_intersections_image_family_sha": 0,
    "photo_summary":dict(summary), "photo_fact":dict(facts), "photo_fact_subsets":recomputed,
    "cat_embedded_bytes_equal_retrieved_original": True,
    "cat_sha256": hashlib.sha256(embedded).hexdigest(), "cat_caption_matches_original": True,
    "scope": "Bounded CPU example and direct audit/recomputation of fixed project records; no GPU training, model inference, or full benchmark replication."}, ensure_ascii=False,indent=2))
