import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.modalities import _media, _sequence, _vision_records
from scripts.course_experiments.run import experiment_spec
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import masked_loss
from tiny_perceptron.multimodal import expand_modalities, scene


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def image_sha(image):
    return hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest()


torch.set_num_threads(1)
torch.manual_seed(42)
frozen_path = "docs/course-review-20261019/freeze-01/sources/11.4.md"
frozen = (ROOT / frozen_path).read_text()
code = re.search(r"```python\n(.*?)```", frozen, re.S)[1]
initial = io.StringIO()
namespace = {}
with contextlib.redirect_stdout(initial):
    exec(compile(code, frozen_path, "exec"), namespace)
records = namespace["records"]
assert len(records) == 2
assert all(isinstance(row["image"], str) for row in records)
exercise_records = records + [
    {"image": "red_circle", "question": "什麼顏色和形狀？", "answer": "紅色圓形"}
]
exercise = io.StringIO()
with contextlib.redirect_stdout(exercise):
    for row in exercise_records:
        print(row["image"], row["question"], "→", row["answer"])
    print("同圖", exercise_records[0]["image"] == exercise_records[1]["image"])
    print("答案不同", exercise_records[0]["answer"] != exercise_records[1]["answer"])
assert len(exercise.getvalue().splitlines()) == len(initial.getvalue().splitlines()) + 1
assert exercise.getvalue().splitlines()[-2:] == initial.getvalue().splitlines()[-2:]
assert len(exercise_records[-1]["answer"]) == 4

report_path = "docs/course-experiments/results/vqa.json"
published = json.loads((ROOT / report_path).read_bytes())
splits = _vision_records(("shape?", "color?"))
ctx = SimpleNamespace(device="cpu", seed=42)
tok = ByteTokenizer()
embedding = torch.nn.Embedding(tok.vocab_size, 8)
families = {}
pixels = {}
split_results = {}
mask_results = []
for split, rows in splits.items():
    stored = published["results"]["data"]["splits"][split]
    assert rows == stored["records"]
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert digest == stored["sha256"]
    assert len(rows) == stored["count"]
    grouped = defaultdict(list)
    images = defaultdict(set)
    for row in rows:
        grouped[row["family"]].append(row)
        image, audio = _media(row, ctx)
        assert audio is None and tuple(image.shape) == (3, 16, 16)
        images[row["family"]].add(image_sha(image))
        ids, labels, count = _sequence(row, ctx)
        prefix = _sequence(row, ctx, answer=False)[0]
        tail = tok.encode(row["answer"]) + [tok.eos_id]
        assert labels[:len(prefix)].tolist() == [-100] * len(prefix)
        assert labels[len(prefix):].tolist() == tail
        assert count == len(tail)
        features = {tok.image_id: torch.zeros(16, 8)}
        _, targets = expand_modalities(ids, labels, embedding, features, {tok.image_id})
        assert targets[targets != -100].tolist() == tail
        assert int((targets != -100).sum()) == count
        logits = torch.zeros(1, targets.shape[1], tok.vocab_size)
        loss = masked_loss(logits, targets)
        altered = logits.clone()
        altered[targets == -100] = torch.arange(tok.vocab_size, dtype=torch.float32)
        assert torch.equal(loss, masked_loss(altered, targets))
        assert abs(float(loss) - math.log(tok.vocab_size)) < 1e-6
        mask_results.append({
            "split": split, "family": row["family"], "question": row["question"],
            "prefix_positions": len(prefix), "answer_and_eos_targets": count,
            "expanded_ignored_positions": int((targets == -100).sum()),
            "uniform_logit_loss": float(loss), "ignored_logits_do_not_change_loss": True,
        })
    for family, members in grouped.items():
        assert len(members) == 2
        assert {row["question"] for row in members} == {"shape?", "color?"}
        assert len(images[family]) == 1
    families[split] = set(grouped)
    pixels[split] = set().union(*images.values())
    split_results[split] = {
        "records": len(rows), "families": len(grouped), "distinct_images": len(pixels[split]),
        "records_sha256": digest, "matches_published_records": True,
        "questions_per_family": dict(Counter(len(v) for v in grouped.values())),
        "all_family_members_have_identical_pixels": True,
    }
intersections = {}
for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
    assert not families[left] & families[right]
    assert not pixels[left] & pixels[right]
    intersections[left + "/" + right] = {
        "families": len(families[left] & families[right]),
        "pixel_hashes": len(pixels[left] & pixels[right]),
    }

svg_path = "course/figures/rewrite-11-same-image-questions.svg"
svg = ET.fromstring((ROOT / svg_path).read_bytes())
ns = {"s": "http://www.w3.org/2000/svg"}
cells = [node for node in svg.findall("s:rect", ns) if node.get("width") == "13"]
assert len(cells) == 256
image = scene("red", "circle")
for cell in cells:
    x = (int(cell.get("x")) - 210) // 13
    y = (int(cell.get("y")) - 82) // 13
    fill = "#ff0000" if float(image[0, y, x]) else "#000000"
    assert cell.get("fill") == fill
render_path = "outputs/course-review-20261019/figure-renders-04/4195cd648e5c2707e5acad33acdb0a93bdfa543d2ea1b8e39357d8ab48ad36be.png"
receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
render_receipt = next(row for row in receipt["records"] if row["page_id"] == "11.4")
assert sha(render_path) == render_receipt["png"]["sha256"]
assert sha(svg_path) == render_receipt["frozen_svg"]["sha256"]

result_inventory = {}
for variant, result in published["results"]["variants"].items():
    groups = {}
    for split in ("validation", "test"):
        evaluation = result[split]
        samples = evaluation["samples"]
        assert len(samples) == len(splits[split]) == evaluation["examples"]
        for sample, row in zip(samples, splits[split], strict=True):
            assert (sample["family"], sample["question"], sample["target"]) == (
                row["family"], row["question"], row["answer"]
            )
        assert sum(s["exact_match"] for s in samples) == evaluation["correct"]
        groups[split] = {"examples": len(samples), "complete_question_records": True}
    history = result["training"]["history"]
    assert len(history) == result["training"]["steps"]
    result_inventory[variant] = {"update_steps_recorded": len(history), "evaluations": groups}

spec = experiment_spec("vqa")
assert spec["function"] == "run_vqa"
red, blue = scene("red", "circle"), scene("blue", "circle")
assert not torch.equal(red, blue)
assert torch.equal(red[0] > 0, blue[2] > 0)
paths = [frozen_path, svg_path, report_path, "docs/course-experiments/README.md",
         "docs/course-experiments/plan.json", "scripts/course_experiments/modalities.py",
         "scripts/course_experiments/run.py", "tiny_perceptron/multimodal.py",
         "tiny_perceptron/model.py", "tiny_perceptron/data.py"]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-11-4/verify.py",
    "result": "全部斷言通過；未執行訓練。",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": "cpu", "platform": platform.platform()},
    "source_sha256": {path: sha(path) for path in paths},
    "original_stdout": initial.getvalue(), "exercise_stdout": exercise.getvalue(),
    "combined_answer_characters": len(exercise_records[-1]["answer"]),
    "record_image_type": type(records[0]["image"]).__name__,
    "splits": split_results, "cross_split_intersections": intersections,
    "answer_mask_checks": mask_results,
    "figure": {"matching_scene_pixels": len(cells), "red_pixels": int((image[0] > 0).sum()),
               "text": [node.text for node in svg.findall("s:text", ns)],
               "render_receipt": render_receipt, "actual_png_sha256": sha(render_path)},
    "fixed_question_red_blue": {"same_shape_region": True, "different_pixels": True,
                                "expected_answers": ["red", "blue"]},
    "question_only_counterexample": {"color_question_prediction": "紅", "shape_question_prediction": "圓形",
                                    "correct_on_original_records_without_image": 2, "records": 2},
    "published_result_inventory": result_inventory,
    "experiment_entry": {"id": spec["id"], "module": spec["module"], "function": spec["function"]},
    "limits": "僅核對短程式、合成資料、遮罩與已保存紀錄；未重跑訓練或驗證模型品質。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
