"""離線核對字形、資料家族、RGB 入口與既有實驗的統計。"""

import ast
import copy
import hashlib
import json
import platform
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import __version__ as pillow_version
from torch.nn import functional as F

from scripts.prepare_ocr import GLYPHS, draw_digits
from scripts.course_experiments.run import experiment_spec
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.multimodal import VisionEncoder


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    torch.manual_seed(0)
    source = ROOT / "docs/course-review-20261019/freeze-01/sources/11.12.md"
    body = source.read_text()
    code = body.split("```python\n", 1)[1].split("```", 1)[0]
    tree = ast.parse(code)
    glyphs = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)))
    images = {
        label: torch.tensor([[int(c) for c in row] for row in rows], dtype=torch.float32)
        for label, rows in glyphs.items()
    }
    assert all(tuple(image.shape) == (5, 3) for image in images.values()), "字形形狀不符"
    assert all(not image.requires_grad for image in images.values()), "資料不應要求梯度"
    exercise = copy.deepcopy(glyphs)
    exercise["1"][0] = "110"
    exercise_row = torch.tensor([int(c) for c in exercise["1"][0]], dtype=torch.float32)
    assert exercise_row.tolist() == [1.0, 1.0, 0.0], "練習的第一列不符"
    assert list(exercise) == list(glyphs), "練習不應改標籤"
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(c.endswith((".step", ".backward")) for c in calls), "資料程式不應更新參數"

    svg_path = ROOT / "course/figures/rewrite-11-glyph-labels.svg"
    svg = ET.fromstring(svg_path.read_bytes())
    cells = [n for n in svg if n.tag.endswith("rect") and n.get("width") == "44"]
    assert len(cells) == 30, "圖中的格數不符"
    svg_rows = {}
    for label, start in (("0", 67), ("1", 367)):
        grid = [[None] * 3 for _ in range(5)]
        for n in cells:
            x, y = int(n.get("x")), int(n.get("y"))
            if start <= x < start + 132:
                grid[(y - 85) // 44][(x - start) // 44] = 1 if n.get("fill") == "#fff" else 0
        assert grid == images[label].int().tolist(), "圖與資料不一致"
        svg_rows[label] = grid

    canvas = torch.zeros(16, 16)
    canvas[5:10, 6:9] = images["0"]
    rgb = canvas.unsqueeze(0).repeat(3, 1, 1)
    encoder = VisionEncoder(width=8)
    with torch.no_grad():
        features = encoder(rgb.unsqueeze(0))
    try:
        encoder(images["0"])
    except ValueError as error:
        shape_error = str(error)
    else:
        raise AssertionError("入口應拒絕未處理的小圖")
    assert torch.equal(rgb[0], rgb[1]) and torch.equal(rgb[1], rgb[2]), "RGB 通道應相同"
    gradients = {}
    for target in (0, 1):
        logits = torch.zeros(1, 2, requires_grad=True)
        loss = F.cross_entropy(logits, torch.tensor([target]))
        loss.backward()
        gradients[str(target)] = {"loss": loss.item(), "gradient": logits.grad.tolist()}

    report_path = ROOT / "docs/course-experiments/results/ocr.json"
    report = json.loads(report_path.read_bytes())
    result = report["results"]
    data = result["data"]
    families = list(range(100))
    random.Random(report["seed"]).shuffle(families)
    expected_splits = {"train": families[:80], "validation": families[80:90], "test": families[90:]}
    split_results = {}
    all_images = 0
    for split, selected in expected_splits.items():
        expected = [
            {"modality": "vision", "digits": str(value), "offset": offset,
             "family": str(value), "question": "read digits", "answer": str(value)}
            for value in selected for offset in (-1, 0, 1)
        ]
        recorded = data["splits"][split]
        assert expected == recorded["records"], "資料切分與記錄不符"
        assert len(expected) == recorded["count"], "資料筆數不符"
        assert sha(json.dumps(expected, ensure_ascii=False, sort_keys=True).encode()) == recorded["sha256"], "資料指紋不符"
        for value in selected:
            shifted = [np.array(draw_digits(str(value), offset)) for offset in (-1, 0, 1)]
            for arr in shifted:
                assert arr.shape == (16, 16, 3), "影像大小或通道數不符"
                assert set(np.unique(arr)).issubset({0, 255}), "像素值應為黑白"
                assert np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 1], arr[:, :, 2]), "點陣 RGB 應一致"
            assert np.array_equal(np.roll(shifted[0], 1, axis=1), shifted[1]), "左移影像不符"
            assert np.array_equal(np.roll(shifted[1], 1, axis=1), shifted[2]), "右移影像不符"
            all_images += 3
        split_results[split] = {
            "strings": [str(x) for x in selected], "families": len(selected),
            "examples": len(expected), "characters": sorted(set("".join(str(x) for x in selected))),
            "offsets": [-1, 0, 1], "records_sha256": recorded["sha256"],
        }
    assert len(set(families)) == 100 and set(families) == set(range(100)), "字串母體不符"
    train_characters = split_results["train"]["characters"]
    assert train_characters == list("0123456789"), "訓練字元覆蓋不符"
    assert all(len(rows) == 7 and all(len(row) == 5 for row in rows) for rows in GLYPHS), "固定字形尺寸不符"
    tokenizer = ByteTokenizer()
    audited_samples = []
    for sample, record in zip(result["test"]["samples"], data["splits"]["test"]["records"], strict=True):
        generated = sample["generated_ids"]
        raw = generated[:generated.index(tokenizer.eos_id)] if tokenizer.eos_id in generated else generated
        exact = raw == tokenizer.encode(record["answer"])
        assert sample["target"] == record["answer"] and sample["family"] == record["family"], "逐題答案或家族不符"
        assert exact == sample["exact_match"], "逐題完整字串計分不符"
        assert tokenizer.decode(raw) == sample["generated"], "生成字串與 token 不符"
        audited_samples.append({"row": sample["row"], "family": record["family"], "offset": record["offset"],
                                "target": record["answer"], "generated": sample["generated"], "exact_match": exact})
    correct = sum(s["exact_match"] for s in audited_samples)
    assert correct == result["test"]["correct"] == 2, "答對題數不符"
    assert len(audited_samples) == result["test"]["examples"] == 30, "測試分母不符"
    assert correct / len(audited_samples) == result["test"]["exact_match"], "完整正確率不符"
    training = result["training"]
    assert round(training["initial_loss"], 2) == 9.28 and round(training["final_loss"], 2) == 0.81, "代價四捨五入不符"
    assert len(training["history"]) == training["steps"] == 500, "更新步數不符"
    assert sum(h["effective_targets"] for h in training["history"]) == training["effective_targets"], "有效目標合計不符"
    rng = random.Random(report["seed"] - 1)
    probe_records = [rng.choice(data["splits"]["train"]["records"]) for _ in range(8)]
    probe_targets = sum(len(tokenizer.encode(row["answer"])) + 1 for row in probe_records)
    code_inputs = {}
    for path in ["scripts/course_experiments/modalities.py", "tiny_perceptron/multimodal.py", "tiny_perceptron/data.py"]:
        actual = sha((ROOT / path).read_bytes())
        assert actual == report["code_sha256"][path], "實驗程式指紋與目前程式不符"
        code_inputs[path] = actual
    revision_probe = subprocess.run(["git", "show", report["revision"] + ":scripts/prepare_ocr.py"], cwd=ROOT, capture_output=True, check=False)
    help_command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"]
    help_result = subprocess.run(help_command, cwd=ROOT, capture_output=True, check=True)
    spec = experiment_spec("ocr")
    assert spec["module"] == "modalities" and spec["function"] == "run_ocr", "OCR 實驗入口不符"
    png = ROOT / "outputs/course-review-20261019/figure-renders-04/a74a4460e2569c88fc0bc02a8ff91fa47de3bc751a67d301627ef2a96c24413a.png"
    return {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-11-12/probe.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "pillow": pillow_version, "device": "cpu"},
        "input_sha256": {p: sha((ROOT / p).read_bytes()) for p in [str(source.relative_to(ROOT)), str(svg_path.relative_to(ROOT)), str(report_path.relative_to(ROOT)), "scripts/prepare_ocr.py", "docs/course-review-20261019/freeze-04.json"]},
        "glyphs": {k: {"label_type": type(k).__name__, "shape": list(v.shape), "dtype": str(v.dtype), "pixels": v.tolist(), "requires_grad": v.requires_grad} for k, v in images.items()},
        "section_calls": calls, "exercise": {"label": "1", "first_row": exercise_row.tolist()},
        "svg_rows": svg_rows, "viewed_png": {"path": str(png.relative_to(ROOT)), "sha256": sha(png.read_bytes())},
        "rgb": {"shape": list(rgb.shape), "feature_shape": list(features.shape), "input_rejection": shape_error},
        "label_conditioned_gradients": gradients,
        "exact_match_example": {"prediction": "00", "target": "0", "equal": tokenizer.encode("00") == tokenizer.encode("0")},
        "single_original_class_holdout": {"test_labels": ["0"], "remaining_train_labels": [k for k in glyphs if k != "0"]},
        "ocr_splits": split_results, "rendered_examples_checked": all_images,
        "ocr_training": {"initial_loss": training["initial_loss"], "final_loss": training["final_loss"], "loss_probe": training["loss_probe"], "steps": training["steps"], "effective_targets": training["effective_targets"], "weights_changed": training["weights_changed"], "probe_batch_size": 8, "probe_answers": [r["answer"] for r in probe_records], "probe_effective_targets": probe_targets},
        "ocr_test": {"examples": len(audited_samples), "correct": correct, "samples": audited_samples},
        "ocr_recorded_environment": {k: report[k] for k in ["revision", "seed", "device", "torch_version", "python_version", "gpu", "step_scale", "evidence_status"]},
        "recorded_code_matches": code_inputs,
        "entrypoint": {"command_argv": help_command, "exit_code": help_result.returncode, "stdout": help_result.stdout.decode(), "spec": spec},
        "historical_renderer_probe": {"command": "git show " + report["revision"] + ":scripts/prepare_ocr.py", "exit_code": revision_probe.returncode, "stderr": revision_probe.stderr.decode(), "renderer_in_recorded_code_hashes": "scripts/prepare_ocr.py" in report["code_sha256"]},
        "scope": "只重算固定資料與既有記錄；沒有重跑 CUDA 訓練、下載權重或量測新字體及手寫辨識。",
    }


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
