"""Own bounded CPU verification of 11.15; no model or dataset downloads."""
import ast
import hashlib
import importlib.util
import json
import platform
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_perceptron.natural_concepts import thin_stroke_report
from tiny_perceptron.multimodal import scene

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parent
expected = {
    "original_shape": [32, 32], "small_shape": [4, 4],
    "original_brightest": 1.0, "small_brightest": 0.125,
    "pixels_at_least_half_bright_before": 32,
    "pixels_at_least_half_bright_after": 0,
}
report = thin_stroke_report()
print("原圖與縮圖尺寸", report["original_shape"], report["small_shape"])
print("最亮的數字", report["original_brightest"], report["small_brightest"])
print("至少半亮的數字個數", report["pixels_at_least_half_bright_before"], report["pixels_at_least_half_bright_after"])
assert report == expected

# Independent Python arithmetic uses neither F.interpolate nor the helper.
pixels = [[int(col == 15) for col in range(32)] for row in range(32)]
means = [[sum(pixels[row][col] for row in range(8*r, 8*r+8)
              for col in range(8*c, 8*c+8))/64
          for c in range(4)] for r in range(4)]
assert means == [[0, 0.125, 0, 0]]*4
assert sum(value >= .5 for row in pixels for value in row) == 32
assert sum(value >= .5 for row in means for value in row) == 0

left = torch.zeros(1, 1, 32, 32)
right = torch.zeros_like(left)
left[:, :, :, 8] = 1
right[:, :, :, 15] = 1
pooled_left = F.interpolate(left, (4, 4), mode="area")
pooled_right = F.interpolate(right, (4, 4), mode="area")
assert not torch.equal(left, right) and torch.equal(pooled_left, pooled_right)
assert pooled_right[0, 0].tolist() == means

crop_checks = []
for offset, before, after in [(7, 45, 8), (0, 81, 64)]:
    image = scene("red", "square", offset=offset)
    crop = image[:, 4:12, 4:12]
    observed = [int((image[0] > 0).sum()), int((crop[0] > 0).sum())]
    assert observed == [before, after]
    crop_checks.append({"offset": offset, "original_shape": list(image.shape),
                        "crop_shape": list(crop.shape), "red_pixel_counts": observed})

spec = importlib.util.spec_from_file_location("own_prepare_ocr", ROOT / "scripts/prepare_ocr.py")
ocr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ocr)
assert len(ocr.GLYPHS) == 10 and all(len(glyph) == 7 for glyph in ocr.GLYPHS)
assert all(len(row) == 5 for glyph in ocr.GLYPHS for row in glyph)
for text in ("0", "1", "99"):
    assert ocr.draw_digits(text).mode == "RGB"
    assert ocr.draw_digits(text).size == (16, 16)
try:
    ocr.draw_digits("未")
except ValueError as error:
    chinese_error = str(error)
else:
    raise AssertionError("Digit-only drawing tool accepted a Chinese character")

# Check the exact versioned smart_resize implementation without importing
# Transformers or impersonating execution of the whole processor.
research = ROOT / "outputs/natural-v4/factual-research/11.15"
tree = ast.parse((research / "originals/transformers-qwen2-image-4.57.6.txt").read_text())
node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "smart_resize")
namespace = {}
exec("import math", namespace)
exec(compile(ast.Module(body=[node], type_ignores=[]), "original-smart-resize", "exec"), namespace)
resize_checks = []
for side, expected_side, expected_tokens in [(256, 256, 64), (512, 512, 256), (1024, 704, 484)]:
    height, width = namespace["smart_resize"](side, side, factor=32, min_pixels=65536, max_pixels=524288)
    assert (height, width) == (expected_side, expected_side)
    tokens = (height // 16) * (width // 16) // 4
    assert tokens == expected_tokens
    resize_checks.append({"original": [side, side], "resized": [height, width],
                          "raw_patches": (height // 16)*(width // 16), "merged_image_tokens": tokens})

run_dir = ROOT / "outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1/review"
training_path = run_dir / "training.json"
training = json.loads(training_path.read_text())
adapter_config_path = run_dir / "adapter/adapter_config.json"
adapter_config = json.loads(adapter_config_path.read_text())
print("Original training keys", sorted(training))
assert training["model"] == "Qwen/Qwen3-VL-2B-Instruct"
assert training["model_revision"] == "89644892e4d85e24eaac8bacfd4f463576704203"
assert training["optimizer_only_lora"] is True
assert len(training["optimizer_parameter_names"]) == 112
assert all("lora_" in name for name in training["optimizer_parameter_names"])
assert adapter_config["r"] == 8
assert adapter_config["target_modules"] == training["lora_targets"]
assert adapter_config["target_modules"] == r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
assert all(re.fullmatch(adapter_config["target_modules"], name.split(".lora_")[0])
           for name in training["optimizer_parameter_names"])
selection = json.loads((ROOT / "docs/natural-assistant/v4/selection.json").read_text())
assert selection["selected_variant"] == "base"
record_summary = {
    "training_json_path": str(training_path.relative_to(ROOT)),
    "training_json_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
    "adapter_config_path": str(adapter_config_path.relative_to(ROOT)),
    "adapter_config_sha256": hashlib.sha256(adapter_config_path.read_bytes()).hexdigest(),
    "model": training["model"], "model_revision": training["model_revision"],
    "versions": training["versions"],
    "min_pixels": training["min_pixels"], "max_pixels": training["max_pixels"],
    "completed_steps": training["completed_steps"], "trained_rows": training["trained_rows"],
    "optimizer_names": len(training["optimizer_parameter_names"]),
    "optimizer_only_lora": training["optimizer_only_lora"],
    "rank": adapter_config["r"], "target_modules": adapter_config["target_modules"],
    "selected_variant": selection["selected_variant"],
    "scope": "Original record and current configuration inspection; no own GPU training replication, model loading, or general-quality assessment.",
}
ocr_record_path = ROOT / "docs/course-experiments/results/ocr.json"
ocr_record = json.loads(ocr_record_path.read_text())
ocr_splits = {}
families = []
for split, expected_count, expected_families in [("train", 240, 80), ("validation", 30, 10), ("test", 30, 10)]:
    rows = ocr_record["results"]["data"]["splits"][split]["records"]
    assert len(rows) == expected_count
    assert all(row["digits"] == row["answer"] == row["family"] and row["digits"].isascii()
               and row["digits"].isdigit() and 1 <= len(row["digits"]) <= 2 for row in rows)
    split_families = {row["family"] for row in rows}
    assert len(split_families) == expected_families
    assert all(sorted(row["offset"] for row in rows if row["family"] == family) == [-1, 0, 1]
               for family in split_families)
    families.append(split_families)
    ocr_splits[split] = {"records": len(rows), "digit_families": len(split_families), "chinese_labels": 0}
assert not any(a & b for i, a in enumerate(families) for b in families[i+1:])
assert set().union(*families) == {str(x) for x in range(100)}
ocr_scope_audit = {"path": str(ocr_record_path.relative_to(ROOT)),
                   "sha256": hashlib.sha256(ocr_record_path.read_bytes()).hexdigest(),
                   "seed": ocr_record["seed"], "splits": ocr_splits,
                   "scope": ocr_record["results"]["scope"],
                   "verification": "Record-label/content audit only; no own OCR/GPU training or recognition replication."}
result = {
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": "cpu", "cuda_available": str(torch.cuda.is_available())},
    "thin_stroke": report, "independent_block_means": means,
    "collision": {"columns": [8, 15], "different_original_pixels": int((left != right).sum()),
                  "same_area_pooled_image": bool(torch.equal(pooled_left, pooled_right))},
    "prerequisite_crop_checks": crop_checks,
    "digit_ocr_scope": {"glyphs": len(ocr.GLYPHS), "glyph_shape": [7, 5], "canvas": [16, 16],
                        "chinese_input_rejected": chinese_error},
    "original_smart_resize_only": resize_checks,
    "course_training_record": record_summary,
    "previous_ocr_scope_audit": ocr_scope_audit,
    "denominators": {"deterministic_images": 1, "input_pixels": 1024, "output_pixels": 16,
                     "block_pixels": 64, "threshold": 0.5, "seeds": "none; no random helper operation",
                     "ocr_recognition_trials": 0, "model_training_updates_executed_by_reviewer": 0},
}
(ART / "bounded-checks.results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
