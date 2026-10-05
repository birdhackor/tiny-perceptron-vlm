"""Independent bounded CPU checks for section 11.15; no model fitting or inference."""

import ast
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from scripts.fetch_natural_release import student_options
from tiny_perceptron.multimodal import patchify
from tiny_perceptron.natural_concepts import thin_stroke_report

assert torch.version.cuda is None
torch.set_default_device("cpu")
torch.set_num_threads(1)
environment = {
    "python": sys.version,
    "executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "device": "cpu",
    "platform": platform.platform(),
    "thread_count": str(torch.get_num_threads()),
}
(OUT / "cpu-check.environment.json").write_text(json.dumps(environment, indent=2) + "\n")

observed = thin_stroke_report()
expected = {
    "original_shape": [32, 32],
    "small_shape": [4, 4],
    "original_brightest": 1.0,
    "small_brightest": 0.125,
    "pixels_at_least_half_bright_before": 32,
    "pixels_at_least_half_bright_after": 0,
}
assert observed == expected

image = torch.zeros(1, 1, 32, 32)
image[:, :, :, 15] = 1
small = F.interpolate(image, (4, 4), mode="area")
manual = image.reshape(1, 1, 4, 8, 4, 8).mean(dim=(3, 5))
assert torch.equal(manual, small)
assert int((image == 1).sum()) == 32
assert float(small.max()) == 8 / 64 == 1 / 8
assert torch.equal(small[0, 0, :, 1], torch.full((4,), 0.125))
assert int(torch.count_nonzero(small)) == 4

other = torch.zeros_like(image)
other[:, :, :, 8] = 1
other_small = F.interpolate(other, (4, 4), mode="area")
assert not torch.equal(image, other)
assert torch.equal(small, other_small)
cross_block = torch.zeros_like(image)
cross_block[:, :, :, 16] = 1
assert not torch.equal(small, F.interpolate(cross_block, (4, 4), mode="area"))

thresholds = {
    str(t): {"before": int((image >= t).sum()), "after": int((small >= t).sum())}
    for t in [0.1, 0.125, 0.5]
}
assert thresholds["0.1"]["after"] == 4
assert thresholds["0.125"]["after"] == 4
assert thresholds["0.5"]["after"] == 0
resolution_variants = []
for size in [4, 8, 16, 32]:
    resized = F.interpolate(image, (size, size), mode="area")
    resolution_variants.append({
        "height_width": [size, size],
        "max": float(resized.max()),
        "pixels_at_least_half": int((resized >= 0.5).sum()),
    })
assert [r["max"] for r in resolution_variants] == [0.125, 0.25, 0.5, 1.0]

patch_counts = []
for size in [16, 32]:
    patches = patchify(torch.zeros(1, 3, size, size), patch_size=4)
    n = (size // 4) ** 2
    assert tuple(patches.shape) == (1, n, 48)
    patch_counts.append({"image_height_width": [size, size], "patch_size": 4,
                         "patch_tensor_shape": list(patches.shape),
                         "text_positions": 20, "total_positions": n + 20,
                         "dense_attention_entries": (n + 20) ** 2})
assert [r["total_positions"] for r in patch_counts] == [36, 84]
assert [r["dense_attention_entries"] for r in patch_counts] == [1296, 7056]

manifest_path = ROOT / "docs/natural-assistant/v4/public-release.json"
manifest = json.loads(manifest_path.read_bytes())
options = student_options(manifest, OUT, device="cpu", local_files_only=True)
assert manifest["selected_variant"] == "base"
assert manifest["adapter_parameters"] == 0
assert options.adapter is None
assert options.model == "Qwen/Qwen3-VL-2B-Instruct"

results = {
    "thin_stroke_report": observed,
    "axes": {"image": "batch, channel, height, width", "pixel_values": "dimensionless normalized brightness",
             "before_grid_pixels": 1024, "after_grid_pixels": 16,
             "bright_input_pixels_per_active_8x8_block": 8, "averaging_denominator": 64},
    "manual_block_mean_exactly_equal_to_area": True,
    "small_raw_matrix": small[0, 0].tolist(),
    "position_collision": {"first_bright_column_zero_based": 15, "second_bright_column_zero_based": 8,
                           "different_originals": True, "identical_area_outputs": True,
                           "column_16_produces_different_output": True},
    "threshold_variants": thresholds,
    "resolution_variants": resolution_variants,
    "patch_count_variants": patch_counts,
    "selected_public_configuration": {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                                      "json_pointers_read_for_outcome": ["/selected_variant", "/adapter_parameters", "/base_model"],
                                      "selected_variant": manifest["selected_variant"],
                                      "adapter_parameters": manifest["adapter_parameters"],
                                      "student_options_adapter": options.adapter,
                                      "student_options_model": options.model,
                                      "model_revision": options.model_revision},
    "boundaries": "Arithmetic and configuration inspection only; no OCR, existing-model inference, training, downloads, or model serialization.",
}
(OUT / "cpu-check.results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2))
