"""Independent bounded CPU checks for section 11.15; no model weights loaded."""

import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from tiny_perceptron.natural_concepts import thin_stroke_report

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None

image = torch.zeros(1, 1, 32, 32, dtype=torch.float32, device="cpu")
image[:, :, :, 15] = 1
small = F.interpolate(image, (4, 4), mode="area")
manual = image.reshape(1, 1, 4, 8, 4, 8).mean((3, 5))
expected = torch.zeros(1, 1, 4, 4)
expected[:, :, :, 1] = 0.125
assert torch.equal(small, manual) and torch.equal(small, expected)
assert int((image >= 0.5).sum()) == 32
assert int((small >= 0.5).sum()) == 0
assert float(image.max()) == 1.0 and float(small.max()) == 0.125
report = thin_stroke_report()
assert report == {
    "original_shape": [32, 32],
    "small_shape": [4, 4],
    "original_brightest": 1.0,
    "small_brightest": 0.125,
    "pixels_at_least_half_bright_before": 32,
    "pixels_at_least_half_bright_after": 0,
}

# Distinct fine positions in the SAME pooling bin give identical model input.
alternate = torch.zeros_like(image)
alternate[:, :, :, 14] = 1
alternate_small = F.interpolate(alternate, (4, 4), mode="area")
assert not torch.equal(image, alternate)
assert torch.equal(small, alternate_small)
restored_size = F.interpolate(small, (32, 32), mode="nearest")
assert not torch.equal(restored_size, image)

# Actual PNG raster check after separately rendering and viewing the SVG.
raster_path = Path("docs/technical-reviews/artifacts/natural-11.15-stroke-render.png")
raster = Image.open(raster_path).convert("RGB")
assert raster.size == (1200, 430)
scan_y = 200
bright_scan = [raster.getpixel((x, scan_y)) == (255, 255, 255) for x in range(55, 375)]
white_width = sum(bright_scan)
# Grid strokes blend with the two edges; they are decoration, not tensor data.
stripe_width = sum(all(channel > 127 for channel in raster.getpixel((x, scan_y))) for x in range(55, 375))
assert white_width == 38
assert stripe_width == 40
assert stripe_width / 320 == 1 / 8
assert raster.getpixel((790, 220)) == (32, 32, 32)
assert abs(32 / 255 - 0.125) < 1 / 255

source_path = Path(inspect.getsourcefile(F.interpolate))
official_source = Path("docs/technical-reviews/artifacts/natural-11.15-pytorch-functional-source.py")
source_equal = source_path.read_bytes() == official_source.read_bytes()
result = {
    "environment": {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda_build": str(torch.version.cuda),
        "device": str(image.device),
        "dtype": str(image.dtype),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "torch_threads": torch.get_num_threads(),
    },
    "helper_report": report,
    "input_shape": list(image.shape),
    "original_bright_column": 15,
    "output_shape": list(small.shape),
    "output_values": small[0, 0].tolist(),
    "manual_block_mean_matches_exactly": torch.equal(small, manual),
    "different_fine_images": True,
    "alternate_bright_column": 14,
    "different_input_values": int((image != alternate).sum()),
    "identical_area_outputs": torch.equal(small, alternate_small),
    "resize_back_restores_original": torch.equal(restored_size, image),
    "figure": {
        "size": list(raster.size),
        "scan_y": scan_y,
        "left_rectangle_width": 320,
        "pure_white_interior_width": white_width,
        "bright_stripe_width": stripe_width,
        "bright_fraction": stripe_width / 320,
        "right_center_rgb": list(raster.getpixel((790, 220))),
        "right_gray_normalized": 32 / 255,
        "note": "Expanded illustrative bins, not a literal 32x32 screenshot. Grid strokes blend the two stripe-edge pixels, leaving 38 pure-white pixels in the 40-pixel bin. Right gray rounds 255/8 to 32; left dark palette is illustrative rather than calibrated intensity.",
    },
    "installed_functional_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "downloaded_official_functional_source_sha256": hashlib.sha256(official_source.read_bytes()).hexdigest(),
    "whole_source_bytes_equal": source_equal,
    "assertions_passed": True,
    "limitations": "Only synthetic float32 CPU pooling and raster geometry were tested. No Chinese OCR model, training, GPU, timing, or quality benchmark was run.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
