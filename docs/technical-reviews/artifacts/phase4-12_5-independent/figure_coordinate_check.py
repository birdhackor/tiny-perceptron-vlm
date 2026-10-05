"""Compare SVG waveform coordinates with an independently calculated sinusoid."""
import json
import math
import re
from pathlib import Path

out = Path(__file__).resolve().parent
svg = (out / "inputs/course/figures/multimodal_audio_axes.svg").read_text()
coords = [tuple(map(float, x.split(","))) for x in
          re.search(r'<polyline points="([^"]+)"', svg)[1].split()]
errors = []
for i, (x, y) in enumerate(coords):
    t = 0.008 * i / (len(coords) - 1)
    expected_x = 60 + 495 * i / (len(coords) - 1)
    expected_y = 165 - 90 * 0.5 * math.sin(2 * math.pi * 440 * t)
    errors.append((abs(x - expected_x), abs(y - expected_y)))
max_errors = [max(e[j] for e in errors) for j in range(2)]
assert all(e <= 0.005001 for e in max_errors)
print(json.dumps({
    "samples": len(coords), "time_seconds": [0, 0.008], "frequency_hz": 440,
    "amplitude": 0.5, "x_range": [60, 555], "y_zero": 165,
    "pixels_per_amplitude": 90, "max_coordinate_error_pixels": max_errors,
    "tolerance_pixels": 0.005001,
    "method": "Generate exact equally spaced sample coordinates, then compare the independently rounded x and y coordinates.",
    "prior_check": "An earlier ad hoc inverse-x check used rounded x as exact time and failed an overly tight 0.01-pixel y tolerance. Its time-rounding error propagated into phase. This final check compares x and y separately against the unrounded sample index.",
    "spectrogram_scope": "Only the displayed schematic axes and labels are checked; brightness is explicitly not measured data.",
}, ensure_ascii=False, indent=2))
