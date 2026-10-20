"""核對窗權重、能量，以及整數週期與非整數週期單音的頻譜。"""

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.multimodal import tone


def spectrum_record(frame, frequency, window):
    count = frame.numel()
    weighted = frame * window
    spectrum = torch.fft.rfft(weighted)
    power = spectrum.abs().square()
    peak = int(power.argmax())
    mask = torch.ones_like(power, dtype=torch.bool)
    mask[peak] = False
    return {
        "frequency_hz": frequency,
        "cycles_per_frame": frequency * count / 16000,
        "first_sample": frame[0].item(),
        "last_sample": frame[-1].item(),
        "repeat_boundary_step": (frame[0] - frame[-1]).item(),
        "energy": frame.square().sum().item(),
        "weighted_energy": weighted.square().sum().item(),
        "peak_bin": peak,
        "peak_magnitude": spectrum.abs()[peak].item(),
        "off_peak_power_fraction": (power[mask].sum() / power.sum()).item(),
        "bin_magnitudes_9_to_14": spectrum.abs()[9:15].tolist(),
    }


torch.set_num_threads(1)
torch.set_default_device("cpu")
frame = tone()[:400]
window = torch.hann_window(400)
symmetric = torch.hann_window(400, periodic=False)
ones = torch.ones(400)
indices = torch.arange(400, dtype=torch.float64)
records = {}
for frequency in (440.0, 450.0):
    exact_frame = 0.5 * torch.sin(2 * math.pi * frequency * indices / 16000)
    records[str(frequency)] = {
        "rectangular": spectrum_record(exact_frame, frequency, torch.ones_like(exact_frame)),
        "hann": spectrum_record(exact_frame, frequency, torch.hann_window(400, dtype=torch.float64)),
        "untruncated_next_sample": 0.5 * math.sin(2 * math.pi * frequency * 400 / 16000),
    }

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-4/window_probe_current_01.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "device": "cpu",
        "platform": platform.platform(),
    },
    "inputs": {
        "sample_rate": 16000,
        "frame_length": 400,
        "tone_amplitude": 0.5,
        "repository_tone_dtype": str(frame.dtype),
        "analytic_probe_dtype": "torch.float64",
        "fft_normalization": "backward",
        "spectrum": "rfft, no padding, no centering, no averaging",
        "power_fraction_denominator": "sum of all rfft magnitude squares; relative diagnostic, no PSD normalization",
    },
    "repository_tone": {
        "rectangular": spectrum_record(frame, 440.0, ones),
        "hann": spectrum_record(frame, 440.0, window),
        "ones_exactly_preserves_frame": torch.equal(frame * ones, frame),
        "frame_shape_after_hann": list((frame * window).shape),
        "requires_grad": bool((frame * window).requires_grad),
        "duration_seconds": len(frame) / 16000,
    },
    "window": {
        "start": window[0].item(),
        "middle": window[200].item(),
        "last": window[-1].item(),
        "last_float64_formula": 0.5 * (1 - math.cos(2 * math.pi * 399 / 400)),
        "minimum": window.min().item(),
        "maximum": window.max().item(),
        "symmetric_start": symmetric[0].item(),
        "symmetric_last": symmetric[-1].item(),
        "periodic_equals_trimmed_symmetric": torch.equal(window, torch.hann_window(401, periodic=False)[:-1]),
    },
    "analytic_tones": records,
    "input_hashes": {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "tiny_perceptron/multimodal.py",
            "docs/course-review-20261019/freeze-05/sources/12.4.md",
        )
    },
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
