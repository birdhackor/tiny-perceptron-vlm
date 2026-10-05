"""Independent bounded checks for current section 12.5; no model or data work."""
import cmath
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.multimodal import tone

torch.set_num_threads(1)
torch.set_default_device("cpu")
OUT = Path(__file__).resolve().parent

def get_power(frequency, center=True):
    wave = tone(frequency, seconds=0.2)
    spectrum = torch.stft(wave, 400, 160, window=torch.hann_window(400),
                          center=center, return_complex=True)
    return wave, spectrum, spectrum.abs().square()

observed = {}
for f in [440.0, 480.0, 450.0]:
    wave, spectrum, power = get_power(f)
    average = power.mean(-1)
    values, indices = average.topk(4)
    observed[str(f)] = {
        "wave_samples": wave.numel(), "shape": list(power.shape),
        "complex_output": spectrum.is_complex(), "time_average_denominator": power.shape[-1],
        "peak_bin": average.argmax().item(),
        "peak_hz": average.argmax().item() * 16000 / 400,
        "top4": [{"bin": k, "hz": k * 40, "mean_squared_magnitude": v}
                 for k, v in zip(indices.tolist(), values.tolist())],
        "frequency_grid_hz": 16000 / 400,
        "mean_equals_sum_over_21_max_error": (average - power.sum(-1) / 21).abs().max().item(),
    }
    assert wave.numel() == 3200 and list(power.shape) == [201, 21]
    assert torch.allclose(average, power.sum(-1) / 21)
assert observed["440.0"]["peak_bin"] == 11
assert observed["480.0"]["peak_bin"] == 12
assert observed["450.0"]["peak_bin"] == 11
assert {11, 12}.issubset({x["bin"] for x in observed["450.0"]["top4"]})

wave, centered, centered_power = get_power(440.0)
_, uncentered, _ = get_power(440.0, center=False)
observed["frame_count"] = {
    "center_true": list(centered.shape), "center_false": list(uncentered.shape),
    "unfold": list(wave.unfold(0, 400, 160).shape),
    "formulas": {"center_true": 1 + 3200 // 160, "center_false": 1 + (3200 - 400) // 160},
    "padding_each_side_samples": 400 // 2,
    "frame_center_seconds_first5": [m * 160 / 16000 for m in range(5)],
    "frame_seconds": 400 / 16000, "hop_seconds": 160 / 16000,
}
assert list(uncentered.shape) == [201, 18]
assert list(wave.unfold(0, 400, 160).shape) == [18, 400]

# Compute one interior coefficient directly from the DFT definition, independent
# of torch.stft and FFT implementations; the centered frame m=10 starts at 1400.
raw = wave.tolist()
manual = sum((0.5 - 0.5 * math.cos(2 * math.pi * k / 400)) * raw[1400 + k]
             * cmath.exp(-2j * math.pi * 11 * k / 400) for k in range(400))
actual = centered[11, 10].item()
error = abs(actual - manual)
assert error < 0.001
observed["direct_dft"] = {
    "frequency_bin": 11, "center_frame": 10, "original_samples": [1400, 1799],
    "manual_complex": [manual.real, manual.imag], "stft_complex": [actual.real, actual.imag],
    "absolute_complex_error": error, "tolerance": 0.001,
}

z = torch.tensor(3 + 4j)
observed["complex_power"] = {
    "magnitude_3_plus_4j": z.abs().item(), "squared_magnitude": z.abs().square().item(),
    "real_imag_identity_max_error":
        (centered_power - (centered.real.square() + centered.imag.square())).abs().max().item(),
}
assert z.abs().item() == 5.0 and z.abs().square().item() == 25.0
assert torch.allclose(centered_power, centered.real.square() + centered.imag.square(), atol=0.001)

mix = tone(440.0, seconds=0.2) + tone(880.0, seconds=0.2)
mixed_power = torch.stft(mix, 400, 160, window=torch.hann_window(400),
                         return_complex=True).abs().square().mean(-1)
indices = mixed_power.topk(2).indices.tolist()
observed["two_tones"] = {"top2_bins": indices, "top2_hz": [i * 40 for i in indices]}
assert set(indices) == {11, 22}

observed["power_scale"] = {
    "definition": "unnormalized squared STFT magnitude in squared waveform units; neither watts nor PSD/Hz",
    "normalized_argument_default": False,
    "doubling_amplitude_max_error_from_4x_power":
        (torch.stft(2 * wave, 400, 160, window=torch.hann_window(400), return_complex=True)
         .abs().square() - 4 * centered_power).abs().max().item(),
}
assert observed["power_scale"]["doubling_amplitude_max_error_from_4x_power"] == 0.0

result = {
    "environment": {"python": sys.version, "python_executable": sys.executable,
                    "torch": str(torch.__version__), "torch_git_version": torch.version.git_version,
                    "device": "cpu", "cuda_available": str(torch.cuda.is_available()),
                    "platform": platform.platform()},
    "observations": observed,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "assertions": "all passed",
    "scope": "Synthetic waveform transforms only; no training, model evaluation or downloads.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
