"""Bounded CPU checks for section 12.4; no data/model downloads or training."""
from pathlib import Path
import hashlib
import json
import math
import platform
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.multimodal import tone

torch.set_num_threads(1)
torch.set_default_device("cpu")
N, FS, NFFT = 400, 16000, 40000
frame = tone()[:N]
window = torch.hann_window(N)
weighted = frame * window
assert torch.equal(frame * torch.ones(N), frame)
assert not frame.requires_grad and not window.requires_grad and not weighted.requires_grad
assert torch.version.cuda is None and not torch.cuda.is_available()

n = torch.arange(N, dtype=torch.float64)
periodic = torch.hann_window(N, dtype=torch.float64)
symmetric = torch.hann_window(N, periodic=False, dtype=torch.float64)
formula = 0.5 - 0.5 * torch.cos(2 * math.pi * n / N)
assert torch.allclose(periodic, formula, atol=1e-15, rtol=0)
assert torch.equal(periodic, torch.hann_window(N + 1, periodic=False, dtype=torch.float64)[:-1])
assert symmetric[0] == symmetric[-1] == 0
integer = 0.5 * torch.sin(2 * math.pi * 440 * n / FS)
assert abs(integer.square().sum().item() - 50) < 1e-12
assert abs((integer * periodic).square().sum().item() - 18.75) < 1e-12

freqs = torch.fft.rfftfreq(NFFT, 1 / FS, dtype=torch.float64)
response = {}
for name, win, first_zero in [
    ("rectangular", torch.ones(N, dtype=torch.float64), 40),
    ("periodic_hann", periodic, 80),
]:
    magnitude = torch.fft.rfft(win, n=NFFT).abs()
    magnitude /= magnitude[0].item()
    idx = round(first_zero * NFFT / FS)
    assert magnitude[idx] < 1e-14
    # The first sidelobe is between the first and second positive zeros.
    stop = idx + NFFT // N
    peak = magnitude[idx + 1 : stop].max().item()
    response[name] = {
        "first_zero_hz": first_zero,
        "full_main_lobe_zero_to_zero_width_hz": 2 * first_zero,
        "first_sidelobe_peak_relative_amplitude_db": 20 * math.log10(peak),
        "first_zero_relative_amplitude": magnitude[idx].item(),
    }
assert response["periodic_hann"]["first_sidelobe_peak_relative_amplitude_db"] < -31
assert response["rectangular"]["first_sidelobe_peak_relative_amplitude_db"] > -14

noninteger = 0.5 * torch.sin(2 * math.pi * 450 * n / FS)
leakage = {}
onesided_factors = torch.full((NFFT // 2 + 1,), 2.0, dtype=torch.float64)
onesided_factors[0] = onesided_factors[-1] = 1
outside = (freqs - 450).abs() >= 80
for name, win in [("rectangular", torch.ones(N, dtype=torch.float64)), ("periodic_hann", periodic)]:
    x = noninteger * win
    energy_spectrum = torch.fft.rfft(x, n=NFFT).abs().square() * onesided_factors
    leakage[name] = {
        "periodic_join_jump_abs": abs((x[0] - x[-1]).item()),
        "outside_common_450_plus_or_minus_80_hz_energy_fraction": (
            energy_spectrum[outside].sum() / energy_spectrum.sum()
        ).item(),
    }
assert leakage["periodic_hann"]["periodic_join_jump_abs"] < leakage["rectangular"]["periodic_join_jump_abs"]
assert leakage["periodic_hann"]["outside_common_450_plus_or_minus_80_hz_energy_fraction"] < leakage["rectangular"]["outside_common_450_plus_or_minus_80_hz_energy_fraction"]

integer_bins = {}
for name, win in [("rectangular", torch.ones(N, dtype=torch.float64)), ("periodic_hann", periodic)]:
    power = torch.fft.rfft(integer * win).abs().square()
    factors = torch.full_like(power, 2)
    factors[0] = factors[-1] = 1
    power *= factors
    integer_bins[name] = {
        "bins_above_1e_minus_12_of_max": torch.where(power > power.max() * 1e-12)[0].tolist(),
        "bin_11_energy_fraction": (power[11] / power.sum()).item(),
    }
assert integer_bins["rectangular"]["bins_above_1e_minus_12_of_max"] == [11]
assert integer_bins["periodic_hann"]["bins_above_1e_minus_12_of_max"] == [10, 11, 12]

result = {
    "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu", "threads": "1", "cuda_build": str(torch.version.cuda)},
    "inputs": {"samples": N, "sample_rate_hz": FS, "duration_seconds": N / FS, "original_tone_hz": 440, "integer_cycles": 440 * N / FS, "variant_tone_hz": 450, "variant_cycles": 450 * N / FS},
    "default_float32": {"first_weight": window[0].item(), "middle_weight": window[N // 2].item(), "last_weight": window[-1].item(), "original_sum_of_squares": frame.square().sum().item(), "weighted_sum_of_squares": weighted.square().sum().item(), "ones_control_equal": True},
    "float64_formula": {"periodic_max_error": (periodic - formula).abs().max().item(), "last_periodic_weight": periodic[-1].item(), "last_symmetric_weight": symmetric[-1].item(), "periodic_equals_symmetric_N_plus_1_trimmed": True},
    "window_frequency_responses": response,
    "noninteger_tone": leakage,
    "integer_tone_original_N_point_DFT": integer_bins,
    "measurement_contract": {"fft_length": NFFT, "frequency_spacing_hz": FS / NFFT, "window_magnitude_normalization": "divide by DC=sum(window) for unit peak", "sidelobe_db": "20*log10(relative amplitude)", "leakage_denominator": "sum of all one-sided FFT squared magnitudes, with non-DC/non-Nyquist bins doubled", "leakage_numerator": "same weighted spectral energy outside the common [370,530] Hz band", "main_lobe_width": "first negative to first positive zero of window frequency response; isolates the window response", "zero_padding_scope": "interpolates a finite window's response; does not add frequency resolution", "scope": "mechanism demonstration; not a model experiment or universal leakage guarantee for every signal"},
    "repository_tone_source": {"path": "tiny_perceptron/multimodal.py", "read_lines": "44-46", "sha256": hashlib.sha256((ROOT / "tiny_perceptron/multimodal.py").read_bytes()).hexdigest()},
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
