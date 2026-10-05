"""Bounded CPU verification of 12.1; no training or model/data downloads."""
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.multimodal import tone

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None
sample_rate = 16000


def peak_hz(wave):
    # Frequencies in cycles/second: k * sample_rate / N.
    power = torch.fft.rfft(wave).abs()
    index = int(power.argmax())
    return {"bin": index, "bin_width_hz": sample_rate / wave.numel(),
            "frequency_hz": index * sample_rate / wave.numel()}


with torch.no_grad():
    wave = tone(440.0)
    longer = tone(440.0, seconds=0.2)
    amplitude_double = 2 * wave
    frequency_double = tone(880.0)
    assert tuple(wave.shape) == (1600,)
    assert tuple(longer.shape) == (3200,)
    assert torch.equal(wave, longer[:1600])
    expected_first = [0.5 * math.sin(2 * math.pi * 440 * n / sample_rate) for n in range(3)]
    error_first = max(abs(a - b) for a, b in zip(wave[:3].tolist(), expected_first))
    assert error_first < 1e-7
    assert [round(v, 3) for v in wave[:3].tolist()] == [0.0, 0.086, 0.169]
    assert abs(wave.min().item() + 0.5) < 1e-6
    assert abs(wave.max().item() - 0.5) < 1e-6
    assert peak_hz(wave)["frequency_hz"] == 440.0
    assert peak_hz(longer)["frequency_hz"] == 440.0
    assert peak_hz(amplitude_double)["frequency_hz"] == 440.0
    assert peak_hz(frequency_double)["frequency_hz"] == 880.0
    batch = torch.stack((wave, tone(220.0)), dim=0)
    assert tuple(batch.shape) == (2, 1600)
    short = tone(440.0, seconds=0.05)
    lengths = torch.tensor([wave.numel(), short.numel()])
    padded = pad_sequence([wave, short], batch_first=True, padding_value=0.0)
    valid = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
    assert tuple(padded.shape) == (2, 1600)
    assert valid.sum(1).tolist() == [1600, 800]
    assert torch.count_nonzero(padded[1, 800:]).item() == 0
    phase_landmarks = [0.5 * math.sin(2 * math.pi * phase) for phase in [0, .25, .5, .75, 1]]
    assert max(abs(a - b) for a, b in zip(phase_landmarks, [0, .5, 0, -.5, 0])) < 1e-15
    result = {
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "torch_git_version": torch.version.git_version,
                        "device": str(wave.device), "dtype": str(wave.dtype),
                        "platform": platform.platform(), "threads": torch.get_num_threads()},
        "code_contract": {"path": "tiny_perceptron/multimodal.py", "lines": [44, 46],
                          "sha256": hashlib.sha256((ROOT / "tiny_perceptron/multimodal.py").read_bytes()).hexdigest(),
                          "formula": "x[n] = 0.5 sin(2 pi f n / fs); N = round(seconds fs); n in [0, N)"},
        "default": {"shape": list(wave.shape), "first_three": wave[:3].tolist(),
                    "rounded_first_three": [round(v, 3) for v in wave[:3].tolist()],
                    "min": wave.min().item(), "max": wave.max().item(),
                    "first_three_max_absolute_error": error_first, "fft": peak_hz(wave),
                    "nominal_duration_seconds": wave.numel() / sample_rate,
                    "last_sample_time_seconds": (wave.numel() - 1) / sample_rate},
        "seconds_0_2": {"shape": list(longer.shape), "fft": peak_hz(longer),
                        "unchanged_first_1600": torch.equal(wave, longer[:1600]),
                        "nominal_duration_seconds": longer.numel() / sample_rate,
                        "last_sample_time_seconds": (longer.numel() - 1) / sample_rate},
        "amplitude_double": {"min": amplitude_double.min().item(), "max": amplitude_double.max().item(),
                             "fft": peak_hz(amplitude_double)},
        "frequency_double": {"shape": list(frequency_double.shape),
                             "min": frequency_double.min().item(), "max": frequency_double.max().item(),
                             "fft": peak_hz(frequency_double)},
        "units": {"sample_interval_seconds": 1 / sample_rate, "period_seconds": 1 / 440,
                  "samples_per_period": sample_rate / 440,
                  "nominal_cycles_default": 440 * .1, "nominal_cycles_longer": 440 * .2,
                  "zero_based_index_440_time_seconds": 440 / sample_rate,
                  "zero_based_index_440_phase_cycles": 440 * 440 / sample_rate},
        "batch_and_padding": {"batch_shape": list(batch.shape), "padded_shape": list(padded.shape),
                              "valid_lengths": lengths.tolist(), "mask_true_counts": valid.sum(1).tolist(),
                              "padded_tail_zero_count": int((padded[1, 800:] == 0).sum())},
        "figure_phase_landmarks": phase_landmarks,
        "requires_grad": wave.requires_grad,
        "scope": "Fixed synthetic mono sinusoids and tensor batching; no speech recognition, hearing test, training or existing-model evaluation. FFT confirms sampled example frequency under its known fs."
    }
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
