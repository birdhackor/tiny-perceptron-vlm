"""核對單音、四點樣板、補點、窗函數與圖中的數值。"""

import hashlib
import inspect
import json
import math
import platform
import re
from pathlib import Path

import torch
from torch.nn import functional as F

from tiny_perceptron.multimodal import log_mel, mel_filter_bank, tone


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(frequency, window):
    waveform = tone(frequency, seconds=0.2)
    spectrum = torch.stft(waveform, 400, 160, window=window, return_complex=True)
    power = spectrum.abs().square()
    average = power.mean(-1)
    peak = average.argmax().item()
    far = torch.ones(201, dtype=torch.bool)
    far[10:13] = False
    return {
        "frequency_hz": frequency,
        "samples": waveform.numel(),
        "dtype": str(waveform.dtype),
        "complex_dtype": str(spectrum.dtype),
        "power_shape": list(power.shape),
        "average_shape": list(average.shape),
        "peak_bin": peak,
        "peak_hz": peak * 16000 / 400,
        "bins_9_to_13": average[9:14].tolist(),
        "relative_bins_9_to_13": (average[9:14] / average[peak]).tolist(),
        "selected_far_bins": {str(k): {"power": average[k].item(), "relative_to_peak": (average[k] / average[peak]).item()} for k in (0, 1, 3, 7, 14, 20, 50, 200)},
        "far_power_sum": average[far].sum().item(),
        "far_power_relative_to_peak": (average[far].sum() / average[peak]).item(),
        "first_five_frame_peak_bins": power[:, :5].argmax(dim=0).tolist(),
    }


def main():
    torch.set_num_threads(1)
    x = torch.tensor([1.0, 0.0, -1.0, 0.0], dtype=torch.float64)
    cosine = x.clone()
    sine = torch.tensor([0.0, 1.0, 0.0, -1.0], dtype=torch.float64)
    twice = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    shifted = x.roll(1)
    scores = [torch.dot(x, template).item() for template in (cosine, sine, twice)]
    shifted_scores = [torch.dot(shifted, template).item() for template in (cosine, sine)]
    phase_sweep = []
    for phase in (0.0, math.pi / 4, math.pi / 2, math.pi):
        signal = torch.cos(2 * math.pi * torch.arange(4, dtype=torch.float64) / 4 - phase)
        a, b = torch.dot(signal, cosine), torch.dot(signal, sine)
        phase_sweep.append({"phase": phase, "cosine_score": a.item(), "sine_score": b.item(), "power": (a*a+b*b).item()})

    waveform = tone(440.0, seconds=0.2)
    hann = torch.hann_window(400)
    padded = F.pad(waveform.view(1, 1, -1), (200, 200), mode="reflect").view(-1)
    frames = padded.unfold(0, 400, 160)
    reconstructed = torch.fft.rfft(frames * hann).T
    actual = torch.stft(waveform, 400, 160, window=hann, return_complex=True)
    toy = torch.arange(5.0).view(1, 1, -1)
    reflected_toy = F.pad(toy, (2, 2), mode="reflect").flatten().tolist()
    frame_number, bin_number = 5, 11
    weighted = (frames[frame_number] * hann).double()
    angle = 2 * math.pi * bin_number * torch.arange(400, dtype=torch.float64) / 400
    manual_real = torch.dot(weighted, angle.cos()).item()
    manual_imag = -torch.dot(weighted, angle.sin()).item()
    coefficient = actual[bin_number, frame_number]
    mixture = tone(440.0, seconds=0.2) + tone(880.0, seconds=0.2)
    mixture_power = torch.stft(mixture, 400, 160, window=hann, return_complex=True).abs().square()
    reference_log_mel = (mel_filter_bank() @ actual.abs().square()).clamp(min=1e-8).log()

    svg = Path("course/figures/multimodal_audio_axes.svg").read_text()
    points = re.search(r'<polyline points="([^"]+)"', svg)[1]
    coordinates = [tuple(map(float, pair.split(","))) for pair in points.split()]
    errors = []
    for px, py in coordinates:
        time = (px - 60) / 495 * 0.008
        expected_y = 165 - 90 * (0.5 * math.sin(2 * math.pi * 440 * time))
        errors.append(abs(py - expected_y))

    report = {
        "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-5/verify_stft.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "default_dtype": str(torch.get_default_dtype()), "threads": str(torch.get_num_threads())},
        "four_point": {"scores_cos_sin_twice": scores, "power_one_cycle": scores[0]**2+scores[1]**2, "shifted_signal": shifted.tolist(), "shifted_scores": shifted_scores, "shifted_power": sum(v*v for v in shifted_scores), "nyquist_sine_max_abs": torch.sin(math.pi*torch.arange(4, dtype=torch.float64)).abs().max().item(), "phase_sweep": phase_sweep},
        "complex_example": {"absolute": abs(3+4j), "absolute_squared": abs(3+4j)**2, "torch_absolute_squared": torch.tensor(3+4j).abs().square().item()},
        "tones_hann": [summarize(frequency, hann) for frequency in (440.0, 480.0, 450.0)],
        "tone_450_rectangular": summarize(450.0, torch.ones(400)),
        "framing": {"padded_samples": padded.numel(), "padded_frame_shape": list(frames.shape), "unfold_without_padding_shape": list(waveform.unfold(0, 400, 160).shape), "stft_without_padding_shape": list(torch.stft(waveform, 400, 160, window=hann, center=False, return_complex=True).shape), "reconstructed_stft_max_error": (actual-reconstructed).abs().max().item(), "first_five_center_sample_indices": [160*n for n in range(5)], "first_five_center_seconds": [160*n/16000 for n in range(5)], "reflect_example_input": list(range(5)), "reflect_example_result": reflected_toy, "left_reflect_excludes_endpoint": torch.equal(padded[:200], waveform[1:201].flip(0)), "right_reflect_excludes_endpoint": torch.equal(padded[-200:], waveform[-201:-1].flip(0)), "frame_seconds": 400/16000, "hop_seconds":160/16000, "bin_spacing_hz":16000/400, "nyquist_hz":16000/2, "frequency_bin_count":400//2+1},
        "coefficient_check": {"frame_index":frame_number, "bin_index":bin_number, "manual_real":manual_real, "manual_imaginary":manual_imag, "stft_real":coefficient.real.item(), "stft_imaginary":coefficient.imag.item(), "absolute_error":abs(complex(manual_real,manual_imag)-complex(coefficient.item()))},
        "multiple_frequency_example": {"frame_index":5, "power_at_440_hz":mixture_power[11,5].item(), "power_at_880_hz":mixture_power[22,5].item(), "interpretation":"合成雙音只示範同框多個頻率成分；真人語音的概念另核對 Praat 原始文件。"},
        "log_mel_power_path": {"max_error": (log_mel(waveform)-reference_log_mel).abs().max().item(), "phase_path":"log_mel 的 STFT 經 abs().square() 後才進 mel filter bank，沒有取用 complex angle。"},
        "figure_waveform": {"point_count":len(coordinates), "cycles_in_0_008_seconds":440*0.008, "max_y_coordinate_error_pixels":max(errors), "coordinate_tolerance_pixels":0.02, "interpretation":"折線座標與 440 Hz、振幅 0.5 的公式一致；下圖是省略頻率格的示意，亮度未宣稱為實測值。"},
        "repository_input": {"path":"tiny_perceptron/multimodal.py", "sha256":digest("tiny_perceptron/multimodal.py"), "tone_source":inspect.getsource(tone)},
    }
    assert scores == [2.0, 0.0, 0.0]
    assert shifted_scores == [0.0, 2.0]
    assert (actual-reconstructed).abs().max().item() < 1e-5
    assert max(errors) < 0.02
    assert [entry["peak_bin"] for entry in report["tones_hann"]] == [11,12,11]
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
