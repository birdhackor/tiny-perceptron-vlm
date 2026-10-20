"""核對時間框倒轉、平均摘要與 log-mel 的資訊範圍。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.multimodal import log_mel, tone
from tiny_perceptron.natural_concepts import audio_order_report


torch.set_num_threads(1)
first = tone(440, seconds=0.1)
second = tone(880, seconds=0.1)
wave = torch.cat((first, second))
frames = log_mel(wave).transpose(-1, -2)
reverse = frames.flip(0)
mean = frames.mean(0)
reverse_mean = reverse.mean(0)
limits = 1e-8 + 1e-5 * reverse_mean.abs()

# 這兩個正功率矩陣有相同的逐頻帶 mean(log(power))，總功率卻不同。
powers_a = torch.tensor([[1.0, 9.0], [9.0, 1.0]], dtype=torch.float64)
powers_b = torch.tensor([[3.0, 3.0], [3.0, 3.0]], dtype=torch.float64)

result = {
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": str(frames.device),
        "dtype": str(frames.dtype),
        "threads": str(torch.get_num_threads()),
    },
    "inputs": {
        "frequencies_hz": [440, 880],
        "sample_rate": 16000,
        "duration_each_seconds": 0.1,
        "samples_each": [first.numel(), second.numel()],
        "samples_total": wave.numel(),
        "duration_total_seconds": wave.numel() / 16000,
        "n_fft": 400,
        "hop_length": 160,
        "center": True,
        "bands": 16,
    },
    "report": audio_order_report(),
    "frames": {
        "shape": list(frames.shape),
        "mean_shape": list(mean.shape),
        "all_finite": bool(torch.isfinite(frames).all()),
        "flip_twice_recovers_original": bool(torch.equal(reverse.flip(0), frames)),
        "same_sorted_values_each_band": bool(torch.equal(frames.sort(0).values, reverse.sort(0).values)),
        "same_sequence": bool(torch.equal(frames, reverse)),
        "mean_original": mean.tolist(),
        "mean_reverse": reverse_mean.tolist(),
        "maximum_absolute_mean_difference": float((mean - reverse_mean).abs().max()),
        "maximum_error_to_allclose_limit_ratio": float(((mean - reverse_mean).abs() / limits).max()),
        "allclose_defaults": bool(torch.allclose(mean, reverse_mean)),
        "mean_float64_accumulation_maximum_difference": float(
            (frames.to(torch.float64).mean(0) - reverse.to(torch.float64).mean(0)).abs().max()
        ),
        "peak_band_original_first_last": [int(frames[0].argmax()), int(frames[-1].argmax())],
        "peak_band_reverse_first_last": [int(reverse[0].argmax()), int(reverse[-1].argmax())],
    },
    "waveform_comparison": {
        "reverse_waveform_spectrum_equals_reversed_frames": bool(
            torch.equal(log_mel(wave.flip(0)).transpose(-1, -2), reverse)
        ),
        "swap_segments_spectrum_equals_reversed_frames": bool(
            torch.equal(log_mel(torch.cat((second, first))).transpose(-1, -2), reverse)
        ),
    },
    "log_power_counterexample": {
        "powers_a": powers_a.tolist(),
        "powers_b": powers_b.tolist(),
        "mean_log_a": powers_a.log().mean(0).tolist(),
        "mean_log_b": powers_b.log().mean(0).tolist(),
        "sum_power_a": powers_a.sum(0).tolist(),
        "sum_power_b": powers_b.sum(0).tolist(),
        "same_mean_log": bool(torch.allclose(powers_a.log().mean(0), powers_b.log().mean(0), rtol=0, atol=1e-14)),
    },
    "repository_code_sha256": {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in ("tiny_perceptron/natural_concepts.py", "tiny_perceptron/multimodal.py")
    },
    "scope": "只核對合成單音與固定 log-mel 配置；功率矩陣反例是代數驗證，沒有訓練或執行語音辨識。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
