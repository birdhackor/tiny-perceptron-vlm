"""Independent bounded CPU check of the representation example in 12.13.

This does not transcribe speech, train a model, or reproduce a benchmark.
"""
from pathlib import Path
import hashlib
import json
import math
import platform
import sys
import contextlib
import io
import re

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.multimodal import log_mel, tone
from tiny_perceptron.natural_concepts import audio_order_report

torch.set_num_threads(1)
report = audio_order_report()
expected = {
    "time_frames": 21,
    "bands": 16,
    "same_time_sequence": False,
    "same_mean_with_roundoff_tolerance": True,
    "mean_shape": [16],
}
assert report == expected, (report, expected)

chapter = (ROOT / "course/chapters/12.md").read_text()
section = chapter[chapter.index("## 12.13 "):]
next_heading = section.find("\n## ", 1)
section = section[:next_heading + 1] if next_heading >= 0 else section
learner_code = re.search(r"```python\n(.*?)\n```", section, re.S).group(1)
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(learner_code, "course/chapters/12.md#12.13", "exec"), {})
assert captured.getvalue() == "時間框與頻帶 21 16\n先後序列相同 False\n平均摘要近似相同 True\n"

# Rebuild the centered, reflected, periodic-Hann-windowed frames without stft.
wave = torch.cat((tone(440, seconds=0.1), tone(880, seconds=0.1)))
assert wave.shape == (3200,) and wave.dtype == torch.float32
pad = 200
reflected = torch.cat((wave[1:pad + 1].flip(0), wave, wave[-pad - 1:-1].flip(0)))
frames = reflected.unfold(0, 400, 160)
assert frames.shape == (21, 400)
window = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(400, dtype=torch.float64) / 400)
power = torch.fft.rfft(frames.to(torch.float64) * window, dim=-1).abs().square().T

# Independent triangular mel bank from the documented constants.
maximum = 2595 * math.log10(1 + 8000 / 700)
edges = [700 * (10 ** ((maximum * j / 17) / 2595) - 1) for j in range(18)]
bank = torch.tensor([
    [max(0.0, min((f - edges[i]) / (edges[i + 1] - edges[i]),
                  (edges[i + 2] - f) / (edges[i + 2] - edges[i + 1])))
     for f in [40 * k for k in range(201)]]
    for i in range(16)
], dtype=torch.float64)
independent = (bank @ power).clamp(min=1e-8).log().T
sequence = log_mel(wave).T
reordered = sequence.flip(0)
means = sequence.mean(0), reordered.mean(0)
allowed = 1e-8 + 1e-5 * means[1].abs()
assert not torch.equal(sequence, reordered)
assert torch.all((means[0] - means[1]).abs() <= allowed)
assert not torch.equal(independent, independent.flip(0))
assert torch.allclose(independent.mean(0), independent.flip(0).mean(0), rtol=1e-12, atol=1e-12)

# fsum makes the permutation check independently of torch.mean's reduction.
column_means = [math.fsum(sequence[:, b].tolist()) / 21 for b in range(16)]
reverse_column_means = [math.fsum(reordered[:, b].tolist()) / 21 for b in range(16)]
assert column_means == reverse_column_means

result = {
    "environment": {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_source_commit": torch.version.git_version, "device": "cpu",
        "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads(),
    },
    "report": report,
    "learner_stdout": captured.getvalue(),
    "inputs": {
        "frequency_hz": [440, 880], "tone_seconds": [0.1, 0.1],
        "sample_rate_hz": 16000, "wave_samples": 3200,
        "n_fft": 400, "hop_samples": 160, "center": True,
        "pad_mode": "reflect", "bands": 16, "dtype": str(wave.dtype),
        "seed": "not applicable; deterministic synthesis, no random operation",
        "training_updates": 0, "recordings_from_people": 0,
    },
    "independent_frame_count": "1 + floor(3200 / 160) = 21",
    "independent_shape": list(independent.shape),
    "frame_reversal_index": "j -> 20-j, same 21 frames, each of width 16",
    "sequence_max_absolute_difference": (sequence - reordered).abs().max().item(),
    "float32_mean_max_absolute_difference": (means[0] - means[1]).abs().max().item(),
    "allclose": {"rtol": 1e-5, "atol": 1e-8, "formula": "abs(a-b) <= atol + rtol*abs(b)"},
    "fsum_mean_vectors_exactly_equal": column_means == reverse_column_means,
    "float64_reconstructed_mean_max_absolute_difference":
        (independent.mean(0) - independent.flip(0).mean(0)).abs().max().item(),
    "scope": "Synthetic feature-frame reversal only; no waveform reversal, ASR, chat, training, or speed/quality benchmark.",
    "sources": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT / "tiny_perceptron/natural_concepts.py", ROOT / "tiny_perceptron/multimodal.py"]
    },
}
print(json.dumps(result, indent=2, ensure_ascii=False))
