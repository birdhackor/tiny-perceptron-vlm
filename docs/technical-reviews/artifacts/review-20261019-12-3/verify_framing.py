"""核對時間框索引、邊界約定與有限長事件的涵蓋條件。"""

import hashlib
import json
import platform
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tiny_perceptron.multimodal import log_mel, tone


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_default_device("cpu")
wave = tone()
length, size, step = wave.numel(), 400, 160
indices = torch.arange(length).unfold(0, size, step)
frames = wave.unfold(0, size, step)
nonoverlap = wave.unfold(0, size, size)
window = torch.hann_window(size)
starts = indices[:, 0].tolist()
ends = indices[:, -1].tolist()
event_start, event_length = 399, 400
event_end = event_start + event_length - 1
containing = [
    start for start, end in zip(starts, ends, strict=True)
    if start <= event_start and end >= event_end
]
events = torch.zeros_like(wave)
events[event_start:event_end + 1] = 1
event_counts = events.unfold(0, size, step).sum(dim=1).tolist()
short_start, short_length = 399, 100
short_end = short_start + short_length - 1
short_containing = [
    start for start, end in zip(starts, ends, strict=True)
    if start <= short_start and end >= short_end
]

result = {
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "platform": platform.platform(),
        "device": str(wave.device),
        "dtype": str(wave.dtype),
    },
    "repository_inputs": {
        path: digest(ROOT / path)
        for path in ["tiny_perceptron/multimodal.py"]
    },
    "tone": {"samples": length, "sample_rate": 16000, "seconds": 0.1, "frequency_hz": 440},
    "overlap": {
        "shape": list(frames.shape),
        "starts": starts,
        "inclusive_ends": ends,
        "shared_indices": [int(indices[1, 0]), int(indices[0, -1])],
        "shared_samples": len(set(indices[0].tolist()) & set(indices[1].tolist())),
        "same_storage": frames.untyped_storage().data_ptr() == wave.untyped_storage().data_ptr(),
        "all_frame_values_match_source": all(
            torch.equal(frame, wave[start:start + size])
            for frame, start in zip(frames, starts, strict=True)
        ),
        "equal_shared_slices": torch.equal(frames[0, 160:], frames[1, :240]),
        "uncovered_tail": length - 1 - ends[-1],
        "frame_seconds_exact": str(Fraction(size, 16000)),
        "frame_seconds_float": size / 16000,
        "hop_seconds_exact": str(Fraction(step, 16000)),
        "hop_seconds_float": step / 16000,
    },
    "nonoverlap": {
        "shape": list(nonoverlap.shape),
        "starts": list(range(0, length - size + 1, size)),
        "shared_samples": len(set(range(400)) & set(range(400, 800))),
        "exercise_equal": torch.equal(nonoverlap[0, 160:], nonoverlap[1, :240]),
        "slice_max_abs_difference": (nonoverlap[0, 160:] - nonoverlap[1, :240]).abs().max().item(),
    },
    "boundary_conventions": {
        "unfold_shape": list(frames.shape),
        "zero_pad_200_each_side_unfold_shape": list(F.pad(wave, (200, 200)).unfold(0, size, step).shape),
        "stft_center_false_shape": list(torch.stft(wave, size, step, window=window, center=False, return_complex=True).shape),
        "stft_center_true_shape": list(torch.stft(wave, size, step, window=window, center=True, return_complex=True).shape),
        "repository_log_mel_shape": list(log_mel(wave).shape),
    },
    "event_coverage": {
        "frame_length": size,
        "hop_length": step,
        "overlap_length": size - step,
        "counterexample": {
            "event_inclusive_range": [event_start, event_end],
            "event_samples": event_length,
            "fully_containing_frame_starts": containing,
            "event_samples_in_each_frame": event_counts,
        },
        "short_event_example": {
            "event_inclusive_range": [short_start, short_end],
            "event_samples": short_length,
            "fully_containing_frame_starts": short_containing,
        },
        "interpretation": "重疊可讓部分短事件在鄰框完整出現，但不保證所有落在框邊的事件都會完整落入某框。",
    },
}
print(json.dumps(result, ensure_ascii=False, indent=2))
