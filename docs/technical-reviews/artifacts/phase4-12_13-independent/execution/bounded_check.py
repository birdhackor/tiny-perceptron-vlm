"""Independent, bounded CPU checks of the 12.13 representation example."""
import hashlib
import json
import platform
from fractions import Fraction
from pathlib import Path

import torch

from tiny_perceptron.multimodal import log_mel, tone
from tiny_perceptron.natural_concepts import audio_order_report

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
wave = torch.cat((tone(440, seconds=0.1), tone(880, seconds=0.1)))
mel = log_mel(wave)
sequence = mel.transpose(-1, -2)
reversed_frames = sequence.flip(0)
summary = sequence.mean(0)
reversed_summary = reversed_frames.mean(0)
delta = (summary - reversed_summary).abs()
tolerance = 1e-8 + 1e-5 * reversed_summary.abs()
assert tuple(wave.shape) == (3200,)
assert tuple(mel.shape) == (16, 21)
assert tuple(sequence.shape) == (21, 16)
assert not torch.equal(sequence, reversed_frames)
assert torch.all(delta <= tolerance)
assert torch.allclose(summary, reversed_summary)
assert torch.equal(reversed_frames.flip(0), sequence)
assert torch.equal(reversed_frames, sequence[torch.arange(20, -1, -1)])

# A second permutation tests the mechanism beyond reversing the same example.
permutation = torch.randperm(21, generator=torch.Generator().manual_seed(1213))
permuted = sequence[permutation]
assert torch.allclose(summary, permuted.mean(0))

# A frequency-axis reversal is a different operation; mean(0) retains bands.
band_reversed = sequence.flip(1)
assert torch.allclose(band_reversed.mean(0), summary.flip(0))
assert not torch.allclose(band_reversed.mean(0), summary)

# Recompute STFT only for two short generated waveforms, not a trained model.
wave_reversed_features = log_mel(wave.flip(0)).transpose(-1, -2)
swapped_wave = torch.cat((tone(880, seconds=0.1), tone(440, seconds=0.1)))
swapped_wave_features = log_mel(swapped_wave).transpose(-1, -2)

# Exact arithmetic makes the information-loss argument independent of roundoff.
toy = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]
toy_reversed = list(reversed(toy))
toy_mean = [sum(row[b] for row in toy) / len(toy) for b in range(2)]
toy_reversed_mean = [sum(row[b] for row in toy_reversed) / len(toy) for b in range(2)]
assert toy_mean == toy_reversed_mean == [Fraction(1, 2), Fraction(1, 2)]
assert toy[0] != toy_reversed[0]

result = {
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git": str(torch.version.git_version),
        "device": str(wave.device),
        "dtype": str(wave.dtype),
        "cuda_build": str(torch.version.cuda),
        "cpu_threads": str(torch.get_num_threads()),
    },
    "parameters": {"frequencies_hz": [440, 880], "seconds_per_tone": 0.1,
                   "sample_rate_hz": 16000, "n_fft_samples": 400,
                   "hop_length_samples": 160, "bands": 16,
                   "stft_center": True, "stft_pad_mode": "reflect"},
    "measurements": {
        "wave_samples": wave.numel(), "wave_seconds": wave.numel() / 16000,
        "mel_shape_band_time": list(mel.shape),
        "sequence_shape_time_band": list(sequence.shape),
        "summary_shape_band": list(summary.shape), "mean_denominator_frames": 21,
        "same_time_sequence": bool(torch.equal(sequence, reversed_frames)),
        "same_mean_default_allclose": bool(torch.allclose(summary, reversed_summary)),
        "same_mean_exact_float_equality": bool(torch.equal(summary, reversed_summary)),
        "max_absolute_mean_roundoff": delta.max().item(),
        "max_tolerance_fraction": (delta / tolerance).max().item(),
        "allclose_rtol": 1e-5, "allclose_atol": 1e-8,
        "frame_permutation_mean_allclose": bool(torch.allclose(summary, permuted.mean(0))),
        "band_reversal_changes_summary": not bool(torch.allclose(summary, band_reversed.mean(0))),
        "wave_flip_is_exact_frame_flip": bool(torch.equal(wave_reversed_features, reversed_frames)),
        "wave_flip_vs_frame_flip_max_difference": (wave_reversed_features - reversed_frames).abs().max().item(),
        "resynthesized_swap_is_exact_frame_flip": bool(torch.equal(swapped_wave_features, reversed_frames)),
        "resynthesized_swap_vs_frame_flip_max_difference": (swapped_wave_features - reversed_frames).abs().max().item(),
        "requires_grad": [bool(x.requires_grad) for x in [wave, sequence, summary]],
    },
    "samples": {"permutation_indices": permutation.tolist(),
                "mean_band_values": summary.tolist(),
                "reversed_mean_band_values": reversed_summary.tolist(),
                "exact_toy_mean": [str(x) for x in toy_mean],
                "exact_toy_reversed_mean": [str(x) for x in toy_reversed_mean]},
    "original_helper_return": audio_order_report(),
    "provenance": {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in ["tiny_perceptron/multimodal.py", "tiny_perceptron/natural_concepts.py"]
    },
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
