"""Independent CPU measurements for the 12.13 representation example."""
import inspect
import json
import math
import platform
import torch
from tiny_perceptron.multimodal import log_mel, mel_filter_bank, tone
from tiny_perceptron.natural_concepts import audio_order_report

torch.set_num_threads(1)
torch.set_default_device("cpu")
sample_rate = 16000
segments = [tone(440, seconds=0.1), tone(880, seconds=0.1)]
waveform = torch.cat(segments)
spectrogram = log_mel(waveform)
sequence = spectrogram.transpose(-1, -2)
reordered = sequence.flip(0)
first_mean = sequence.mean(0)
second_mean = reordered.mean(0)
reversed_waveform_sequence = log_mel(waveform.flip(0)).transpose(-1, -2)
mean_difference = (first_mean - second_mean).abs()
tolerance = 1e-8 + 1e-5 * second_mean.abs()
print(json.dumps({
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": str(waveform.device), "dtype": str(waveform.dtype),
                    "threads": torch.get_num_threads(), "cuda_build": str(torch.version.cuda)},
    "tone_signature": str(inspect.signature(tone)),
    "log_mel_signature": str(inspect.signature(log_mel)),
    "frequencies_hz": [440, 880], "segment_seconds": [0.1, 0.1],
    "segment_samples": [len(x) for x in segments],
    "total_samples": len(waveform), "duration_seconds": len(waveform)/sample_rate,
    "band_bank_shape": list(mel_filter_bank().shape),
    "individual_spectrogram_shapes": [list(log_mel(x).shape) for x in segments],
    "concatenated_spectrogram_shape": list(spectrogram.shape),
    "sequence_shape": list(sequence.shape), "mean_shape": list(first_mean.shape),
    "report": audio_order_report(),
    "permutation_is_reversal": bool(torch.equal(reordered[0], sequence[-1])),
    "same_time_sequence": bool(torch.equal(sequence, reordered)),
    "means_exactly_equal_in_this_run": bool(torch.equal(first_mean, second_mean)),
    "means_default_allclose": bool(torch.allclose(first_mean, second_mean)),
    "default_allclose_rtol": 1e-5, "default_allclose_atol": 1e-8,
    "allclose_elementwise_bound_satisfied": bool((mean_difference <= tolerance).all()),
    "mean_max_absolute_difference": float(mean_difference.max()),
    "mean_differences": mean_difference.tolist(),
    "mean_values_original": first_mean.tolist(), "mean_values_reordered": second_mean.tolist(),
    "reversed_waveform_shape": list(reversed_waveform_sequence.shape),
    "reversed_waveform_recomputed_equals_reordered_features": bool(torch.equal(reversed_waveform_sequence, reordered)),
    "reversed_waveform_recomputed_allclose_reordered_features": bool(torch.allclose(reversed_waveform_sequence, reordered)),
    "reversed_waveform_recomputed_max_difference": float((reversed_waveform_sequence-reordered).abs().max()),
    "frame_count_derivation": "N=round(0.1*16000)*2=3200; center=True gives 200 reflect samples per side; floor((3200+400-400)/160)+1=21. One segment gives 11. 16 mel filters each combine the 400//2+1=201 one-sided FFT bins.",
}, ensure_ascii=False, indent=2))
