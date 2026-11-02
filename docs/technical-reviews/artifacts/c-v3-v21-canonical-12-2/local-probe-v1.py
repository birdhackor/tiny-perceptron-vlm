import hashlib, json, platform, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from tiny_perceptron.multimodal import tone
from tiny_perceptron.modal_data import modal_example

torch.set_num_threads(1)
root = Path.cwd()
result = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "soundfile": sf.__version__, "device": "cpu", "dtype": str(torch.get_default_dtype()), "torch_threads": str(torch.get_num_threads())},
    "scope": "Local CPU examples and decoder-stub input-contract branch checks only; no training, recorded-audio reconstruction, GPU claim or resampling-quality measurement.",
    "sample_rates": [{"samples": 1600, "rate": rate, "duration": 1600 / rate, "nyquist": rate / 2, "samples_per_440hz_cycle": round(rate / 440, 2)} for rate in [16000, 8000]],
    "tone_pairs": [],
    "input_contract": [],
    "allclose_docstring": torch.allclose.__doc__,
}
for high_frequency, low_frequency in [(9000.0, 7000.0), (9500.0, 6500.0)]:
    high = tone(high_frequency, seconds=0.001)
    low = tone(low_frequency, seconds=0.001)
    equal = bool(torch.allclose(high, -low, atol=1e-5))
    maximum_error = float((high + low).abs().max())
    result["tone_pairs"].append({"high_hz": high_frequency, "low_hz": low_frequency, "shape": list(high.shape), "dtype": str(high.dtype), "high_first4": [round(v, 3) for v in high[:4].tolist()], "low_first4": [round(v, 3) for v in low[:4].tolist()], "allclose_high_negative_low": equal, "max_abs_high_plus_low": maximum_error, "manual_allclose_bound_satisfied": bool(torch.all((high + low).abs() <= 1e-5 + 1e-5 * low.abs()))})
    assert high.shape == low.shape == (16,)
    assert equal
assert result["tone_pairs"][0]["high_first4"] == [0.0, -0.191, 0.354, -0.462]
assert result["tone_pairs"][0]["low_first4"] == [0.0, 0.191, -0.354, 0.462]
for name, values, rate, expect in [
    ("mono_16000", np.zeros(16, dtype=np.float32), 16000, "accepted"),
    ("mono_8000", np.zeros(16, dtype=np.float32), 8000, "rejected"),
    ("stereo_16000", np.zeros((16, 2), dtype=np.float32), 16000, "rejected"),
    ("empty_16000", np.zeros(0, dtype=np.float32), 16000, "rejected"),
]:
    with patch("tiny_perceptron.modal_data.sf.read", return_value=(values, rate)):
        try:
            _, _, _, waveform = modal_example({"audio": "decoder-stub.wav", "question": "pitch?"}, root, "audio", include_answer=False)
            actual = "accepted"
            observation = {"waveform_shape": list(waveform.shape), "waveform_dtype": str(waveform.dtype)}
        except ValueError as error:
            actual = "rejected"
            observation = {"message": str(error)}
    assert actual == expect
    result["input_contract"].append({"name": name, "decoder_shape": list(values.shape), "decoder_rate": rate, "expected": expect, "observed": actual, **observation})
p = root / "docs/course-experiments/results/real_modal.json"
raw = json.loads(p.read_text())
row = raw["results"]["fsdd"]["resampling"][0]
result["fixed_raw_record"] = {"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "locator": "/results/fsdd/resampling/0", "record": row, "duration_before": row["samples_before"] / row["source_rate"], "duration_after": row["samples_after"] / row["target_rate"]}
result["source_files"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ["tiny_perceptron/multimodal.py", "tiny_perceptron/modal_data.py", "scripts/audio_utils.py", "course/chapters/19.md", "course/training.md"]}
print(json.dumps(result, ensure_ascii=False, indent=2))
