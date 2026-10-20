"""核對取樣算例、浮點比較契約、原始錄音與輸入限制。"""

import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.course_experiments.modalities import _resample_8_to_16
from tiny_perceptron.modal_data import modal_example
from tiny_perceptron.multimodal import tone


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url):
    command = ["curl", "-fLsS", "--max-time", "40", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    return completed.stdout, command


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag == "img" and not self.hidden:
            self.parts.extend(value for key, value in attrs if key == "alt")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-2/verify.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "numpy": str(np.__version__),
        "soundfile": str(sf.__version__),
        "libsndfile": str(sf.__libsndfile_version__),
        "device": "cpu",
        "platform": platform.platform(),
    },
    "retrieved_at_utc": datetime.now(UTC).isoformat(),
}
torch.set_num_threads(1)
frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/12.2.md"
live = (ROOT / "course/chapters/12.md").read_bytes()
section = re.search(rb"(?ms)^## 12\.2 .*?(?=^## |\Z)", live)[0]
result["freeze"] = {
    "source_sha256": digest(frozen.read_bytes()),
    "live_section_sha256": digest(section),
    "identical": section == frozen.read_bytes(),
    "freeze_manifest_sha256": digest((ROOT / "docs/course-review-20261019/freeze-04.json").read_bytes()),
    "image_references": re.findall(r"!\[[^\]]*\]\(([^)]+)\)", section.decode()),
}
result["arithmetic"] = {
    "duration": [1600 / 16000, 1600 / 8000],
    "samples_per_cycle": [round(16000 / 440, 2), round(8000 / 440, 2)],
    "nyquist": [16000 / 2, 8000 / 2],
    "relabeled_frequency": 440 * 8000 / 16000,
    "resampled_duration": [4591 / 8000, 9182 / 16000],
    "exercise_samples": 8000 * 16000 // 8000,
}
result["aliases"] = []
for high_hz, low_hz in [(9000.0, 7000.0), (9500.0, 6500.0)]:
    high, low = tone(high_hz, seconds=0.001), tone(low_hz, seconds=0.001)
    result["aliases"].append({
        "frequencies_hz": [high_hz, low_hz],
        "samples": len(high),
        "dtype": str(high.dtype),
        "high_first_four": [round(v, 3) for v in high[:4].tolist()],
        "low_first_four": [round(v, 3) for v in low[:4].tolist()],
        "max_absolute_difference_from_negative_low": float((high + low).abs().max()),
        "allclose_default_rtol": torch.allclose(high, -low, atol=1e-5),
        "allclose_zero_rtol": torch.allclose(high, -low, atol=1e-5, rtol=0),
    })
left = torch.tensor([0.5000125], dtype=torch.float64)
right = torch.tensor([0.5], dtype=torch.float64)
result["allclose_counterexample"] = {
    "input": left.tolist(),
    "other": right.tolist(),
    "absolute_difference": float((left - right).abs().max()),
    "atol": 1e-5,
    "default_rtol": 1e-5,
    "combined_tolerance": 1e-5 + 1e-5 * 0.5,
    "default_result": torch.allclose(left, right, atol=1e-5),
    "zero_rtol_result": torch.allclose(left, right, atol=1e-5, rtol=0),
}

version = "26eb9aaf76e81b692f806f9140c2d2777410d7a1"
url = f"https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/{version}/recordings/0_jackson_5.wav"
raw, command = fetch(url)
wave, rate = sf.read(io.BytesIO(raw), dtype="float32")
converted = _resample_8_to_16(wave)
buffer = io.BytesIO()
sf.write(buffer, converted, 16000, format="WAV", subtype="FLOAT")
record = json.loads((ROOT / "docs/course-experiments/results/real_modal.json").read_bytes())["results"]["fsdd"]["resampling"][0]
result["recording"] = {
    "url": url,
    "download_command": command,
    "source_sha256": digest(raw),
    "matches_recorded_source": digest(raw) == record["source_sha256"],
    "source_rate": rate,
    "source_shape": list(wave.shape),
    "samples_before": len(wave),
    "samples_after": len(converted),
    "duration_before": len(wave) / rate,
    "duration_after": len(converted) / 16000,
    "source_peak_absolute_amplitude": float(np.abs(wave).max()),
    "derived_sha256": digest(buffer.getvalue()),
    "matches_recorded_derived": digest(buffer.getvalue()) == record["derived_sha256"],
    "record": record,
    "scope": "只重做這一筆錄音的重取樣，未重新訓練模型。",
}

result["input_validation"] = []
for name, values, rate in [
    ("有效單聲道", np.zeros(16, dtype=np.float32), 16000),
    ("錯誤取樣率", np.zeros(16, dtype=np.float32), 8000),
    ("空錄音", np.zeros(0, dtype=np.float32), 16000),
    ("多聲道", np.zeros((16, 2), dtype=np.float32), 16000),
]:
    with patch("tiny_perceptron.modal_data.sf.read", return_value=(values, rate)):
        try:
            modal_example({"audio": "voice.wav", "question": "digit?", "answer": "0"}, ROOT, "audio")
            observed = "接受"
        except ValueError as error:
            observed = str(error)
    result["input_validation"].append({"case": name, "shape": list(values.shape), "rate": rate, "result": observed})

result["wav_metadata"] = []
for rate, amplitude in [(8000, 0.1), (16000, 0.5)]:
    buffer = io.BytesIO()
    sf.write(buffer, np.full(16, amplitude), rate, format="WAV", subtype="FLOAT")
    values, observed_rate = sf.read(io.BytesIO(buffer.getvalue()), dtype="float32")
    result["wav_metadata"].append({"extension": ".wav", "rate": observed_rate, "peak": float(np.abs(values).max())})

specifications = [
    (
        "sampling",
        "https://www.dsprelated.com/freebooks/mDFT/Sampling_Theory.html",
        "Mathematics of the DFT，Sampling Theory，線上 HTML；以本次回應 SHA-256 固定實際讀取版本",
        "Sampling Theory / Introduction to Sampling / Reconstruction from Samples--The Math / Aliasing of Sampled Signals / Sampling Theorem",
        ["sampling rate in samples-per-second (Hz)", "f_a = f_s/2 - \\epsilon"],
    ),
    (
        "antialias",
        "https://www.dsprelated.com/freebooks/mdft/Alias_Operator.html",
        "Mathematics of the DFT，Alias Operator，線上 HTML；以本次回應 SHA-256 固定實際讀取版本",
        "Alias Operator，關於不可逆與降低取樣率前的低通處理之段落",
        ["aliasing is not invertible in general.", "an anti-aliasing lowpass filter is generally used."],
    ),
    (
        "allclose",
        "https://docs.pytorch.org/docs/2.14/generated/torch.allclose.html",
        "PyTorch 2.14 官方文件，對應本機 2.14.1",
        "torch.allclose signature / elementwise inequality / rtol parameter",
        ["rtol ( float , optional ) – relative tolerance. Default: 1e-05"],
    ),
    (
        "fsdd",
        f"https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/{version}/README.md",
        version,
        "README / Contributions",
        ["All recordings should be mono 8kHz `wav` files"],
    ),
]
result["original_sources"] = []
for identifier, url, version, locator, excerpts in specifications:
    raw, command = fetch(url)
    parser = Text()
    parser.feed(raw.decode())
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    inspected = []
    for excerpt in excerpts:
        inspected.append({"text": excerpt, "found_in_original": excerpt in text})
    result["original_sources"].append({
        "id": identifier,
        "url": url,
        "version": version,
        "locator": locator,
        "command": command,
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "response_sha256": digest(raw),
        "excerpts": inspected,
    })

print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
