"""核對浮點容許差的修訂與可沿用的原始證據。"""

import difflib
import hashlib
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
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.modal_data import modal_example
from tiny_perceptron.multimodal import tone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv):
    completed = subprocess.run(argv, cwd=ROOT, capture_output=True)
    return {"argv": argv, "exit_code": completed.returncode,
            "stdout": completed.stdout.decode(), "stderr": completed.stderr.decode()}


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_12-2/technical-12-2-original-01.json"
original = json.loads(original_path.read_bytes())
old_path = ROOT / "docs/course-review-20261019/freeze-01/sources/12.2.md"
new_path = ROOT / "docs/course-review-20261019/freeze-05/sources/12.2.md"
manifest_path = ROOT / "docs/course-review-20261019/freeze-06.json"
old = old_path.read_bytes()
new = new_path.read_bytes()
live = (ROOT / "course/chapters/12.md").read_bytes()
section = re.search(rb"(?ms)^## 12\.2 .*?(?=^## |\Z)", live)[0]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-2/verify-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version),
                    "numpy": str(np.__version__), "device": "cpu", "platform": platform.platform()},
    "checked_at_utc": datetime.now(UTC).isoformat(),
    "original_report_sha256": sha(original_path),
    "source": {"old_sha256": sha(old_path), "new_sha256": sha(new_path),
               "live_sha256": hashlib.sha256(section).hexdigest(), "live_matches_freeze": section == new,
               "manifest_sha256": sha(manifest_path)},
    "section_diff": command(["git", "diff", "--no-index", str(old_path.relative_to(ROOT)), str(new_path.relative_to(ROOT))]),
    "training_diff": command(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"]),
    "repository_revision": command(["git", "rev-parse", "HEAD"])["stdout"].strip(),
}
assert result["original_report_sha256"] == "c5d2a80aa7e81a5cd42fdab9829652998e512aaf82d4f58075915c0ef661413d"
assert result["source"]["new_sha256"] == "05eb1ea9b9088d0f2496f59613bdbb461133c76e9a05819f509d1c6c63162945"
assert result["source"]["live_matches_freeze"]
assert result["source"]["manifest_sha256"] == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
result["inherited_evidence"] = []
for group in ("sources", "artifacts"):
    for item in original[group]:
        if "path" not in item:
            continue
        actual = sha(ROOT / item["path"])
        result["inherited_evidence"].append({"group": group, "id": item["id"], "path": item["path"],
                                             "recorded_sha256": item["sha256"], "current_sha256": actual,
                                             "unchanged": actual == item["sha256"]})
        if item["id"] != "training_doc":
            assert actual == item["sha256"], item["path"]

training = (ROOT / "course/training.md").read_text()
previous_training = command(["git", "show", "5ef56bc756f3:course/training.md"])["stdout"]
def audio_paragraph(text):
    return next(p for p in text.split("\n\n") if p.startswith("自己的圖像紀錄可寫成"))
result["training_audio_scope"] = {
    "path": "course/training.md", "current_sha256": sha(ROOT / "course/training.md"),
    "quoted_paragraph": audio_paragraph(training),
    "unchanged_relevant_paragraph": audio_paragraph(training) == audio_paragraph(previous_training),
    "prior_version_full_sha256": hashlib.sha256(previous_training.encode()).hexdigest(),
}
assert result["training_audio_scope"]["unchanged_relevant_paragraph"]
assert result["training_audio_scope"]["current_sha256"] == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"

run_path = ROOT / "outputs/reviewer-tools/runs/20261020T101004503547Z"
result["section_run"] = {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/12.md#12.2" --execute',
                         "copied_from": str(run_path.relative_to(ROOT)), "files": {}}
for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json"):
    raw = (run_path / name).read_bytes()
    result["section_run"]["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                                             "content": json.loads(raw) if name.endswith(".json") else raw.decode()}
assert (run_path / "section.md").read_bytes() == new
assert result["section_run"]["files"]["execution.json"]["content"]["exit_code"] == 0

result["arithmetic"] = {"duration": [1600/16000, 1600/8000],
                        "samples_per_cycle": [round(16000/440, 2), round(8000/440, 2)],
                        "nyquist": [16000/2, 8000/2], "relabeled_frequency": 440*8000/16000,
                        "recording_duration": [4591/8000, 9182/16000], "exercise_samples": 8000*16000//8000}
torch.set_num_threads(1)
result["pointwise_checks"] = []
for high_hz, low_hz in ((9000.0, 7000.0), (9500.0, 6500.0)):
    high = tone(high_hz, seconds=0.001)
    low = tone(low_hz, seconds=0.001)
    other = -low
    difference = (high-other).abs()
    threshold = 1e-5+1e-5*other.abs()
    explicit = bool((difference <= threshold).all())
    observed = torch.allclose(high, other, atol=1e-5)
    result["pointwise_checks"].append({
        "frequencies_hz": [high_hz, low_hz], "samples": len(high), "dtype": str(high.dtype),
        "high_first_four": [round(v, 3) for v in high[:4].tolist()],
        "low_first_four": [round(v, 3) for v in low[:4].tolist()],
        "formula": "abs(high-other) <= 1e-5 + 1e-5*abs(other), other=-low",
        "absolute_differences": difference.tolist(), "thresholds": threshold.tolist(),
        "explicit_all": explicit, "torch_allclose": observed, "matches": explicit == observed,
    })
    assert explicit == observed and observed
result["threshold_examples"] = []
for value, other_value in ((0.5000125, 0.5), (0.50002, 0.5), (-0.5000125, -0.5), (-0.50002, -0.5)):
    left = torch.tensor([value], dtype=torch.float64)
    right = torch.tensor([other_value], dtype=torch.float64)
    explicit = bool(((left-right).abs() <= 1e-5+1e-5*right.abs()).all())
    observed = torch.allclose(left, right, atol=1e-5)
    result["threshold_examples"].append({"input": value, "other": other_value,
                                         "absolute_difference": float((left-right).abs().max()),
                                         "threshold": float((1e-5+1e-5*right.abs()).max()),
                                         "explicit": explicit, "allclose": observed})
    assert explicit == observed

result["input_validation"] = []
for case, values, rate in (("有效單聲道", np.zeros(16, dtype=np.float32), 16000),
                           ("錯誤取樣率", np.zeros(16, dtype=np.float32), 8000),
                           ("空錄音", np.zeros(0, dtype=np.float32), 16000),
                           ("多聲道", np.zeros((16, 2), dtype=np.float32), 16000)):
    with patch("tiny_perceptron.modal_data.sf.read", return_value=(values, rate)):
        try:
            modal_example({"audio": "voice.wav", "question": "digit?", "answer": "0"}, ROOT, "audio")
            observed = "接受"
        except ValueError as error:
            observed = str(error)
    result["input_validation"].append({"case": case, "shape": list(values.shape), "rate": rate, "result": observed})

url = "https://docs.pytorch.org/docs/2.14/generated/torch.allclose.html"
argv = ["curl", "-fLsS", "--max-time", "40", url]
response = subprocess.run(argv, capture_output=True, check=True).stdout
parser = Text()
parser.feed(response.decode())
text = re.sub(r"\s+", " ", " ".join(parser.parts))
excerpt = "rtol ( float , optional ) – relative tolerance. Default: 1e-05"
assert excerpt in text
result["official_source"] = {"url": url, "version": "PyTorch 2.14 官方 API 文件，對應本機 2.14.1",
                              "command": argv, "accessed_on": datetime.now(UTC).date().isoformat(),
                              "response_sha256": hashlib.sha256(response).hexdigest(),
                              "locator": "torch.allclose signature / elementwise inequality / rtol parameter",
                              "excerpt": excerpt, "excerpt_found": True,
                              "inspected_formula": "abs(input_i-other_i) <= atol + rtol*abs(other_i)"}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
