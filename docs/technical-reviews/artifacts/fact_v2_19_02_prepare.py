"""Freeze independently read inputs and fetch the primary sources for 19.2."""

import hashlib
import json
import platform
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_19_02_"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    path = OUT / (PREFIX + name)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


records = []
for filename, ids in [("19.md", {"19.2", "19.10"}), ("15.md", {"15.2", "15.10", "15.11", "15.13"})]:
    original = ROOT / "course/chapters" / filename
    for lesson, body in sections(original):
        if lesson in ids:
            frozen = OUT / (PREFIX + "section_" + lesson.replace(".", "_") + ".md")
            frozen.write_text(body, encoding="utf-8")
            records.append({"path": str(original.relative_to(ROOT)), "lesson": lesson, "sha256": sha(frozen),
                            "snapshot": str(frozen.relative_to(ROOT)), "read": "Complete original section read before audit."})
for rel in ["docs/technical-review-guide.md", "course/figures/capstone_resources.svg",
            "tiny_perceptron/capstone.py", "tiny_perceptron/modern.py", "tiny_perceptron/model.py",
            "tiny_perceptron/data.py", "tiny_perceptron/attention.py",
            "scripts/course_experiments/capstone.py", "scripts/course_experiments/capstone_deployment.py"]:
    path = ROOT / rel
    snapshot = OUT / (PREFIX + rel.replace("/", "_") + ".txt")
    snapshot.write_bytes(path.read_bytes())
    records.append({"path": rel, "sha256": sha(path), "snapshot": str(snapshot.relative_to(ROOT))})
save("read_manifest.json", {"reviewer_task": "/root/integration_technical_coordinator/fact_v2_19_02",
                            "recorded_at": datetime.now(UTC).isoformat(), "inputs": records})

fetches = []
sources = [
    ("mixtral_v1.pdf", "https://arxiv.org/pdf/2401.04088v1"),
    ("switch_jmlr.pdf", "https://jmlr.org/papers/volume23/21-0998/21-0998.pdf"),
    ("rfc3629.txt", "https://www.rfc-editor.org/rfc/rfc3629.txt"),
    ("torch_module.py.txt", f"https://raw.githubusercontent.com/pytorch/pytorch/{torch.version.git_version}/torch/nn/modules/module.py"),
    ("torch_adam.py.txt", f"https://raw.githubusercontent.com/pytorch/pytorch/{torch.version.git_version}/torch/optim/adam.py"),
]
for name, url in sources:
    receipt = {"url": url, "started_at": datetime.now(UTC).isoformat()}
    target = OUT / (PREFIX + name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "independent-course-review/19.2"})
        with urllib.request.urlopen(request, timeout=40) as response:
            payload = response.read()
            receipt.update(status=response.status, final_url=response.url, content_type=response.headers.get("Content-Type"))
        target.write_bytes(payload)
        receipt.update(path=str(target.relative_to(ROOT)), bytes=len(payload), sha256=sha(target))
        if name.endswith(".pdf"):
            text = target.with_suffix(".txt")
            result = subprocess.run(["pdftotext", "-layout", str(target), str(text)], capture_output=True, text=True, check=False)
            receipt.update(pdftotext_exit=result.returncode, pdftotext_stderr=result.stderr,
                           text_path=str(text.relative_to(ROOT)), text_sha256=sha(text))
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
    fetches.append(receipt)
    save("fetch_result.json", {"command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_02_prepare.py",
                              "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
                              "sources": fetches})
print(json.dumps(fetches, ensure_ascii=False, indent=2))
sys.exit(0 if all("sha256" in item for item in fetches) else 1)
