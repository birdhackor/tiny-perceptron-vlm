"""Fresh 5.6 review: preserve original inputs and fetch original authorities only."""
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "outputs/reviewer-tools/phase4-5_6-independent-original"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


records = []
for rel in [
    "docs/review-tools/factual-reviewer-instructions.md",
    "docs/review-tools/section_facts.py",
    "scripts/check_technical_reviews.py",
    ".agents/skills/clear-tutorial/references/review-protocol.md",
    "tiny_perceptron/training.py",
    "scripts/train.py",
    "scripts/build_course.py",
    "course/training.md",
    "course/chapters/05.md",
    "course/figures/rewrite-05-06-learning-rate.svg",
]:
    source = ROOT / rel
    target = OUT / "inputs" / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    records.append({"kind": "repository_input", "original_path": rel,
                    "saved_path": str(target.relative_to(ROOT)), "sha256": sha(target)})
for source in RUN.rglob("*"):
    if source.is_file() and "workspace" not in source.relative_to(RUN).parts:
        target = OUT / "original-run" / source.relative_to(RUN)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append({"kind": "current_original_fence_run", "original_path": str(source.relative_to(ROOT)),
                        "saved_path": str(target.relative_to(ROOT)), "sha256": sha(target)})
write(OUT / "input-provenance.json", {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_5_6",
    "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "original_fence_command": ".venv/bin/python docs/review-tools/section_facts.py 'course/chapters/05.md#5.6' --output outputs/reviewer-tools/phase4-5_6-independent-original --execute --timeout 30",
    "section_bytes_policy": "Unmodified UTF-8 original bytes, extracted by original_section, no newline normalization",
    "read_scope": ["course/chapters/05.md#5.6", "tiny_perceptron/training.py all functions", "scripts/train.py parser, training_metadata, main resume/schedule sections", "scripts/build_course.py BOOTSTRAP", "course/training.md continuation contract only", "four prescribed review-method/schema files"],
    "prior_review_content_read": False,
    "large_weight_or_dataset_inputs": [],
    "records": records,
})
sources = [
    ("sgdr-1608.03983v1.pdf", "https://arxiv.org/pdf/1608.03983v1"),
    ("hf-v4.57.1-optimization.py", "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/optimization.py"),
]
receipts = []
for name, url in sources:
    receipt = {"url": url, "accessed_on": "2026-10-05", "timeout_seconds": 20,
               "requested_snapshot": name, "python": sys.version, "tls_verification": "Python HTTPS default enabled"}
    target = OUT / "sources" / name
    target.parent.mkdir(exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise RuntimeError("Source exceeds 4 MiB cap")
            target.write_bytes(raw)
            receipt.update(status=response.status, final_url=response.url,
                           content_type=response.headers.get("Content-Type"), bytes=len(raw), sha256=sha(target))
        print(name, receipt["status"], receipt["bytes"], receipt["sha256"])
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        print(name, receipt["error_type"], receipt["error"])
    receipts.append(receipt)
write(OUT / "source-acquisition.json", receipts)
write(OUT / "prepare-environment.json", {"python": sys.version, "python_executable": sys.executable,
                                        "platform": platform.platform(), "device": "CPU only"})
