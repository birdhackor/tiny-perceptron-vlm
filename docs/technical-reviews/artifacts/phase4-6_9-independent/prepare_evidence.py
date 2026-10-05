"""Preserve this review's original bytes and acquire primary sources only."""
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

copies = [
    "docs/review-tools/factual-reviewer-instructions.md",
    "docs/review-tools/section_facts.py",
    "scripts/check_technical_reviews.py",
    ".agents/skills/clear-tutorial/references/review-protocol.md",
    "scripts/build_course.py", "pyproject.toml",
    "tiny_perceptron/data.py", "tiny_perceptron/tokenization.py",
    "scripts/course_experiments/text.py", "scripts/course_experiments/common.py",
    "docs/course-experiments/results/tokenizer.json",
]
copy_manifest = []
for relative in copies:
    src, dst = ROOT / relative, OUT / "inputs/current" / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    copy_manifest.append({"input": relative, "snapshot": str(dst.relative_to(ROOT)), "sha256": sha(dst)})
run = Path("/tmp/phase4-6_9-independent-original")
(OUT / "original-run").mkdir(exist_ok=True)
for src in run.iterdir():
    if src.is_file():
        shutil.copyfile(src, OUT / "original-run" / src.name)

result = json.loads((ROOT / "docs/course-experiments/results/tokenizer.json").read_bytes())
revision = result["revision"]
history_manifest = []
for relative in ["scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "tiny_perceptron/data.py"]:
    raw = subprocess.run(["git", "show", f"{revision}:{relative}"], cwd=ROOT, capture_output=True, check=True).stdout
    dst = OUT / "inputs/historical" / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(raw)
    history_manifest.append({"revision": revision, "path": relative, "snapshot": str(dst.relative_to(ROOT)), "sha256": sha(dst), "expected_sha256": result["code_sha256"][relative], "matches": sha(dst) == result["code_sha256"][relative]})

base = ROOT / "outputs/text-behavior-interface-check/tokenizer"
artifact_manifest = []
for relative in ["tokenizer-byte.json", "tokenizer-bpe512.json", "data/manifest.json", "data/train.jsonl", "data/validation.jsonl", "data/test.jsonl"]:
    src, dst = base / relative, OUT / "inputs/original-tokenizer" / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    expected = next(a for a in result["artifacts"] if a["path"] == relative)
    artifact_manifest.append({"input": str(src.relative_to(ROOT)), "snapshot": str(dst.relative_to(ROOT)), "sha256": sha(dst), "expected_sha256": expected["sha256"], "matches": sha(dst) == expected["sha256"], "bytes": dst.stat().st_size})

sources = {
    "hf-whitespace.rs": "https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/tokenizers/src/pre_tokenizers/whitespace.rs",
    "hf-byte_level.rs": "https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/tokenizers/src/pre_tokenizers/byte_level.rs",
    "hf-bpe-model.rs": "https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/tokenizers/src/models/bpe/model.rs",
    "hf-bpe-trainer.rs": "https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/tokenizers/src/models/bpe/trainer.rs",
    "hf-tokenizer.rs": "https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/tokenizers/src/tokenizer/mod.rs",
    "hf-python-tokenizer.rs": "https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/bindings/python/src/tokenizer.rs",
    "python-codecs.rst": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/codecs.rst",
    "python-io.rst": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/io.rst",
}
(OUT / "sources").mkdir(exist_ok=True)
fetches = []
for name, url in sources.items():
    item = {"url": url, "accessed_at": datetime.now(timezone.utc).isoformat(), "snapshot": str((OUT / "sources" / name).relative_to(ROOT))}
    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            raw = response.read(500_001)
            if len(raw) > 500_000:
                raise ValueError("primary source exceeds acquisition bound")
            item.update(status=response.status, resolved_url=response.url)
        target = OUT / "sources" / name
        target.write_bytes(raw)
        item.update(bytes=len(raw), sha256=sha(target))
    except Exception as exc:
        item.update(error=repr(exc))
    fetches.append(item)
    print(json.dumps(item, ensure_ascii=False))
write(OUT / "provenance.json", {"current_git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(), "copies": copy_manifest, "historical_code": history_manifest, "original_artifacts": artifact_manifest, "sources": fetches, "scope": "Read-only provenance checks; no weights copied or loaded, no datasets generated, no GPU training or model inference."})
