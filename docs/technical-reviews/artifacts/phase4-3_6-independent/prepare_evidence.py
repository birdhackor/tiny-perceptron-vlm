"""Preserve this review's original bytes and immutable official sources."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

(OUT / "original").mkdir(exist_ok=True)
temporary = Path("/tmp/phase4-3_6-original")
for path in temporary.rglob("*"):
    if "workspace" not in path.relative_to(temporary).parts and path.is_file():
        target = OUT / "original" / path.relative_to(temporary)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
for relative in ["tiny_perceptron/attention.py", "docs/review-tools/section_facts.py", "scripts/build_course.py"]:
    target = OUT / "code" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relative, target)

source_specs = [
    ("phase4-3_1-independent/sources/vaswani-2017-v7.pdf", "vaswani-2017-v7.pdf", "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697", "https://arxiv.org/pdf/1706.03762v7", "arXiv:1706.03762v7; 2023-08-02"),
    ("phase4-3_5-independent/sources/torch-docs-installed-commit.py", "torch-docs-installed-commit.py", "f3962a2a4973e7d4917a3b3f265903d4a534085343bbe40d003728385d1949e5", "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_torch_docs.py", "5c4886908584029761b579af026dcfb627c84070"),
    ("phase4-3_5-independent/sources/tensor-docs-installed-commit.py", "tensor-docs-installed-commit.py", "9c3d4d0bcef8041352cc87cff68980c39ddd8c506d4e8db6410af306bd46f189", "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_tensor_docs.py", "5c4886908584029761b579af026dcfb627c84070"),
    ("phase4-3_5-independent/sources/functional-installed-commit.py", "functional-installed-commit.py", "95ff403085bb179477a01df63acfddf59dfac0fb58829e99a9c8b5c2fe98899b", "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py", "5c4886908584029761b579af026dcfb627c84070"),
]
(OUT / "sources").mkdir(exist_ok=True)
receipts = []
for relative, name, expected, url, version in source_specs:
    original = ROOT / "docs/technical-reviews/artifacts" / relative
    observed = digest(original)
    assert observed == expected, (relative, observed)
    target = OUT / "sources" / name
    shutil.copyfile(original, target)
    receipts.append({"url": url, "version": version, "accessed_on": "2026-10-05", "origin": original.relative_to(ROOT).as_posix(), "path": target.relative_to(ROOT).as_posix(), "sha256": digest(target), "bytes": target.stat().st_size, "retrieval": "Reused immutable original source; independently verified SHA-256 against acquisition receipt. Read source itself, not another review."})
(OUT / "sources/receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
command = ["pdftotext", "-layout", str(OUT / "sources/vaswani-2017-v7.pdf"), str(OUT / "sources/vaswani-2017-v7.txt")]
result = subprocess.run(command, capture_output=True, text=True, check=True)
(OUT / "sources/pdf-extraction.json").write_text(json.dumps({"command_argv": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "pdf_sha256": digest(OUT / "sources/vaswani-2017-v7.pdf"), "text_sha256": digest(OUT / "sources/vaswani-2017-v7.txt")}, indent=2) + "\n")
print(json.dumps({"source_count": len(receipts), "original_source_sha256": digest(OUT / "original/section.md"), "sources": receipts}, ensure_ascii=False, indent=2))
