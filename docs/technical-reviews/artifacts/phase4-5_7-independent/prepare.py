"""Freeze only lesson inputs and primary implementation; never read review reports."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/phase4-5_7-independent-original"
spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
metadata = facts.extract("course/chapters/05.md#5.7", OUT)
code = facts.execute(OUT, metadata, timeout=45)
for item in OUT.iterdir():
    if item.is_file():
        destination = HERE / "original" / item.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, destination)

paths = [
    "docs/review-tools/factual-reviewer-instructions.md", "docs/review-tools/section_facts.py",
    "scripts/check_technical_reviews.py", ".agents/skills/clear-tutorial/references/review-protocol.md",
    "tiny_perceptron/training.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py",
    "tiny_perceptron/modern.py", "tiny_perceptron/multimodal.py", "scripts/train.py",
    "scripts/build_course.py", "scripts/course_experiments/text.py", "scripts/course_experiments/common.py",
    "docs/course-experiments/results/text_foundation.json", "pyproject.toml",
]
inputs = []
for name in paths:
    source = ROOT / name
    destination = HERE / "inputs" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    inputs.append({"path": name, "snapshot": str(destination.relative_to(ROOT)),
                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size})
historical = json.loads((ROOT / "docs/course-experiments/results/text_foundation.json").read_text())
for name in ["scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "tiny_perceptron/training.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/data.py"]:
    proc = subprocess.run(["git", "show", f"{historical['revision']}:{name}"], cwd=ROOT, capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode())
    raw = proc.stdout
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == historical["code_sha256"][name], (name, digest)
    target = HERE / "historical" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    inputs.append({"path": name, "git_revision": historical["revision"], "snapshot": str(target.relative_to(ROOT)), "sha256": digest, "bytes": len(raw), "matches_original_result_code_hash": True})
(HERE / "input-provenance.json").write_text(json.dumps({
    "scope": "Original UTF-8 section, CPU original fence, current source files, and original recorded GPU result with its hash-matched implementation. No prior review read; no existing weights copied; no long recipe run.",
    "source": metadata, "inputs": inputs, "original_execution_exit_code": code,
    "freeze_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"original_exit": code, "source_sha256": metadata["source_sha256"], "python_fences": len(metadata["python_fences"]), "figures": len(metadata["svg_references"]), "inputs": len(inputs)}, ensure_ascii=False))
raise SystemExit(code)
