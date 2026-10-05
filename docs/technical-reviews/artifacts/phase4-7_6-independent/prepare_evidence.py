"""Save real inputs and original-run proof under the permanent artifact directory."""
import hashlib
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
for name in ["course/chapters/07.md", "course/training.md", "tiny_perceptron/model.py", "tiny_perceptron/data.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py", "tiny_perceptron/__init__.py", "scripts/train.py", "scripts/build_course.py", "scripts/check_technical_reviews.py", "docs/review-tools/section_facts.py", "docs/review-tools/factual-reviewer-instructions.md", ".agents/skills/clear-tutorial/references/review-protocol.md", "pyproject.toml"]:
    target = BASE / "inputs" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, target)
run = ROOT / "outputs/reviewer-tools/phase4-7_6-independent-original"
original = BASE / "original-run"
original.mkdir(exist_ok=True)
for name in ["section.md", "fence-1.py", "bootstrap.py", "extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"]:
    shutil.copyfile(run / name, original / name)
manifest = {"input_provenance": "Repository originals and my section_facts.py extraction/execution only. Old technical/reader reports, conclusions and other reviewer judgments were not read.", "original_command": ".venv/bin/python docs/review-tools/section_facts.py course/chapters/07.md#7.6 --output outputs/reviewer-tools/phase4-7_6-independent-original --execute --timeout 60", "retained_originals": []}
for path in sorted((BASE / "inputs").rglob("*")) + sorted(original.rglob("*")):
    if path.is_file():
        manifest["retained_originals"].append({"path": str(path.relative_to(BASE)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
(BASE / "input-provenance.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
