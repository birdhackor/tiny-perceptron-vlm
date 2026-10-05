"""Fresh 5.10 input capture and exact original-fence execution; no training."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent

def h(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save_json(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n")

paths = [
    "docs/review-tools/factual-reviewer-instructions.md", "scripts/check_technical_reviews.py",
    "docs/review-tools/section_facts.py", ".agents/skills/clear-tutorial/references/review-protocol.md",
    "course/chapters/05.md", "course/training.md", "scripts/build_course.py",
    "scripts/course_experiments/text.py", "scripts/course_experiments/common.py",
    "scripts/course_experiments/run.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py",
    "docs/course-experiments/results/real_text.json",
    "data/training/text-initial/tinystories-train-512.jsonl",
    "data/training/text-initial/tinystories-train-prefix-complete.txt",
    "data/training/text-initial/tinystories-acquisition.json",
    "data/training/text-initial/tinystories-source-README.md",
    "data/training/text-initial/tinystories-source-api.json",
    "data/training/text-initial/asset-manifest.json",
    "data/training/text-initial/chinese-classical-train-365.jsonl",
    "data/training/text-initial/tang300-source.json",
    "data/training/text-initial/chinese-poetry-README.md",
]
manifest=[]
for rel in paths:
    src=ROOT/rel
    dst=OUT/"inputs"/rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src,dst)
    manifest.append({"original_path":rel,"snapshot_path":str(dst.relative_to(ROOT)),"bytes":src.stat().st_size,"sha256":h(src)})
save_json(OUT/"input-provenance.json",{"scope":"Personally captured current raw inputs, no prior review reports read; data are existing local inputs, not downloaded. No weights used or copied.","files":manifest})
helper=ROOT/"docs/review-tools/section_facts.py"
spec=importlib.util.spec_from_file_location("section_facts",helper)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
temporary=ROOT/"outputs/reviewer-tools/phase4-5_10-independent-original"
metadata=module.extract("course/chapters/05.md#5.10",temporary)
exit_code=module.execute(temporary,metadata,30)
target=OUT/"original"
target.mkdir(exist_ok=False)
for file in temporary.iterdir():
    if file.is_file(): shutil.copyfile(file,target/file.name)
print(json.dumps({"section_sha256":metadata["source_sha256"],"original_exit_code":exit_code,"python_fences":len(metadata["python_fences"]),"figures":metadata["svg_references"],"permanent_original":str(target.relative_to(ROOT))},ensure_ascii=False))
print((target/"stdout.txt").read_text())
print((target/"stderr.txt").read_text())
if exit_code: raise SystemExit(exit_code)
