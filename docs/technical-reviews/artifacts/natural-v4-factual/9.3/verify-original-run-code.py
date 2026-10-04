"""Trace pinned original run code; compare relevant functions with current code."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import requests

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
saved = json.loads((ROOT / "docs/course-experiments/results/safety.json").read_text())
name = "scripts/course_experiments/behavior.py"
url = "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/" + saved["revision"] + "/" + name
original = ROOT / "outputs/natural-v4/factual-research/9.3/original-run-behavior.py"
if not original.exists():
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(response.content)
old, current = original.read_bytes(), (ROOT / name).read_bytes()
digest = hashlib.sha256(old).hexdigest()
assert digest == saved["code_sha256"][name]
print("PYTHON", sys.version)
print("ORIGINAL_SOURCE_URL", url)
print("ORIGINAL_SOURCE_SHA256", digest)
print("CURRENT_SOURCE_SHA256", hashlib.sha256(current).hexdigest())
oldtree, newtree = ast.parse(old), ast.parse(current)
for fn in ["_conversation", "_safety_records", "_safety_evaluations", "run_safety"]:
    a = next(n for n in oldtree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
    b = next(n for n in newtree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    print("AST_EQUAL", fn, "original lines", a.lineno, a.end_lineno, "current lines", b.lineno, b.end_lineno)
for name in ["scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py"]:
    digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert digest == saved["code_sha256"][name]
    print("ORIGINAL_RUN_FULL_FILE_SHA_MATCH", name, digest)
print("PASS: behavior.py full file changed, relevant original-run functions match; remaining audited files match recorded full SHA")
