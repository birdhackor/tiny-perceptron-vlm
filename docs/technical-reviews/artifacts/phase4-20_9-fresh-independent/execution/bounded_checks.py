"""Independent CPU checks of exact lesson fence and existing scoring contract.

No model inference, training, new dataset acquisition, or score generation.
"""
import ast
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time

BASE = Path(__file__).resolve().parents[1]
started = time.perf_counter()
source = BASE / "original/fence-1.py"
namespace = {}
stdout = io.StringIO()
with redirect_stdout(stdout):
    exec(compile(source.read_bytes(), str(source), "exec"), namespace)
expected = "卡 0 物件吻合 True\n活動符合本題 True\n卡 1 物件吻合 True\n活動符合本題 False\n"
assert stdout.getvalue() == expected
print("EXACT ORIGINAL FENCE OUTPUT:")
print(stdout.getvalue(), end="")
truth = namespace["truth"]
assert {"腳踏車", "人"} == truth["objects"]
assert {"人", "腳踏車", "狗"} != truth["objects"]
assert "正在騎腳踏車" != truth["activity"]
print("VARIANTS: order of object labels is ignored; added dog fails object-set equality;")
print("equivalent natural wording fails literal equality, hence the fence is only a controlled label example.")

manifest_file = BASE / "original/docs/natural-assistant/v4/manifest.json"
protocol_file = BASE / "original/docs/natural-assistant/v4/validation-protocol-lower-lr.json"
manifest = json.loads(manifest_file.read_bytes())
protocol = json.loads(protocol_file.read_bytes())
scorer = BASE / "original/scripts/score_natural_v4_validation.py"
tree = ast.parse(scorer.read_bytes())
allowed_constants = {"WEIGHTS", "DENOMINATOR"}
allowed_functions = {"require", "case_id", "expected_cases"}
nodes = []
locators = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in allowed_constants for t in node.targets):
        nodes.append(node)
        locators.append(["constant", node.lineno, node.end_lineno])
    elif isinstance(node, ast.FunctionDef) and node.name in allowed_functions:
        nodes.append(node)
        locators.append([node.name, node.lineno, node.end_lineno])
contract = {"Counter": Counter}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(scorer), "exec"), contract)
cases, audio = contract["expected_cases"](manifest, protocol)
counts = dict(Counter(case["group"] for case in cases.values()))
assert counts["photo_summary"] == 28 and counts["photo_fact"] == 56
assert len(cases) == 132 and len(audio) == 16
print("ORIGINAL expected_cases AST scope:", locators)
print("VALIDATION EXISTING QUESTION DENOMINATORS:", json.dumps(counts, sort_keys=True))
photo_cases = [c for c in cases.values() if c["group"] in {"photo_summary", "photo_fact"}]
images = {c["row"]["image"] for c in photo_cases}
per_image = defaultdict(Counter)
for c in photo_cases:
    per_image[c["row"]["image"]][c["group"]] += 1
assert len(images) == 28
assert all(c == {"photo_summary": 1, "photo_fact": 2} for c in per_image.values())
print("PHOTO DENOMINATOR UNIT: 84 questions on 28 images; 1 summary + 2 facts per image, not 84 independent photos.")
assert all(
    "synonyms" in c["row"]["references"]["rubric"].lower()
    and "equivalent" in c["row"]["references"]["rubric"].lower()
    and any(term in c["row"]["references"]["rubric"].lower() for term in ["reject", "do not accept"])
    for c in photo_cases
)
print("ALL 84 PHOTO QUESTIONS: predeclared semantic rubric present.")
family_splits = defaultdict(set)
image_splits = defaultdict(set)
for r in manifest["rows"]:
    if r["task"] == "scene":
        family_splits[r["family"]].add(r["split"])
        image_splits[r["image"]].add(r["split"])
family_leaks = {k: sorted(v) for k, v in family_splits.items() if len(v) > 1}
image_leaks = {k: sorted(v) for k, v in image_splits.items() if len(v) > 1}
assert not family_leaks and not image_leaks
print("EXISTING PHOTO SPLITS: cross-split declared-family overlap=0; image-path overlap=0.")
print("Scope: declared similarity families and file paths; no claim of distinct subjects, capture sessions or upstream pretraining exclusion.")
cat_rows = [(i, r) for i, r in enumerate(manifest["rows"]) if "docci/train_01827/" in r["id"]]
assert [i for i, r in cat_rows] == [206, 207]
assert all(r["split"] == "train" for i, r in cat_rows)
assert cat_rows[1][1]["answer"] == "兩隻貓。"
assert cat_rows[0][1]["source"]["image_sha256"] == hashlib.sha256((BASE / "render/cat-original.jpg").read_bytes()).hexdigest()
print("CAT SOURCE: manifest /rows/206 and /rows/207 are training demonstrations; image byte SHA matches embedded original JPEG.")
print("UNRUN: model image swaps, capability evaluation, heldout performance, GPU, training.")
print("ENVIRONMENT:", json.dumps({"python": sys.version, "platform": platform.platform(), "device": "CPU-only standard library", "scorer_sha256": hashlib.sha256(scorer.read_bytes()).hexdigest()}, sort_keys=True))
print("ELAPSED_SECONDS:", time.perf_counter() - started)
