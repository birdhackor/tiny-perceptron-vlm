"""Informed callback: verify freeze-09 source, original claims and cost boundaries."""
from contextlib import redirect_stdout
from fractions import Fraction
from io import StringIO
from pathlib import Path
import ast
import difflib
import hashlib
import json
import math
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = ROOT / "course/chapters/09.md"
body = lesson_slices(source.read_bytes().decode("utf-8"))["9.16"]
freezes = {v: json.loads((ROOT / f"docs/course-revision-20261016/freeze-{v}.json").read_text()) for v in ("08", "09")}
pages = {v: next(p for p in f["pages"] if p["page_id"] == "9.16") for v, f in freezes.items()}
old_body, new_body = [(ROOT / pages[v]["snapshot"]).read_bytes().decode("utf-8") for v in ("08", "09")]
assert new_body == body
for v, text in [("08", old_body), ("09", new_body)]:
    assert hashlib.sha256(text.encode()).hexdigest() == pages[v]["source_sha256"]
assert pages["09"]["figures_sha256"] == {}

blocks = lambda text: re.findall(r"```python\n(.*?)```", text, re.S)
old_blocks, new_blocks = blocks(old_body), blocks(new_body)
assert len(old_blocks) == len(new_blocks) == 1
old_ast, new_ast = [ast.dump(ast.parse(code)) for code in (old_blocks[0], new_blocks[0])]
assert old_ast == new_ast
noncode = lambda text: re.sub(r"```python\n.*?```", "<PYTHON-CODE>", text, flags=re.S)
assert noncode(old_body) == noncode(new_body)

expected_stdout = "0.6 2.0 澄清\n0.8 1.0 澄清\n0.9 0.5 回答\n"
stream = StringIO()
with redirect_stdout(stream):
    exec(compile(new_blocks[0], pages["09"]["snapshot"], "exec"), {})
assert stream.getvalue() == expected_stdout

# Execute the unchanged body of the author's for-loop with reviewer boundary inputs.
loop = ast.parse(new_blocks[0]).body[0]
assert isinstance(loop, ast.For)
loop_body = compile(ast.fix_missing_locations(ast.Module(body=loop.body, type_ignores=[])), "9.16:reviewer-boundary-input", "exec")
probe_inputs = [
    ("zero", 0.0, "澄清"),
    ("displayed-low", 0.6, "澄清"),
    ("just-below-decimal-tie", math.nextafter(0.8, -math.inf), "澄清"),
    ("displayed-tie", 0.8, "澄清"),
    ("just-above-decimal-tie", math.nextafter(0.8, math.inf), "澄清"),
    ("inside-epsilon-band", 0.8000000001, "澄清"),
    ("epsilon-boundary-rounded", 0.8000000002, "澄清"),
    ("outside-epsilon-band", 0.8000000004, "回答"),
    ("displayed-high", 0.9, "回答"),
    ("one", 1.0, "回答"),
]
probes = []
for label, p, expected in probe_inputs:
    ns = {"p": p}
    with redirect_stdout(StringIO()):
        exec(loop_body, ns)
    assert ns["action"] == expected, (label, ns)
    exact_p = Fraction(str(p))
    exact_cost = 5 * (1 - exact_p)
    ideal = "回答" if exact_cost < 1 else "澄清"
    probes.append({"label": label, "p": repr(p), "float_answer_cost": repr(ns["answer_cost"]),
                   "authored_action": ns["action"], "expected_authored_action": expected,
                   "exact_decimal_cost": str(exact_cost), "ideal_exact_cost_action": ideal})

# Independent exact arithmetic for the exercise and both strict boundaries.
thresholds = []
for cost in [Fraction(1), Fraction(1, 2)]:
    threshold = 1 - cost / 5
    checks = []
    for p in [threshold - Fraction(1, 10**10), threshold, threshold + Fraction(1, 10**10)]:
        answer_cost = 5 * (1 - p)
        action = "回答" if answer_cost < cost else "澄清"
        checks.append({"p": str(p), "answer_cost": str(answer_cost), "action": action})
    assert [r["action"] for r in checks] == ["澄清", "澄清", "回答"]
    thresholds.append({"clarification_cost": str(cost), "exact_threshold": str(threshold), "checks": checks})
assert [t["exact_threshold"] for t in thresholds] == ["4/5", "9/10"]

external_paths = {
 "guo17a.pdf": Path('/workspace/work/curriculum-20261016/technical-9.16/guo17a.pdf'),
 "geifman19a.pdf": Path('/workspace/work/curriculum-20261016/technical-9.16/geifman19a.pdf'),
 "launch.html": Path('/workspace/work/curriculum-20261016/technical-9.16/launch.html'),
 "deskmind-training.source": Path('/workspace/work/curriculum-20261016/fulltexts/deskmind-training.source'),
 "jev-intro.source": Path('/workspace/work/curriculum-20261016/fulltexts/jev-intro.source'),
}
expected_hashes = {
 "guo17a.pdf": "df7c93f97204f8a3c2b10f6d5bf5265baac25f00c1ae3b473a3e1dbc102dacb2",
 "geifman19a.pdf": "121c044e3fc396f0b274a1bdbbddd1f73f016814ee76a7aed8a6898fb51388f9",
 "launch.html": "dbad6fdb7e0acad42181e4d8cde6805f5d0978f5f73b89f7a17860ddc39e38fd",
 "deskmind-training.source": "f5dc2fbc28844eef273c7ce9e2b149544a28e05c8aceea99fcf3ccfe7d272725",
 "jev-intro.source": "75746b82d1678fc99948b5efdaf9ec43e5a11f0837a8105998f4172375a3a018",
}
external_checks = []
for name, path in external_paths.items():
    actual = sha(path)
    assert actual == expected_hashes[name], (name, actual)
    external_checks.append({"name": name, "path_outside_git": str(path), "sha256": actual, "same_as_original_review": True})
auto_page = next(p for p in freezes["09"]["pages"] if p["page_id"] == "19.17")
auto_body = (ROOT / auto_page["snapshot"]).read_bytes().decode()
assert hashlib.sha256(auto_body.encode()).hexdigest() == auto_page["source_sha256"]
original = ROOT / "docs/course-revision-20261016/reviews/technical-9.16-original.json"
receipt = {
 "reviewer_task": "/root/tech_20261016_9_16", "initial_review_context": "fresh", "current_callback_context": "informed",
 "source_sha256": pages["09"]["source_sha256"], "snapshot": pages["09"]["snapshot"],
 "source_matches_live_and_freeze": True, "figure_sha256": {},
 "original_report_path": str(original.relative_to(ROOT)), "original_report_sha256_at_callback": sha(original),
 "old_source_sha256": pages["08"]["source_sha256"],
 "noncode_equal": True, "ast_equal_ignoring_source_locations": True,
 "diff": ''.join(difflib.unified_diff(old_body.splitlines(keepends=True), new_body.splitlines(keepends=True))),
 "code_sha256": hashlib.sha256(new_blocks[0].encode()).hexdigest(),
 "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/9.16-revision09-callback-01-verify.py",
 "environment": {"python": platform.python_version(), "implementation": platform.python_implementation(), "device": "CPU; standard library only", "platform": platform.platform()},
 "stdout": stream.getvalue(), "expected_stdout": expected_stdout,
 "boundary_probes": probes, "exact_cost_threshold_checks": thresholds,
 "retained_original_source_versions": external_checks,
 "necessary_internal_reference": {"source": "course/chapters/19.md#19.17", "snapshot": auto_page["snapshot"], "sha256": auto_page["source_sha256"], "reread": True, "finding": "The current expanded design still first checks task support/information sufficiency and then selects B's compute budget. It explicitly leaves calibrated thresholds and completed results for later validation."},
 "result": "All assertions passed: current source and retained originals match recorded versions; AST/noncode unchanged; authored stdout and 10 boundary probes match; both exact-cost thresholds rechecked below/equal/above.",
 "scope": "Actual callback execution is this standard-library verification only. The author's separate kernel-execution09 (12 new kernels) was not rerun or claimed as reviewer execution. No training, GPU effect, product-quality test, real service action or site/layout check.",
 "epsilon_interpretation": "Ideal expected loss answers iff p>0.8. Authored code deliberately treats losses within 1e-9 below 1 as ties: just-above-0.8 and p=0.8000000001 still clarify, p=0.8000000004 answers. This pre-existing numerical tolerance is unchanged and does not invalidate the three displayed examples or the exact cost derivation."
}
out = ROOT / "docs/technical-reviews/artifacts/revision-20261016/9.16-revision09-callback-01-execution.json"
with out.open('x') as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({k:receipt[k] for k in ['source_sha256','noncode_equal','ast_equal_ignoring_source_locations','stdout','result','epsilon_interpretation']}, ensure_ascii=False, indent=2))
