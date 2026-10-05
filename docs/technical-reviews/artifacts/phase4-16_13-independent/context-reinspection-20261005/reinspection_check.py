"""Narrow same-owner byte/AST proof for the necessary 16.12 context change."""
import ast
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[4]
TASK = "/root/phase4_factual_coordinator/factual_16_13"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

prior = json.loads((HERE / "prior-report.opaque.json").read_text())
assert prior["reviewer_task"] == TASK
records = json.loads((HERE / "context-comparison.json").read_text())
assert records["16.13"]["exact_bytes_equal"] is True
assert records["16.9"]["exact_bytes_equal"] is True
assert records["16.12"]["exact_bytes_equal"] is False
spec = importlib.util.spec_from_file_location("facts_reinspection", BASE / "inputs/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)

old_fences = facts.fences((HERE / "frozen-16.12.md").read_bytes(), records["16.12"]["frozen_first_line"])
new_fences = facts.fences((HERE / "current-16.12.md").read_bytes(), records["16.12"]["current_first_line"])
assert len(old_fences) == len(new_fences) == 1
assert old_fences[0]["language"] == new_fences[0]["language"] == "python"
old_code, new_code = old_fences[0]["raw"], new_fences[0]["raw"]
(HERE / "frozen-16.12-fence.py").write_bytes(old_code)
(HERE / "current-16.12-fence.py").write_bytes(new_code)
assert ast.dump(ast.parse(old_code), include_attributes=False) == ast.dump(ast.parse(new_code), include_attributes=False)
assert [l for l in old_code.splitlines() if l.strip()] == [l for l in new_code.splitlines() if l.strip()]
old_body = (HERE / "frozen-16.12.md").read_bytes()
new_body = (HERE / "current-16.12.md").read_bytes()
assert [l for l in old_body.splitlines() if l.strip()] == [l for l in new_body.splitlines() if l.strip()]
assert new_body.count(b"\n") == old_body.count(b"\n") + 2

verified_artifacts = []
for item in prior["artifacts"]:
    path = ROOT / item["path"]
    actual = sha(path)
    assert actual == item["sha256"], (item["id"], actual, item["sha256"])
    verified_artifacts.append({"id": item["id"], "path": item["path"], "sha256": actual})
current_figure = ROOT / "course/figures/rewrite-16-13-visible-pairs.svg"
assert sha(current_figure) == prior["figure_sha256"]["course/figures/rewrite-16-13-visible-pairs.svg"]
assert current_figure.read_bytes() == (BASE / "inputs/visible-pairs.svg").read_bytes()
prior_figure_context = ROOT / "course/figures/rewrite-16-12-sliding-path.svg"
# This context figure was not re-reviewed; its fingerprint is recorded to delimit context versions.
context_figure_sha = sha(prior_figure_context)

current_16_9_fence = facts.fences((HERE / "current-16.9.md").read_bytes(), records["16.9"]["current_first_line"])[0]["raw"]
assert hashlib.sha256(current_16_9_fence).hexdigest() == sha(BASE / "inputs/referenced-16.9-original-fence.py")

method_files = [
    (ROOT / "docs/review-tools/factual-reviewer-instructions.md", BASE / "inputs/factual-reviewer-instructions.md"),
    (ROOT / "docs/review-tools/section_facts.py", BASE / "inputs/section_facts.py"),
    (ROOT / "scripts/check_technical_reviews.py", BASE / "inputs/check_technical_reviews.py"),
]
methods = [{"path": str(current.relative_to(ROOT)), "current_sha256": sha(current),
            "frozen_sha256": sha(frozen), "exact_bytes_equal": current.read_bytes() == frozen.read_bytes()}
           for current, frozen in method_files]
result = {
    "reviewer_task": TASK,
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu", "operation": "byte, SHA256 and Python AST comparison only"},
    "context_comparison": records,
    "16_12_code": {"frozen_sha256": hashlib.sha256(old_code).hexdigest(), "current_sha256": hashlib.sha256(new_code).hexdigest(),
                   "ast_equivalent": True, "all_nonempty_lines_equal": True, "only_change": "Two additional empty lines in the Python fence, before and after visible()."},
    "current_16_13_figure_sha256": sha(current_figure),
    "current_16_12_context_figure_sha256": context_figure_sha,
    "current_16_9_fence_sha256": hashlib.sha256(current_16_9_fence).hexdigest(),
    "verified_unchanged_prior_artifacts": verified_artifacts,
    "method_versions": methods,
    "cpu_reexecuted": False, "figure_rerendered": False, "primary_sources_refetched": False,
    "scope": "Same-owner necessary-context reinspection. Existing CPU, SVG render, original papers and official code are reused only after exact recorded-artifact hash checks. No full chapter/GPU/training rerun.",
}
(HERE / "reinspection-proof.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"reviewer_task": TASK, "own_body_unchanged": True, "own_figure_unchanged": True,
                  "16_12_ast_equivalent": True, "16_12_only_two_added_empty_lines": True,
                  "prior_artifacts_hash_verified": len(verified_artifacts), "method_versions": methods}, ensure_ascii=False, indent=2))
