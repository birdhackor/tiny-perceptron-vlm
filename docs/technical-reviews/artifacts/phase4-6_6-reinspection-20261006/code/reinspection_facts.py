"""Reinspection facts only: retain own history and verify actual input changes."""
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

ART = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
OLD = ROOT / "docs/technical-reviews/artifacts/phase4-6_6-independent"
digest = lambda raw: hashlib.sha256(raw).hexdigest()
prior_bytes = (ART / "history/prior-report.json").read_bytes()
assert digest(prior_bytes) == "847f573b15632cb1f57daa59aa247f09ddee1f520d400045c7f74635676cf01c"
prior = json.loads(prior_bytes)
raw = (ROOT / "course/chapters/06.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
i = next(i for i, h in enumerate(headers) if h[0].startswith(b"## 6.6 "))
body = raw[headers[i].start():headers[i+1].start()]
assert digest(body) == "e2ae1bb6b59c00e0e78834582e4f134897b5c76225c3a2eda307e7bb5f72a81e"
assert body == (ART / "inputs/current-section.md").read_bytes()
original = (OLD / "execution/original-fence/section.md").read_bytes()
replacements = {"登记": "登記", "仍须": "仍須", "决定": "決定", "编碼": "編碼"}
expected = original.decode("utf-8")
for before, after in replacements.items():
    assert expected.count(before) == 1, before
    expected = expected.replace(before, after)
assert body == expected.encode("utf-8")
fences = lambda b: re.findall(rb"```python\r?\n(.*?)```", b, re.S)
assert fences(body) == fences(original) and len(fences(body)) == 1
assert not re.findall(rb"!\[[^\]]*\]\(([^)]+)\)", body)
artifact_checks = []
for item in prior["artifacts"]:
    observed = digest((ROOT / item["path"]).read_bytes())
    assert observed == item["sha256"], item["path"]
    artifact_checks.append({"id": item["id"], "path": item["path"], "sha256": observed, "unchanged": True})
current_checks = []
for path in ("tiny_perceptron/data.py", "tiny_perceptron/tokenization.py", "tiny_perceptron/modal_data.py", "tiny_perceptron/multimodal.py", "docs/course-experiments/results/tokenizer.json"):
    frozen_sha = digest((OLD / "inputs" / path).read_bytes())
    observed_sha = digest((ROOT / path).read_bytes())
    assert frozen_sha == observed_sha, path
    current_checks.append({"path": path, "frozen_sha256": frozen_sha, "current_sha256": observed_sha, "unchanged": True})
result = {"reviewer_task": "/root/phase4_factual_coordinator/factual_6_6", "date": "2026-10-06", "source": "course/chapters/06.md#6.6", "prior_report_sha256": digest(prior_bytes), "original_section_sha256": digest(original), "current_section_sha256": digest(body), "current_section_first_line": raw[:headers[i].start()].count(b"\n") + 1, "current_chapter_frozen_snapshot_sha256": digest((ART / "inputs/current-chapter-frozen.md").read_bytes()), "actual_change": {"replacements": replacements, "changed_claim": "C4 wording only; no substantive technical assertion, API, arithmetic, example, scope, or Python fence change", "python_fences_unchanged": True, "figure_references": []}, "original_artifact_hash_checks": artifact_checks, "current_dependency_hash_checks": current_checks, "reuse": "Original actual CPU executions, official v0.23.2 snapshots, historical implementation, and measured-tokenizer provenance retained after hash and personally-read support checks. No rerun of unchanged fence or numerical checks.", "new_fence_execution": False, "executed_code_sha256": digest(Path(__file__).read_bytes()), "python": sys.version}
(ART / "execution/reinspection-facts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"current_section_sha256": result["current_section_sha256"], "prior_artifacts_unchanged": len(artifact_checks), "current_dependencies_unchanged": len(current_checks), "change": result["actual_change"], "result": "all assertions passed"}, ensure_ascii=False, indent=2))
