"""Same-owner current-section reinspection; reuse only byte-verified own proof."""
import difflib
import hashlib
import json
import platform
import re
import sys
import urllib.request
from pathlib import Path

R = Path(__file__).resolve().parent
A = R.parent
ROOT = A.parents[3]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


prior_raw = (ROOT / "docs/technical-reviews/7.13.json").read_bytes()
prior = json.loads(prior_raw)
prior_hash = sha(prior_raw)
history = ROOT / "docs/technical-reviews/history" / f"phase4-7_13-own-before-current-reinspection-20261006-{prior_hash}.json"
assert history.read_bytes() == prior_raw
(R / "own-prior-report.json").write_bytes(prior_raw)
raw = (ROOT / "course/chapters/07.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
index = next(i for i, h in enumerate(headers) if h[0].startswith(b"## 7.13 "))
current = raw[headers[index].start():headers[index + 1].start()]
assert sha(current) == "22011a607dedd48e8254c8aeb35ed282529d034b73deea1f80b3c888a234171e"
current.decode("utf-8")
(R / "current-section.md").write_bytes(current)
frozen_path = ROOT / next(a["path"] for a in prior["artifacts"] if a["id"] == "new_section")
frozen = frozen_path.read_bytes()
assert sha(frozen) == prior["source_sha256"]
expected = frozen.decode().replace("风格", "風格").replace("字段", "欄位").replace("標准", "標準").encode()
assert expected == current
diff = "".join(difflib.unified_diff(frozen.decode().splitlines(True), current.decode().splitlines(True),
                                    fromfile="own frozen original", tofile="current original"))
(R / "own-frozen-to-current.diff").write_text(diff)
fence = re.search(rb"```python\r?\n(.*?)```", current, re.S)[1]
assert fence == (A / "fence-1.py").read_bytes()
figures = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", current.decode())
assert not figures

own_evidence_hashes = []
for artifact in prior["artifacts"]:
    body = (ROOT / artifact["path"]).read_bytes()
    assert sha(body) == artifact["sha256"]
    own_evidence_hashes.append({"id": artifact["id"], "path": artifact["path"], "sha256": sha(body)})
current_code_hashes = []
for name in ["scripts/course_experiments/common.py", "scripts/course_experiments/text.py",
             "scripts/course_experiments/compression.py", "scripts/prepare_data.py",
             "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/training.py"]:
    body = (ROOT / name).read_bytes()
    assert body == (A / "inputs/current" / name).read_bytes()
    current_code_hashes.append({"path": name, "sha256": sha(body)})
result_raw = (ROOT / "docs/course-experiments/results/sft.json").read_bytes()
assert result_raw == (A / "inputs/results/sft.json").read_bytes()
result = json.loads(result_raw)
assert result["results"]["training"]["steps"] == 900
assert result["results"]["training"]["checkpoint"] == "model.pt"
assert result["results"]["after"]["test"]["matches"] == 5
assert result["results"]["after"]["test"]["records"] == 10

page_url = "http://127.0.0.1:8765/7.13.html"
with urllib.request.urlopen(page_url, timeout=10) as response:
    html = response.read()
    (R / "current-page.html").write_bytes(html)
    page_response = {"url": page_url, "status": response.status, "sha256": sha(html), "bytes": len(html)}
assert "改風格" in html.decode() and "answer欄位" in html.decode()
receipt = {
    "reviewer_task": prior["reviewer_task"], "accessed_on": "2026-10-06",
    "kind": "same_owner_current_section_reinspection",
    "environment": {"python": platform.python_version(), "executable": sys.executable,
                    "device": "CPU; standard-library-only fingerprint/locator inspection", "cwd": str(ROOT)},
    "source_sha256": sha(current), "prior_source_sha256": prior["source_sha256"],
    "prior_report_path": str(history.relative_to(ROOT)), "prior_report_sha256": prior_hash,
    "actual_full_read": "Reviewer personally read all current 7.13 prose/fence/details before this execution, and compared the complete own frozen original diff.",
    "actual_changes": ["风格→風格", "字段→欄位", "標准→標準"],
    "substantive_claim_changes": [], "exact_lexical_substitutions_only": True,
    "changed_claim_support": {"C1": "欄位 names the same JSON dictionary key contract, without changing accepted values or scope.",
                              "C3": "欄位 in exercise names the same answer/result key lookup, retaining both-false result exercise.",
                              "C4": "標準 is orthographic normalization; task-specific criterion and safety limitations unchanged.",
                              "C5": "風格 is orthographic normalization; independent format/content metrics unchanged."},
    "fence_sha256": sha(fence), "figure_references": figures,
    "own_prior_evidence_hashes": own_evidence_hashes,
    "unchanged_current_implementation": current_code_hashes,
    "original_measurement_sha256": sha(result_raw),
    "primary_sources_reused": "Personally inspected CPython v3.13.5, HELM v1, sklearn 1.7.2, HF Transformers v4.57.1 source snapshots and actual original sft JSON from own prior evidence all retain their verified bytes and earlier precise support ranges.",
    "cpu_evidence_reused": "Original fence, bounded reply variations and historical-result denominator replay were not rerun; unchanged fence, implementations, original input JSON and own artifact hashes substantiate reuse.",
    "page_response": page_response, "necessary_context": "No new context needed for orthographic term normalization; prior own dict API and fixed-baseline context retained, with source identity and support checked.",
    "unresolved_questions": [],
    "limits": "No GPU/training/model or dataset download/upload, unrelated paper search or pipeline run. No manuscript/figure modification. Browser rendering is a separate saved check, not assumed here.",
}
(R / "verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k not in ["own_prior_evidence_hashes", "unchanged_current_implementation"]}, ensure_ascii=False, indent=2))
