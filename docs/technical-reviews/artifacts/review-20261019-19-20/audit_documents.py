"""核對工程重算的身分收據及正文涵蓋範圍。"""

import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-19-20/audit_documents.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu"},
}
receipts = []
for name in ("engineering-final-b-mechanisms-execution.json", "engineering-final-c-execution.json"):
    path = "docs/lesson-products/reviews/" + name
    raw = (ROOT / path).read_bytes()
    doc = json.loads(raw)
    if isinstance(doc["checks"], list):
        checks = [{"name": c["name"], "status": c["status"]} for c in doc["checks"]]
        report_sha = doc["checked_documents"]["outputs/lesson-products/final-report-v1.json"]
        raw_sha = doc["checked_documents"]["outputs/lesson-products/final-report-v1.predictions.jsonl"]
    else:
        checks = [{"name": k, "status": v["passed"]} for k, v in doc["checks"].items()]
        raw_sha = doc["checks"]["raw_full_file_SHA_bytes_match_report"]["details"]["sha256"]
        report_sha = None
    receipts.append({"path": path, "sha256": hashlib.sha256(raw).hexdigest(),
                     "reviewer_task": doc["reviewer_task"], "reviewer_context": doc["reviewer_context"],
                     "environment": doc["environment"], "report_sha256": report_sha,
                     "raw_sha256": raw_sha, "recorded_checks": checks})
assert len({r["reviewer_task"] for r in receipts}) == 2
assert all(r["reviewer_task"] != "/root" for r in receipts)
assert all(r["raw_sha256"] == "8b593421043b2d0ee8b81caea34612b468e8af7eafd5b84c552f8c25b0cf4896" for r in receipts)
result["engineering_execution_receipts"] = receipts
result["inspection_scope"] = "只讀工程執行收據，核對兩個不同task曾記錄對同一原始分數的重算；不讀本節先前技術／讀者報告，不採用工程裁決作本次成績的證明。無法由JSON身分欄位單獨證明兩個執行context完全獨立。"
source = ROOT / "docs/course-review-20261019/freeze-03/sources/19.20.md"
body = source.read_text()
result["source_contract"] = {"path": str(source.relative_to(ROOT)), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                              "python_fences": len(re.findall(r"^```python", body, re.M)),
                              "svg_links": re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", body)}
assert result["source_contract"]["python_fences"] == 0
assert result["source_contract"]["svg_links"] == []
freeze = json.loads((ROOT / "docs/course-review-20261019/freeze-04.json").read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "19.20")
assert page["source_sha256"] == result["source_contract"]["sha256"] and page["figures_sha256"] == {}
result["freeze_entry"] = page
result["result"] = "斷言通過；身分與版本限於保存收據可支持的範圍。"
print(json.dumps(result, ensure_ascii=False, indent=2))
