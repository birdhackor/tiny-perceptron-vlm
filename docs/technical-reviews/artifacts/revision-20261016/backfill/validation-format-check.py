"""Format checks do not replace the recorded independent substantive inspection."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[5]
PATH = ROOT / "docs/course-revision-20261016/reviews/technical-validation-backfill-original-01.json"
spec = importlib.util.spec_from_file_location("technical_review_checker", ROOT / "scripts/check_technical_reviews.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
report = json.loads(PATH.read_text())
freeze = json.loads((ROOT / "docs/course-revision-20261016/freeze-14.json").read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "validation")
body = (ROOT / page["snapshot"]).read_text()
assert hashlib.sha256(body.encode()).hexdigest() == page["source_sha256"]
assert (ROOT / "docs/validation.md").read_text() == body
args = (ROOT, ROOT / "docs/validation.md", "validation", body)
ownership = {"/root/backfill_tech_validation": ["validation"]}
errors = checker._validate(*args, report, ownership, set())
altered = deepcopy(report)
altered["source_sha256"] = "0" * 64
negative = checker._validate(*args, altered, ownership, set())
assert "事實審閱不是目前小節原文" in negative
result = {"reviewer_task": "/root/backfill_tech_validation", "command": "uv run --frozen --extra cpu python docs/technical-reviews/artifacts/revision-20261016/backfill/validation-format-check.py", "environment": {"python": platform.python_version(), "device": "cpu"}, "source_sha256": page["source_sha256"], "positive_errors": errors, "negative_altered_source_errors": negative, "scope": "Direct _validate identity/schema/source/artifact check on this one page; negative case changed only an in-memory copy, never modifies report/source. Does not prove truth or actual agent independence."}
Path(__file__).with_name("validation-format-check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False))
if errors:
    raise SystemExit(1)
