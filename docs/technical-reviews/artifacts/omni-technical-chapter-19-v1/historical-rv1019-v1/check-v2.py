"""保存v1失敗收據；只修正bounded closed_item呼叫漏傳的unavailability。"""

from pathlib import Path
import hashlib
import json

base = Path(__file__).with_name("check.py")
root = Path(__file__).resolve().parents[5]
prior_path = root / "docs/course-revision-20261029-omni/reviews/technical/technical_chapter_19/historical-callback-self-check-v1.json"
assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == "128041be22cd7c73bbd9ada93622a98c13de6ddecfc737cb4e0f705c84044973"
prior = json.loads(prior_path.read_bytes())
assert hashlib.sha256(base.read_bytes()).hexdigest() == prior["script"]["sha256"]
source = base.read_text()
replacements = [
    ('OUTPUT = ROOT / BASE / "historical-callback-self-check-v1.json"', 'OUTPUT = ROOT / BASE / "historical-callback-self-check-v2.json"'),
    ('"command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-chapter-19-v1/historical-rv1019-v1/check.py"', '"command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-chapter-19-v1/historical-rv1019-v1/check-v2.py"'),
    ('checker.closed_item({"kind": "page"}, row, original, appendix, TASK, "chapter-19", appendix_at)', 'checker.closed_item({"kind": "page", "unavailability": request_group["unavailability"]}, row, original, appendix, TASK, "chapter-19", appendix_at)'),
    ('receipt["completed_at"] = datetime.now(UTC).isoformat()', 'receipt["previous_failed_receipt"] = ref(BASE + "/historical-callback-self-check-v1.json")\nreceipt["composed_verifier"] = {"base": ref("docs/technical-reviews/artifacts/omni-technical-chapter-19-v1/historical-rv1019-v1/check.py"), "wrapper": ref(str(Path(__file__).relative_to(ROOT))), "correction": "Pass the already verified real unavailability reference into closed_item; retain prior failed receipt and all original reports."}\nreceipt["completed_at"] = datetime.now(UTC).isoformat()'),
]
for old, new in replacements:
    assert source.count(old) == 1
    source = source.replace(old, new)
exec(compile(source, str(base), "exec"), {"__file__": __file__, "__name__": "__main__"})
