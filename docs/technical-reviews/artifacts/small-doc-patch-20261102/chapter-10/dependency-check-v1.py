"""Bounded original-reviewer callback; hashes and exact text only."""

import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
PRIOR = "docs/course-revision-20261029-omni/reviews/P5-actual/technical-chapter-10/technical-original-v1.json"
FREEZE = "docs/course-small-patch-20261102/source-freeze.json"
OLD_GLOSSARY = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix06/sources/glossary.md"
digest = lambda raw: hashlib.sha256(raw).hexdigest()
prior_raw = (ROOT / PRIOR).read_bytes()
assert digest(prior_raw) == "b6ab17242b61a4c2c4cc7f02abd7fb2a88ad110b83413feb4de03d035302aa99"
prior = json.loads(prior_raw)
assert prior["reviewer_task"] == "/root/omni_p5_technical_chapter_10"
freeze = json.loads((ROOT / FREEZE).read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "chapter-10")
chapter = (ROOT / page["source"]).read_bytes()
body = chapter[:re.search(rb"^## ", chapter, re.M).start()]
assert digest(body) == prior["source_sha256"] == page["source_sha256"]
assert body == (ROOT / page["snapshot"]).read_bytes()
assert page["figures_sha256"] == prior["figure_sha256"] == {}
assert not re.findall(rb"!\[[^\]]*\]\(([^)]+)\)", body)

current = (ROOT / "course/glossary.md").read_bytes()
old = (ROOT / OLD_GLOSSARY).read_bytes()
assert digest(current) == "148dfdc8f74b96d67695a236f0b19d6d17fe0f6e55433428f49d0f376b172cb1"
assert digest(old) == "c71853d2368dc981d52c279eaedd7029e8f69d65fbf39a652606c83510f6f056"
added_row = "| C-v3裝置判斷器 | 目前建議的裝置決策成品，讀取3–6個裝置的明確欄位資料，回答動作、資訊是否齊全及固定規準評分；與舊文字C、C-v2研究版本分開驗收 | [19.25](chapters/19.md#19.25) |\n"
added_index = " · [C-v3](chapters/19.md#19.25)（裝置判斷器）"
restored = current.decode()
for change in [added_row, added_index]:
    assert restored.count(change) == 1
    restored = restored.replace(change, "", 1)
assert restored.encode() == old
relevant_rows = []
for term in ["| 圖片入口（", "| patch、patch embedding |", "| 影格（frame）、時間戳（timestamp） |"]:
    before = next(line for line in old.decode().splitlines() if line.startswith(term))
    after = next(line for line in current.decode().splitlines() if line.startswith(term))
    assert before == after
    relevant_rows.append(after)
other_dependencies = []
for source in prior["sources"]:
    if source["kind"] == "repository_code" and source["id"] != "s-glossary":
        actual = digest((ROOT / source["path"]).read_bytes())
        assert actual == source["sha256"]
        other_dependencies.append({"id": source["id"], "path": source["path"], "sha256": actual, "unchanged": True})
assert b"[../glossary.md]" not in body
assert "[名詞頁](../glossary.md)" in body.decode()
result = {
    "reviewer_task": "/root/omni_p5_technical_chapter_10",
    "created_at": datetime.now(UTC).isoformat(),
    "scope": "Original actor bounded glossary dependency callback; no new full-page or original-science review",
    "source_sha256": digest(body),
    "source_bytes": len(body),
    "figure_sha256": {},
    "current_body_matches_prior_and_current_freeze": True,
    "notebook_scope": "The chapter introduction page/job assigns no notebook; no numbered notebook review is claimed.",
    "glossary_before_sha256": digest(old),
    "glossary_after_sha256": digest(current),
    "inverse_two_additions_equals_old_bytes": True,
    "added_row": added_row.rstrip("\n"),
    "added_index": added_index,
    "unchanged_relevant_glossary_rows": relevant_rows,
    "other_prior_repository_dependencies": other_dependencies,
    "prior_claim": "c10: relevant glossary lookup and navigation only; not whole glossary technical accuracy",
    "all_assertions_passed": True,
    "environment": {"python": sys.version.split()[0], "device": "CPU; byte hashing and text inspection; no model imports"},
}
print(json.dumps(result, ensure_ascii=False, indent=2))
