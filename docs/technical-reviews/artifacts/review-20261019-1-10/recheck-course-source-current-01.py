"""比對原判斷所用章檔與現行內容，記錄實際依賴的段落是否相同。"""

import difflib
import hashlib
import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PREVIOUS = "docs/course-review-20261019/reviews/rv1019_technical_1-10/technical-1-10-original-01.json"
FREEZE = "docs/course-review-20261019/freeze-02.json"
COURSE = "course/chapters/01.md"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    completed = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
    return completed.stdout


def sections(raw):
    headings = list(re.finditer(rb"(?m)^## ([0-9]+\.[0-9]+) [^\r\n]+", raw))
    result = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
        result[heading[1].decode()] = raw[heading.start():end]
    return result


previous_raw = (ROOT / PREVIOUS).read_bytes()
assert digest(previous_raw) == "b7a19eb099b992640fbcbc76c44ec6f7081dee2fbdba88fa6f84da57b904e5bf"
previous = json.loads(previous_raw)
course_source = next(s for s in previous["sources"] if s["id"] == "s_course")
commit = git("rev-parse", "c889d619^{commit}").decode().strip()
before = git("show", f"{commit}^:{COURSE}")
committed_after = git("show", f"{commit}:{COURSE}")
current = (ROOT / COURSE).read_bytes()
assert digest(before) == course_source["sha256"]
assert current == committed_after
assert digest(current) == "d2fcd73563506b8d38f070dbc6ae23adf25050e4f0af86435170fd576c269787"

old_sections = sections(before)
new_sections = sections(current)
assert old_sections.keys() == new_sections.keys()
changed_sections = [key for key in old_sections if old_sections[key] != new_sections[key]]
assert changed_sections == ["1.4"]

old_lines = before.splitlines(keepends=True)
new_lines = current.splitlines(keepends=True)
changes = []
for operation, old_start, old_end, new_start, new_end in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
    if operation != "equal":
        changes.append({
            "operation": operation,
            "previous_line_start": old_start + 1,
            "current_line_start": new_start + 1,
            "previous_lines": [line.decode() for line in old_lines[old_start:old_end]],
            "current_lines": [line.decode() for line in new_lines[new_start:new_end]],
        })
assert len(changes) == 1
assert changes[0]["operation"] == "replace"
assert len(changes[0]["previous_lines"]) == len(changes[0]["current_lines"]) == 1

comparisons = []
for key in ("1.7", "1.8", "1.10"):
    assert old_sections[key] == new_sections[key]
    comparisons.append({
        "lesson_id": key,
        "heading": new_sections[key].splitlines()[0].decode(),
        "previous_sha256": digest(old_sections[key]),
        "current_sha256": digest(new_sections[key]),
        "byte_identical": True,
    })

freeze_raw = (ROOT / FREEZE).read_bytes()
assert digest(freeze_raw) == "2494c3c9718c3d22f675bfac5bf0735bad0713db08a4a451d9f6aa992f94e4f1"
page = next(p for p in json.loads(freeze_raw)["pages"] if p["page_id"] == "1.10")
snapshot = (ROOT / page["snapshot"]).read_bytes()
assert snapshot == new_sections["1.10"]
assert digest(snapshot) == page["source_sha256"] == previous["source_sha256"]
assert page["figures_sha256"] == previous["figure_sha256"]
figure_checks = []
for path, expected in page["figures_sha256"].items():
    observed = digest((ROOT / path).read_bytes())
    assert observed == expected
    figure_checks.append({"path": path, "sha256": observed, "matches_previous_and_freeze": True})

artifact_checks = []
for artifact in previous["artifacts"]:
    observed = digest((ROOT / artifact["path"]).read_bytes())
    assert observed == artifact["sha256"]
    artifact_checks.append({"id": artifact["id"], "path": artifact["path"], "sha256": observed, "matches_previous": True})

other_local_sources = []
for source in previous["sources"]:
    if source["id"] != "s_course" and source["kind"] == "repository_code":
        observed = digest((ROOT / source["path"]).read_bytes())
        assert observed == source["sha256"]
        other_local_sources.append({"id": source["id"], "path": source["path"], "sha256": observed})

dependent_claims = [c["id"] for c in previous["claims"] if any(e["source_id"] == "s_course" for e in c["evidence"])]
assert dependent_claims == ["c01"]
assert all(c["status"] == "verified" for c in previous["claims"])

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-10/recheck-course-source-current-01.py",
    "executed_at": datetime.now(UTC).isoformat(),
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "git": git("--version").decode().strip(), "device": "不適用：純 bytes 比對，未執行 tensor 計算"},
    "program_sha256": digest(Path(__file__).read_bytes()),
    "previous_report": {"path": PREVIOUS, "sha256": digest(previous_raw)},
    "source_freeze": {"path": FREEZE, "sha256": digest(freeze_raw)},
    "course_source": {"path": COURSE, "previous_sha256": digest(before), "current_sha256": digest(current), "current_equals_commit_blob": current == committed_after},
    "commit": commit,
    "git_show_command": f"git show --format=fuller --no-ext-diff --no-textconv {commit} -- {COURSE} notebooks/01/1.4.ipynb",
    "git_show_output": git("show", "--format=fuller", "--no-ext-diff", "--no-textconv", commit, "--", COURSE, "notebooks/01/1.4.ipynb").decode(),
    "course_changes": changes,
    "changed_sections": [{"lesson_id": key, "heading": new_sections[key].splitlines()[0].decode()} for key in changed_sections],
    "dependency_comparisons": comparisons,
    "s_course_inspection_note": course_source["inspection_note"],
    "s_course_dependent_claim_ids": dependent_claims,
    "target_equals_frozen_snapshot": True,
    "target_source_sha256": digest(snapshot),
    "figure_checks": figure_checks,
    "previous_artifact_checks": artifact_checks,
    "other_local_source_checks": other_local_sources,
    "unresolved_claims": [],
    "contradicted_claims": [],
    "result": "所有比對通過；變更僅在 Counter 說明，所依賴的 softmax、答案代價與鏈式法則小節逐位元組相同。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
