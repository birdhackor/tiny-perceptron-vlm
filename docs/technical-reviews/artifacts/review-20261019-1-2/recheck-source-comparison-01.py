"""以原始位元組比較來源依賴，並核對既有證據的雜湊。"""

import difflib
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PREVIOUS = Path("docs/course-review-20261019/reviews/rv1019_technical_1-2/technical-1-2-original-01.json")
FREEZE = Path("docs/course-review-20261019/freeze-02.json")
CHAPTER = Path("course/chapters/01.md")
COMMIT = "c889d619"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def sections(raw):
    headings = list(re.finditer(rb"(?m)^## ([0-9]+\.[0-9]+) [^\r\n]*", raw))
    result = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
        result[heading[1].decode()] = raw[heading.start():end]
    return result


previous_raw = (ROOT / PREVIOUS).read_bytes()
assert sha(previous_raw) == "142d3404834ecd03aa3414da88fe05d2a4f7335af106b28c569558654e240678"
previous = json.loads(previous_raw)
linked = next(source for source in previous["sources"] if source["id"] == "linked_examples")
commit = git("rev-parse", COMMIT).decode().strip()
parent = git("rev-parse", COMMIT + "^").decode().strip()
before = git("show", f"{parent}:{CHAPTER}")
committed = git("show", f"{commit}:{CHAPTER}")
current = (ROOT / CHAPTER).read_bytes()
assert sha(before) == linked["sha256"]
assert current == committed
assert sha(current) == "d2fcd73563506b8d38f070dbc6ae23adf25050e4f0af86435170fd576c269787"

old_sections, new_sections = sections(before), sections(current)
assert old_sections.keys() == new_sections.keys()
changed = [key for key in old_sections if old_sections[key] != new_sections[key]]
assert changed == ["1.4"]
old_lines, new_lines = before.splitlines(keepends=True), current.splitlines(keepends=True)
changes = [op for op in difflib.SequenceMatcher(a=old_lines, b=new_lines).get_opcodes() if op[0] != "equal"]
assert len(changes) == 1
kind, old_start, old_end, new_start, new_end = changes[0]
assert kind == "replace" and old_end-old_start == new_end-new_start == 1
assert "所以字典的寫法都能照用" in old_lines[old_start].decode()
assert "所以本節用到的 `.values()`、`.items()` 都能照用" in new_lines[new_start].decode()

dependencies = []
for key in ("1.1", "1.2", "1.12", "1.14"):
    old, new = old_sections[key], new_sections[key]
    assert old == new
    dependencies.append({"selector": key, "heading": new.splitlines()[0].decode(), "old_sha256": sha(old), "current_sha256": sha(new), "bytes_equal": True, "bytes": len(new)})
assert b"chars = sorted(set(text))" in new_sections["1.1"]
reproduced_characters = sorted(set("貓看狗，狗看貓。"))
assert reproduced_characters == ["。", "狗", "看", "貓", "，"]

freeze_raw = (ROOT / FREEZE).read_bytes()
assert sha(freeze_raw) == "2494c3c9718c3d22f675bfac5bf0735bad0713db08a4a451d9f6aa992f94e4f1"
page = next(page for page in json.loads(freeze_raw)["pages"] if page["page_id"] == previous["lesson_id"])
snapshot = (ROOT / page["snapshot"]).read_bytes()
assert snapshot == new_sections[previous["lesson_id"]]
assert sha(snapshot) == page["source_sha256"] == previous["source_sha256"]
assert page["figures_sha256"] == previous["figure_sha256"] == {}

source_checks = []
for source in previous["sources"]:
    if source["kind"] != "repository_code":
        continue
    actual = sha((ROOT / source["path"]).read_bytes())
    source_checks.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual, "bytes_unchanged": source["sha256"] == actual})
    if source["id"] != "linked_examples":
        assert actual == source["sha256"]

artifact_checks = []
for artifact in previous["artifacts"]:
    actual = sha((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    artifact_checks.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "matches_previous": True})

linked_claims = [claim["id"] for claim in previous["claims"] if any(e["source_id"] == "linked_examples" for e in claim["evidence"])]
assert linked_claims == ["c23"]
changed_files = git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).decode().splitlines()
assert set(changed_files) == {str(CHAPTER), "notebooks/01/1.4.ipynb"}
history = git("log", "--format=%H", f"{parent}..HEAD", "--", str(CHAPTER)).decode().splitlines()
assert history == [commit]

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-2/recheck-source-comparison-01.py",
    "environment": {"python": platform.python_version(), "implementation": platform.python_implementation(), "platform": platform.platform(), "device": "cpu"},
    "result": "全部來源比較與證據雜湊斷言通過；原判斷依賴內容未變。",
    "script_sha256": sha(Path(__file__).read_bytes()),
    "previous_report": {"path": str(PREVIOUS), "sha256": sha(previous_raw)},
    "source_freeze": {"path": str(FREEZE), "sha256": sha(freeze_raw)},
    "source_comparison": {"path": str(CHAPTER), "commit": commit, "parent": parent, "old_sha256": sha(before), "current_sha256": sha(current), "current_equals_commit": current == committed, "changes_in_history": history, "changed_sections": changed, "changed_files": changed_files, "changed_lines": {"removed": old_lines[old_start].decode(), "added": new_lines[new_start].decode()}, "commit_diff_command": f"git show --format= {commit}", "commit_diff": git("show", "--format=", commit).decode()},
    "dependency_comparisons": dependencies,
    "original_inspection_note": linked["inspection_note"],
    "directly_dependent_claims": linked_claims,
    "sorted_characters": reproduced_characters,
    "frozen_target_matches_current": True,
    "figure_sha256": page["figures_sha256"],
    "repository_source_checks": source_checks,
    "original_artifact_checks": artifact_checks,
    "unresolved_claims": [],
    "contradicted_claims": [],
}, ensure_ascii=False, indent=2))
