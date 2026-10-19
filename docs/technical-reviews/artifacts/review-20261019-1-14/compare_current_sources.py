"""比對已封存判斷與現行證據，確認修改是否影響所依賴的內容。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
REPORT = Path("docs/course-review-20261019/reviews/rv1019_technical_1-14/technical-1-14-original-01.json")
FREEZE = Path("docs/course-review-20261019/freeze-02.json")
COMMIT = "c889d6192fff95c2d12dde160d864c4f4f36f892"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    command = ["git", *args]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    return result.stdout, {"command_argv": command, "exit_code": result.returncode}


def sections(raw):
    matches = list(re.finditer(rb"(?m)^## ([^\r\n]+)", raw))
    result = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        identifier, title = match[1].decode().split(" ", 1)
        result[title] = {"id": identifier, "raw": raw[match.start():end]}
    return result


raw_report = (ROOT / REPORT).read_bytes()
assert sha256(raw_report) == "f5bc1c1e6cac207ed66ee6b7798b0ab8d54585a6f522fad73da252a6c7e7c667"
report = json.loads(raw_report)
chapter_source = next(source for source in report["sources"] if source["id"] == "repo_chapter")
chapter_path = chapter_source["path"]
old, parent_command = git("show", f"{COMMIT}^:{chapter_path}")
committed, committed_command = git("show", f"{COMMIT}:{chapter_path}")
current = (ROOT / chapter_path).read_bytes()
assert sha256(old) == chapter_source["sha256"]
assert current == committed
assert sha256(current) == "d2fcd73563506b8d38f070dbc6ae23adf25050e4f0af86435170fd576c269787"
show, show_command = git("show", COMMIT)
history, history_command = git("log", "--format=%H %s", f"{COMMIT}^..HEAD", "--", chapter_path)
assert [line.split()[0] for line in history.decode().splitlines()] == [COMMIT]
old_sections, current_sections = sections(old), sections(current)
assert old_sections.keys() == current_sections.keys()
changed = [title for title in old_sections if old_sections[title]["raw"] != current_sections[title]["raw"]]
assert len(changed) == 1
assert "`Counter`" in current_sections[changed[0]]["raw"].decode()
old_lines, current_lines = old.splitlines(keepends=True), current.splitlines(keepends=True)
assert len(old_lines) == len(current_lines)
changed_lines = [index for index, (before, after) in enumerate(zip(old_lines, current_lines)) if before != after]
assert len(changed_lines) == 1
before = old_lines[changed_lines[0]].decode()
after = current_lines[changed_lines[0]].decode()
assert before.replace("所以字典的寫法都能照用", "所以本節用到的 `.values()`、`.items()` 都能照用") == after
dependent_titles = ["文字怎麼變成可還原的編號？", "怎樣留下真正沒練過的題目？", "真的更新一次，代價會怎樣？"]
comparisons = []
for title in dependent_titles:
    before, after = old_sections[title]["raw"], current_sections[title]["raw"]
    assert before == after
    comparisons.append({"section_id": current_sections[title]["id"], "title": title, "before_sha256": sha256(before), "current_sha256": sha256(after), "byte_equal": True, "bytes": len(after)})
raw_freeze = (ROOT / FREEZE).read_bytes()
assert sha256(raw_freeze) == "2494c3c9718c3d22f675bfac5bf0735bad0713db08a4a451d9f6aa992f94e4f1"
freeze = json.loads(raw_freeze)
page = next(page for page in freeze["pages"] if page["page_id"] == report["lesson_id"])
target = next(item for item in current_sections.values() if item["id"] == report["lesson_id"])["raw"]
snapshot = (ROOT / page["snapshot"]).read_bytes()
old_target = next(item for item in old_sections.values() if item["id"] == report["lesson_id"])["raw"]
assert target == old_target == snapshot
assert sha256(target) == report["source_sha256"] == page["source_sha256"]
assert page["figures_sha256"] == report["figure_sha256"] == {}
source_checks = []
for source in report["sources"]:
    if source["kind"] != "repository_code":
        continue
    observed = sha256((ROOT / source["path"]).read_bytes())
    matches = source["sha256"] == observed
    assert matches or source["id"] == chapter_source["id"]
    source_checks.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": observed, "matches_previous": matches})
artifact_checks = []
for artifact in report["artifacts"]:
    observed = sha256((ROOT / artifact["path"]).read_bytes())
    assert observed == artifact["sha256"]
    artifact_checks.append({"id": artifact["id"], "path": artifact["path"], "sha256": observed, "matches_previous": True})
verification = json.loads((ROOT / "docs/technical-reviews/artifacts/review-20261019-1-14/verification.json").read_bytes())
additional_inputs = []
for path, expected in verification["source_sha256"].items():
    if path == chapter_path:
        continue
    observed = sha256((ROOT / path).read_bytes())
    assert observed == expected
    additional_inputs.append({"path": path, "sha256": observed, "matches_previous_execution": True})
assert all(claim["status"] == "verified" for claim in report["claims"])
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-14/compare_current_sources.py > docs/technical-reviews/artifacts/review-20261019-1-14/current-source-comparison-01.json",
    "environment": {"python": platform.python_version(), "python_executable": sys.executable, "platform": platform.platform(), "device": "不涉及 tensor 運算；僅唯讀 Git 與檔案位元組比較"},
    "checked_at": datetime.now().astimezone().isoformat(),
    "previous_report": {"path": str(REPORT), "sha256": sha256(raw_report)},
    "git_show": {**show_command, "stdout": show.decode()},
    "git_history": {**history_command, "stdout": history.decode()},
    "chapter_comparison": {"path": chapter_path, "previous_source_command": parent_command, "committed_source_command": committed_command, "previous_sha256": sha256(old), "current_sha256": sha256(current), "worktree_equals_commit": True, "changed_line_count": len(changed_lines), "changed_sections": [{"section_id": current_sections[title]["id"], "title": title} for title in changed]},
    "dependent_sections": comparisons,
    "reviewed_page": {"lesson_id": report["lesson_id"], "sha256": sha256(target), "byte_equal_to_previous": True, "byte_equal_to_snapshot": True, "source_freeze": {"path": str(FREEZE), "sha256": sha256(raw_freeze)}, "figure_sha256": page["figures_sha256"]},
    "repository_sources": source_checks,
    "previous_artifacts": artifact_checks,
    "additional_execution_inputs": additional_inputs,
    "affected_claims": [],
    "unresolved_claims": [],
    "contradicted_claims": [],
    "result": "全部比對斷言通過。唯一文字修改位於 Counter 的補充說明，未碰到原判斷依賴的段落；保留原有實跑證據與全部已核實主張。此次未重新執行訓練或外部來源下載。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
