"""比對原報告依賴的教材區段、現行檔案與凍結版本，輸出複查收據。"""

import difflib
import hashlib
import json
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
commands = []
checks = []


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*arguments, preserve_text=True):
    argv = ["git", *arguments]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True)
    record = {"command": shlex.join(argv), "exit_code": result.returncode, "stderr": result.stderr.decode()}
    if preserve_text:
        record["stdout"] = result.stdout.decode()
    else:
        record["stdout_sha256"] = digest(result.stdout)
        record["stdout_bytes"] = len(result.stdout)
    commands.append(record)
    return result.stdout


def check(name, expected, observed):
    if expected != observed:
        raise AssertionError(f"{name}：預期 {expected!r}，實際 {observed!r}")
    checks.append({"name": name, "expected": expected, "observed": observed})


def sections(raw):
    headings = list(re.finditer(rb"(?m)^## ([^ \r\n]+) ([^\r\n]+)", raw))
    result = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
        result[heading[1].decode()] = {
            "title": heading[2].decode(),
            "raw": raw[heading.start() : end],
            "start": heading.start(),
            "end": end,
        }
    return result


previous_path = "docs/course-review-20261019/reviews/rv1019_technical_1-6/technical-1-6-original-01.json"
previous_raw = (ROOT / previous_path).read_bytes()
previous = json.loads(previous_raw)
check(
    "原始判斷的內容雜湊",
    "a718ea60a9e5ed946cfb5d582b1334f40ce73bba5619b43d05e2bcd954045292",
    digest(previous_raw),
)
sources = {source["id"]: source for source in previous["sources"]}
commit = git("rev-parse", "c889d619").decode().strip()
git("show", "c889d619")
baseline = git("show", f"{commit}^:{sources['s_prior']['path']}", preserve_text=False)
committed = git("show", f"{commit}:{sources['s_prior']['path']}", preserve_text=False)
current = (ROOT / sources["s_prior"]["path"]).read_bytes()
check("提交前版本與原報告依據相同", sources["s_prior"]["sha256"], digest(baseline))
check("現行教材的內容雜湊", "d2fcd73563506b8d38f070dbc6ae23adf25050e4f0af86435170fd576c269787", digest(current))
check("工作樹教材與指定提交相同", committed, current)
# 逐位元組相等的成功收據只保存雜湊，避免另存教材全文。
checks[-1]["expected"] = digest(committed)
checks[-1]["observed"] = digest(current)
old_lines = baseline.decode().splitlines(keepends=True)
new_lines = current.decode().splitlines(keepends=True)
changes = []
for operation, a, b, c, d in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
    if operation != "equal":
        changes.append({"operation": operation, "before": old_lines[a:b], "after": new_lines[c:d]})
check("教材差異只有一段替換", ["replace"], [change["operation"] for change in changes])
check("替換前後各一行", [[1, 1]], [[len(change["before"]), len(change["after"])] for change in changes])
changed_paths = git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).decode().splitlines()
check("指定提交只涉及教材與其 Notebook", ["course/chapters/01.md", "notebooks/01/1.4.ipynb"], changed_paths)
before_sections = sections(baseline)
after_sections = sections(current)
changed_sections = [key for key in before_sections if before_sections[key]["raw"] != after_sections[key]["raw"]]
notebook_section = next(Path(path).stem for path in changed_paths if path.endswith(".ipynb"))
check("修改只在對應 Notebook 所屬小節", [notebook_section], changed_sections)
dependencies = []
for title in ["怎樣留下真正沒練過的題目？", "知道前一字，怎樣改變下一字的機率？"]:
    key = next(key for key, section in before_sections.items() if section["title"] == title)
    old, new = before_sections[key]["raw"], after_sections[key]["raw"]
    check(f"依賴段落「{title}」完整位元組相同", digest(old), digest(new))
    markers = ["可調數字", "字表有七種字", 'text = "貓看狗。狗吃魚。貓看鳥。"']
    excerpts = [paragraph for paragraph in new.decode().split("\n\n") if any(marker in paragraph for marker in markers)]
    dependencies.append({"lesson_id": key, "title": title, "before_sha256": digest(old), "current_sha256": digest(new), "byte_identical": old == new, "relevant_excerpts": excerpts})
freeze_path = "docs/course-review-20261019/freeze-02.json"
freeze_raw = (ROOT / freeze_path).read_bytes()
check("現行凍結清單的內容雜湊", "2494c3c9718c3d22f675bfac5bf0735bad0713db08a4a451d9f6aa992f94e4f1", digest(freeze_raw))
freeze = json.loads(freeze_raw)
page = next(page for page in freeze["pages"] if page["page_id"] == previous["lesson_id"])
snapshot = (ROOT / page["snapshot"]).read_bytes()
lesson = after_sections[previous["lesson_id"]]["raw"]
check("現行小節與原判斷相同", previous["source_sha256"], digest(lesson))
check("現行凍結快照與原判斷相同", previous["source_sha256"], digest(snapshot))
check("凍結清單登記的小節雜湊相同", previous["source_sha256"], page["source_sha256"])
check("圖清單仍為空", {}, page["figures_sha256"])
followup = (ROOT / sources["s_followup"]["path"]).read_bytes()
check("後續特徵向量教材的內容雜湊未變", sources["s_followup"]["sha256"], digest(followup))
artifacts = []
for artifact in previous["artifacts"]:
    observed = digest((ROOT / artifact["path"]).read_bytes())
    check("既有證據的內容雜湊：" + artifact["path"], artifact["sha256"], observed)
    artifacts.append({"path": artifact["path"], "sha256": observed})
receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-6/current-source-check.py",
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "git": git("--version").decode().strip()},
    "previous_report": {"path": previous_path, "sha256": digest(previous_raw)},
    "commit": commit,
    "git_commands": commands,
    "prior_source": {"path": sources["s_prior"]["path"], "before_sha256": digest(baseline), "current_sha256": digest(current)},
    "changes": changes,
    "changed_sections": [{"lesson_id": key, "title": after_sections[key]["title"]} for key in changed_sections],
    "dependencies": dependencies,
    "dependent_claims": [claim["id"] for claim in previous["claims"] if any(item["source_id"] == "s_prior" for item in claim["evidence"])],
    "source_freeze": {"path": freeze_path, "sha256": digest(freeze_raw)},
    "lesson_sha256": digest(lesson),
    "snapshot_sha256": digest(snapshot),
    "figure_sha256": page["figures_sha256"],
    "followup_source": {"path": sources["s_followup"]["path"], "sha256": digest(followup)},
    "preserved_artifacts": artifacts,
    "checks": checks,
    "result": "依賴的完整小節與目標小節均未改變；僅 Counter 補充句修訂；既有證據的內容雜湊全部相符。",
}
print(json.dumps(receipt, ensure_ascii=False, indent=2))
