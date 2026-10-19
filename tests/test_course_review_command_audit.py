"""可觀察工具與命令稽核（docs/course-review-20261019/tools/command_audit.py）的離線測試。

驗證：讀者命令字面裡的禁讀路徑、只能當工具參數的凍結檔與閱讀計畫、Git 歷史、連網、字面程式裡的
禁讀路徑、網頁搜尋，以及看了收據以外的圖，都會被標出；照讀者指示的合法命令不被標。
不涵蓋：命令執行的程式內部開了哪些檔；這是工具本身的限制，寫在工具檔頭。
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "command_audit", ROOT / "docs/course-review-20261019/tools/command_audit.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

# 命令裡的 repo 根目錄；故意和本機 repo 不同，確認路徑照 session 的 POSIX 路徑解讀，不看本機作業系統。
SESSION_ROOT = "/work/tiny-perceptron-vlm"
TASK = "/root/rv1019_reader_ch01"
FREEZE = "docs/course-review-20261019/freeze-01.json"
PLAN = "docs/course-review-20261019/reader-plan-ch01.json"
RECEIPT = "docs/course-review-20261019/figure-renders.json"
READER = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/course-revision-20261016/tools/reader.py"
ALLOWED_PNG = json.loads((ROOT / RECEIPT).read_bytes())["records"][0]["png"]["path"]


def policy():
    return AUDIT.ReaderPolicy(TASK, FREEZE, PLAN, RECEIPT)


def log_with(tmp_path, *commands, searches=()):
    lines = [{"type": "thread.started", "thread_id": "t"}]
    for command in commands:
        item = {"type": "command_execution", "command": f"/bin/zsh -lc {json.dumps(command)}", "exit_code": 0}
        lines.append({"type": "item.completed", "item": item})
    for query in searches:
        lines.append({"type": "item.completed", "item": {"type": "web_search", "query": query}})
    path = tmp_path / "codex.log"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def levels(result):
    return sorted(f["level"] for entry in result["flagged_commands"] for f in entry["findings"])


def test_reader_commands_from_the_instructions_are_not_flagged(tmp_path):
    log = log_with(
        tmp_path,
        f"{READER} start 1.1 --reviewer {TASK} --plan {PLAN} --freeze {FREEZE}",
        f"{READER} supplement first-steps --reviewer {TASK} --plan {PLAN} --freeze {FREEZE}",
        f"shasum -a 256 {PLAN} {FREEZE}",
        f"shasum -a 256 {SESSION_ROOT}/docs/course-revision-20261016/frozen-figures/"
        "65f29ee754f2bb61393cb2cbf79048329058c16ec24b9b16dbb2e142fa2c4e92.svg",
        "ls docs/course-review-20261019/reviews/rv1019_reader_ch01/drafts",
        ".venv/bin/python docs/course-review-20261019/tools/executed_outputs.py show "
        + "a" * 32
        + f" --reviewer {TASK}",
    )
    result = AUDIT.audit(policy(), [log], [], root=SESSION_ROOT)
    assert result["commands_total"] == 6
    assert result["flagged_commands"] == []


@pytest.mark.parametrize(
    "command",
    [
        "cat course/chapters/01.md",
        f"sed -n 1,5p {SESSION_ROOT}/course/first-steps.md",
        f"cat {PLAN}",
        f"cat {PLAN} && {READER} start 1.1 --reviewer {TASK} --plan {PLAN} --freeze {FREEZE}",
        "git log --oneline -3",
        "curl -s https://example.com",
        "cat outputs/course-inline-code-20261017/pub/orders/rv-content.md",
        "cat docs/course-repair-20261017/manifest-final-01.json",
        "cat docs/course-review-20261019/executed-outputs-ch01.json",
        "cat docs/course-review-20261019/output-access/x.json",
        ".venv/bin/python -c 'print(open(\"course/chapters/01.md\").read())'",
    ],
)
def test_reads_outside_the_reader_scope_are_violations(tmp_path, command):
    result = AUDIT.audit(policy(), [log_with(tmp_path, command)], [], root=SESSION_ROOT)
    assert "violation" in levels(result), command


def test_searches_and_inline_programs_need_a_look_but_are_not_violations(tmp_path):
    log = log_with(
        tmp_path,
        "rg 已足夠 docs/course-review-20261019/reviews/rv1019_reader_ch01/drafts",
        ".venv/bin/python -c 'import json; print(1)'",
        "cat ../elsewhere/notes.txt",
    )
    result = AUDIT.audit(policy(), [log], [], root=SESSION_ROOT)
    assert levels(result) == ["review", "review", "review"]
    assert result["inline_code_variants"] == [{"code": "import json; print(1)", "count": 1}]


def test_a_sibling_folder_sharing_the_repo_name_prefix_is_outside_the_repo(tmp_path):
    log = log_with(tmp_path, f"cat {SESSION_ROOT}-old/course/chapters/01.md")
    result = AUDIT.audit(policy(), [log], [], root=SESSION_ROOT)
    assert [f["reason"] for f in result["flagged_commands"][0]["findings"]] == ["repo 外的路徑"]


def test_web_search_is_a_violation(tmp_path):
    result = AUDIT.audit(policy(), [log_with(tmp_path, searches=["perceptron"])], [], root=SESSION_ROOT)
    assert result["summary"]["violation"] == 1


def test_only_receipt_images_may_be_viewed(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    calls = [
        f'await tools.view_image({{path:"{SESSION_ROOT}/{ALLOWED_PNG}", detail:"original"}})',
        'await tools.view_image({path:"outputs/course-inline-code-20261017/other.png"})',
    ]
    rows = [{"timestamp": "t", "payload": {"type": "custom_tool_call", "name": "exec", "input": c}} for c in calls]
    rollout.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = AUDIT.audit(policy(), [log_with(tmp_path)], [rollout], root=SESSION_ROOT)
    assert [image["level"] for image in result["view_image"]] == [None, "violation"]
