"""核對凍結原文、證據完整性，以及生成流程前置連結。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OLD_REPORT = "docs/course-review-20261019/reviews/rv1019_technical_14-12/technical-14-12-original-01.json"
OLD_REPORT_SHA = "cc163052b89ca7cd35c838e3d61bb6c9777caed9bb29c8e0aac04479b8b476b1"
OLD_SNAPSHOT = "docs/course-review-20261019/freeze-01/sources/14.12.md"
NEW_SNAPSHOT = "docs/course-review-20261019/freeze-07/sources/14.12.md"
NEW_SNAPSHOT_SHA = "d7ddc1d648a2107db665bc5304bb6abcdf919248e5f27563d28d908657e1ead4"
FREEZE = "docs/course-review-20261019/freeze-07.json"
FREEZE_SHA = "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def section(path, section_id):
    raw = (ROOT / path).read_bytes()
    pattern = rb"(?m)^## " + re.escape(section_id.encode()) + rb" [^\n]*\n"
    starts = list(re.finditer(pattern, raw))
    assert len(starts) == 1, "找不到唯一的小節標題"
    begin = starts[0].start()
    next_heading = re.search(rb"(?m)^## ", raw[starts[0].end():])
    end = starts[0].end() + next_heading.start() if next_heading else len(raw)
    return raw[begin:end]


old = json.loads((ROOT / OLD_REPORT).read_bytes())
assert digest(OLD_REPORT) == OLD_REPORT_SHA, "原始報告的位元組已變動"
assert digest(OLD_SNAPSHOT) == old["source_sha256"], "原始凍結原文不符"
assert digest(NEW_SNAPSHOT) == NEW_SNAPSHOT_SHA, "新版凍結原文不符"
assert digest(FREEZE) == FREEZE_SHA, "凍結清單不符"
assert old["issues"] == [], "原始報告有待核對問題"
checks = []
for kind in ("sources", "artifacts"):
    for item in old[kind]:
        if "path" not in item:
            continue
        observed = digest(item["path"])
        assert observed == item["sha256"], "既有來源或證據已變動"
        checks.append({"category": kind, "path": item["path"], "sha256": observed, "unchanged": True})

before = (ROOT / OLD_SNAPSHOT).read_bytes()
after = (ROOT / NEW_SNAPSHOT).read_bytes()
old_link, new_link = b"[7.20](07.md#7.20)", b"[1.14](01.md#1.14)"
assert before.count(old_link) == 1, "舊版連結出現次數不符"
assert before.replace(old_link, new_link) == after, "除指定連結以外還有其他變更"
source_path, page_id = old["source"].split("#")
assert section(source_path, page_id) == after, "現行章檔不等於凍結小節"
freeze = json.loads((ROOT / FREEZE).read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == page_id)
assert page["source_sha256"] == NEW_SNAPSHOT_SHA and page["figures_sha256"] == {}, "凍結頁面資料不符"

command = ["git", "diff", "--no-index", OLD_SNAPSHOT, NEW_SNAPSHOT]
diff = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
assert diff.returncode == 1, "兩份凍結原文的差異狀態不符"

link_path, target_id = new_link.decode().split("](")[1][:-1].split("#")
target_path = (Path(source_path).parent / link_path).as_posix()
target = section(target_path, target_id)
text = target.decode()
assert "先算下一字分配、選一字、接回句尾，再把新字當下一次前文" in text, "目標原文沒有對應的生成說明"
fences = re.findall(r"(?ms)^```python\n(.*?)^```", text)
assert len(fences) == 1, "目標原碼區塊數不符"
trace = []
multinomial = torch.multinomial


def observe(probabilities, *args, **kwargs):
    sample = multinomial(probabilities, *args, **kwargs)
    trace.append({"probabilities": probabilities.tolist(), "selected_id": sample.item()})
    return sample


torch.multinomial = observe
namespace = {"__name__": "__main__"}
stdout = io.StringIO()
try:
    with contextlib.redirect_stdout(stdout):
        exec(compile(fences[0], target_path, "exec"), namespace)
finally:
    torch.multinomial = multinomial
previous = 0
assert len(trace) == 12 and len(namespace["output"]) == 13, "接字次數與總長度不符"
for step, event in enumerate(trace):
    assert event["probabilities"] == namespace["p"][previous].tolist(), "下一步沒有使用前一步選出的 token"
    assert namespace["output"][step + 1] == namespace["chars"][event["selected_id"]], "抽出的 token 沒有接到輸出末端"
    previous = event["selected_id"]
assert previous == namespace["current"], "最後保存的 token 不符"

changed_paragraph = next(p for p in after.decode().split("\n\n") if new_link.decode() in p)
changed_sentence = next(s + "。" for s in changed_paragraph.split("。") if new_link.decode() in s)
result = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "original_report": {"path": OLD_REPORT, "sha256": OLD_REPORT_SHA, "issues": old["issues"]},
    "integrity_checks": checks,
    "source_sha256": NEW_SNAPSHOT_SHA,
    "source_freeze": {"path": FREEZE, "sha256": FREEZE_SHA},
    "current_source_matches_freeze": True,
    "diff": {"command_argv": command, "exit_code": diff.returncode, "stdout": diff.stdout, "stderr": diff.stderr},
    "change": {"only_link_changed": True, "old_link": old_link.decode(), "new_link": new_link.decode(), "new_sentence": changed_sentence},
    "link_target": {
        "path": target_path, "sha256": digest(target_path), "section_id": target_id,
        "section_sha256": hashlib.sha256(target).hexdigest(), "title": text.splitlines()[0],
        "code_sha256": hashlib.sha256(fences[0].encode()).hexdigest(), "code": fences[0],
        "stdout": stdout.getvalue(), "trace": trace, "generated_characters": len(namespace["output"]),
        "each_selected_token_used_for_next_step": True,
    },
    "result": "全部斷言通過；差異只有前置連結，接字流程與前文相符",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
