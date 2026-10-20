"""核對凍結課文的有限算例，並保存當次原始來源讀取收據。"""

import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
RUN = ROOT / "outputs/reviewer-tools/runs/20261020T061439681929Z"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(name, value):
    with (ARTIFACT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def execute(code):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "frozen-example", "exec"), namespace)
    return namespace, output.getvalue()


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def retrieve_original(url, pages, locator, excerpts):
    command = ["curl", "-fsSL", "--max-time", "45", url]
    fetched = subprocess.run(command, check=True, capture_output=True)
    raw = fetched.stdout
    extraction = None
    if pages:
        javascript = (
            'ObjC.import("Foundation"); ObjC.import("PDFKit"); '
            'var data = $.NSFileHandle.fileHandleWithStandardInput.readDataToEndOfFile; '
            'var doc = $.PDFDocument.alloc.initWithData(data); '
            + json.dumps([page - 1 for page in pages])
            + '.map(i => ObjC.unwrap(doc.pageAtIndex(i).string)).join("\\n\\n");'
        )
        extraction = ["osascript", "-l", "JavaScript", "-e", javascript]
        text = subprocess.run(extraction, input=raw, check=True, capture_output=True).stdout.decode()
    else:
        parser = TextParser()
        parser.feed(raw.decode())
        text = " ".join(parser.parts)
    normalized = " ".join(text.split())
    for excerpt in excerpts:
        assert " ".join(excerpt.split()) in normalized, "原始來源未包含指定摘錄"
    title = None
    if not pages:
        title = re.search(r"<title>(.*?)</title>", raw.decode(), re.S).group(1)
    return {
        "url": url,
        "accessed_on": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(),
        "retrieval_command_argv": command,
        "retrieval_exit_code": fetched.returncode,
        "original_sha256": sha(raw),
        "original_bytes": len(raw),
        "extraction_command_argv": extraction,
        "inspected_pdf_pages": pages,
        "locator": locator,
        "html_title": title,
        "short_excerpts": excerpts,
        "excerpt_presence_verified": True,
        "storage_policy": "全文只在記憶體處理；保存收據與支持主張的短摘錄。",
    }


def main():
    source_path = ROOT / "docs/course-review-20261019/freeze-01/sources/8.14.md"
    frozen = source_path.read_bytes()
    assert sha(frozen) == "13e68daf1525f96b2d1ccbcca168576fd83153e6784570f263c805c07cfe178c"
    chapter = (ROOT / "course/chapters/08.md").read_bytes()
    start = re.search(rb"^## 8\.14 ", chapter, re.M).start()
    following = re.search(rb"^## ", chapter[start + 1 :], re.M)
    body = chapter[start : start + 1 + following.start()]
    assert body == frozen, "目前原文與凍結原文不同"
    code = re.search(r"```python\n(.*?)```", frozen.decode(), re.S).group(1)
    ast.parse(code)
    namespace, stdout = execute(code)
    notice, target, replies = (namespace[key] for key in ("notice", "target", "replies"))
    assert target == ["今日休館", "明日開放"]
    assert [language for language, _ in notice] == ["中文", "英文", "中文", "英文"]
    assert ["今日休館" in reply for reply in replies.values()] == [True] * 4
    assert [reply.splitlines() == target for reply in replies.values()] == [False, False, False, True]
    assert (RUN / "stdout.txt").read_text() == stdout
    extraction = json.loads((RUN / "extraction.json").read_bytes())
    environment = json.loads((RUN / "environment.json").read_bytes())
    execution = json.loads((RUN / "execution.json").read_bytes())
    assert extraction["source_sha256"] == sha(frozen)
    assert execution["exit_code"] == 0
    copied = {}
    for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json"):
        raw = (RUN / name).read_bytes()
        copied[name] = {"sha256": sha(raw), "utf8": raw.decode()}
    write_new("section-execution.json", {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/08.md#8.14" --execute',
        "result": "exit_code=0；正文輸出與凍結程式重跑完全相同。",
        "copied_files": copied,
        "environment": environment,
    })
    replacement = 'target = [text for language, text in notice if language == "中文"]'
    assert code.count(replacement) == 1
    exercise, exercise_stdout = execute(code.replace(replacement, "target = [notice[0][1]]"))
    assert exercise["target"] == ["今日休館"]
    exercise_matches = [reply.splitlines() == exercise["target"] for reply in replies.values()]
    assert exercise_matches == [False, True, False, False]
    assert exercise["notice"] == notice
    choices = {
        "只抄中文": [text for language, text in notice if language == "中文"],
        "只抄英文": [text for language, text in notice if language == "英文"],
        "只抄第一行": [notice[0][1]],
        "只列明天安排": [text for _, text in notice if text in ("明日開放", "Open tomorrow")],
    }
    assert choices["只抄英文"] == ["Closed today", "Open tomorrow"]
    assert choices["只列明天安排"] == ["明日開放", "Open tomorrow"]
    additions = {
        "加說明": "以下是中文版本\n今日休館\n明日開放",
        "交換順序": "明日開放\n今日休館",
        "重複項目": "今日休館\n明日開放\n明日開放",
        "加列號": "1 今日休館\n3 明日開放",
        "錯字": "今日開館\n明日開放",
        "空白行": "今日休館\n\n明日開放",
        "CRLF": "今日休館\r\n明日開放",
        "末尾換行": "今日休館\n明日開放\n",
    }
    additional_results = {
        name: {"contains": "今日休館" in reply, "splitlines": reply.splitlines(), "matches": reply.splitlines() == target}
        for name, reply in additions.items()
    }
    assert [item["matches"] for item in additional_results.values()] == [False] * 6 + [True, True]
    selected_rows = [position + 1 for position, (language, _) in enumerate(notice) if language == "中文"]
    assert selected_rows == [1, 3]
    svg_path = ROOT / "course/figures/rewrite-08-new-8.14-selected-lines.svg"
    png_path = ROOT / "outputs/course-review-20261019/figure-renders-04/d55bd46951f2f6ea24ea2df0e3e682de67a066f4365198046e3b49ef7b5c210c.png"
    assert sha(svg_path.read_bytes()) == "d55bd46951f2f6ea24ea2df0e3e682de67a066f4365198046e3b49ef7b5c210c"
    assert sha(png_path.read_bytes()) == "840bb08f3fd9a711b5a3ebca62da879e291728b52d8670a040c6bda6c57f4b41"
    runtime = {
        "python": sys.version,
        "torch": importlib.metadata.version("torch"),
        "platform": platform.platform(),
        "device": "CPU；算例只用 Python 字串與清單",
    }
    write_new("supplemental-execution.json", {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-14/verify.py",
        "environment": runtime,
        "source_sha256": sha(frozen),
        "source_matches_current_chapter": True,
        "baseline_stdout": stdout,
        "exercise_stdout": exercise_stdout,
        "exercise_match_values": exercise_matches,
        "alternative_targets": choices,
        "selected_row_numbers": selected_rows,
        "additional_cases": additional_results,
        "figure_png_sha256": sha(png_path.read_bytes()),
        "result": "所有有限算例斷言通過；splitlines 比對允許 CRLF 與單個末尾換行，不是逐位元組比對。",
    })
    specifications = [
        ("https://arxiv.org/pdf/2203.02155v1", [9, 12], "Table 3；Figure 4 與圖說", [
            "Fails to follow the correct instruction / task",
            "Satisifies constraint provided in the instruction",
            "Hallucination",
            "Attempts correct instruction",
            "Follows explicit constraints",
        ]),
        ("https://arxiv.org/pdf/2109.01652v5", [3, 4], "Tasks & Templates；Classification with Options；Training Details", [
            "natural language instructions",
            "This makes the model aware of which choices are desired when responding to classification tasks.",
        ]),
        ("https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", [2], "Approach，式 (1) 與任務條件機率段落", [
            "it should condition not only on the input but also on the task to be performed.",
        ]),
        ("https://docs.python.org/3.13/library/stdtypes.html", [], "str.splitlines；str.find；Common Sequence Operations", [
            "Return a list of the lines in the string, breaking at line boundaries.",
        ]),
        ("https://docs.python.org/3.13/tutorial/datastructures.html", [], "List Comprehensions；Tuples and Sequences", [
            "create a subsequence of those elements that satisfy a certain condition.",
        ]),
    ]
    receipts = [retrieve_original(*specification) for specification in specifications]
    write_new("original-sources.json", {
        "environment": runtime,
        "retrieval_tool": subprocess.run(["curl", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0],
        "pdf_text_extractor": "macOS PDFKit，osascript JavaScript for Automation；不安裝套件",
        "sources": receipts,
    })
    print(json.dumps({"result": "所有有限算例與短摘錄存在性檢查通過", "environment": runtime, "source_titles": [item["html_title"] for item in receipts], "artifacts": ["section-execution.json", "supplemental-execution.json", "original-sources.json"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
