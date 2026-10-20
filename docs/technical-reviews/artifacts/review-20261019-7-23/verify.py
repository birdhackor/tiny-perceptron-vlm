"""核對凍結程式的輸出、事件軌跡與假設耗時，保存官方來源收據。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def check_example():
    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/7.23.md").read_bytes()
    code = re.search(rb"```python\n(.*?)\n```", frozen, re.S)[1].decode()
    namespace = {}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(code, "frozen-example", "exec"), namespace)
    expected = [
        ("2+3", "low", 5, 2),
        ("2+3", "medium", 5, 2),
        ("2+3", "high", 5, 2),
        ("訂單", "low", 5, 2),
        ("訂單", "medium", 1, 3),
        ("訂單", "high", 7, 5),
    ]
    rows = []
    for name, effort, answer, rounds in expected:
        observed = namespace["run"](namespace["tasks"][name], effort)
        assert observed == (answer, rounds)
        rows.append({"task": name, "effort": effort, "answer": answer, "rounds": rounds,
                     "allowed": namespace["limits"][effort], "hypothetical_seconds": rounds})
    assert len(captured.getvalue().splitlines()) == 6
    copied = json.loads((HERE / "section-execution.json").read_text())
    assert captured.getvalue() == copied["files"]["stdout.txt"]["content"]
    cases = {"改成會蓋掉前值": ([("加", 8), ("改成", 1)], "high", (1, 2)),
             "空事件不補足額度": ([], "high", (0, 0)),
             "抵達額度就停止": ([("加", 2), ("加", 3), ("改成", 1)], "low", (5, 2))}
    for events, effort, result in cases.values():
        assert namespace["run"](events, effort) == result
    trace, count = [], 0
    for action, amount in namespace["tasks"]["訂單"]:
        count = amount if action == "改成" else count + amount
        trace.append(count)
    assert trace == [2, 5, 1, 5, 7]
    return {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
            "result": "所有精確比較通過；假設每輪一秒，未實際等待或量測服務延遲。",
            "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu"},
            "source_sha256": hashlib.sha256(frozen).hexdigest(),
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "rows": rows,
            "order_trace": trace, "additional_cases": list(cases), "stdout": captured.getvalue()}


def sources():
    specs = [
        ("openai-reasoning", "https://developers.openai.com/api/docs/guides/reasoning.md",
         "Reasoning effort; How reasoning works; Managing the context window",
         "using fewer tokens for simpler tasks and thinking harder for complex tasks.",
         "effort 引導思考投入；名稱與預設依模型而異；推理 token 數隨題目變動，與可見答案分開計量。"),
        ("claude-thinking", "https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost.md",
         "How Claude determines when to think; Effort levels",
         "No level guarantees a thinking block on every request.",
         "模型逐次決定是否思考與深度；effort 是柔性引導，包含工具結果之間的思考；可用檔位需查模型。"),
        ("gemini-current", "https://ai.google.dev/gemini-api/docs/thinking",
         "Controlling thinking; Token limits and max_output_tokens",
         "automatically adjusting the amount of reasoning effort based on the complexity of the request.",
         "目前的 Interactions 文件有動態思考與 thinking_level；硬性輸出 cutoff 和思考投入是不同控制。"),
        ("gemini-generate-content", "https://ai.google.dev/gemini-api/docs/generate-content/thinking?hl=en",
         "Controlling thinking; Thinking levels; Thinking budgets",
         "Setting the thinkingBudget to -1 turns on dynamic thinking",
         "Gemini 2.5 使用 thinkingBudget，表列預設與範圍不同；Gemini 3/4 使用 thinkingLevel；預算不等於精確實際 token 數。"),
        ("openai-response-schema", "https://developers.openai.com/api/reference/resources/responses/methods/create.md",
         "Request body: temperature, text.verbosity, max_output_tokens, max_tool_calls, truncation",
         "What sampling temperature to use, between 0 and 2.",
         "temperature 控制抽樣隨機程度；text.verbosity 控制詳略；max_output_tokens 包含推理與可見 token；工具呼叫另有上限。"),
        ("huginn", "https://arxiv.org/html/2502.05171v2",
         "Abstract; Scaling by Thinking in Continuous Space; A scalable recurrent architecture / Macroscopic Design",
         "Our model works by iterating a recurrent block",
         "v2（2025-02-17）提出循環區塊在 latent space 增加計算，與生成更多中間 token 的方式不同；不能反推任何商用服務採此架構。"),
    ]
    records = []
    for identifier, url, locator, excerpt, note in specs:
        command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45",
                   "--write-out", "\nRECEIPT:%{url_effective}|%{http_code}|%{content_type}", url]
        result = subprocess.run(command, capture_output=True, check=True)
        raw, tail = result.stdout.rsplit(b"\nRECEIPT:", 1)
        effective, status, content_type = tail.decode().split("|", 2)
        decoded = raw.decode()
        if "html" in content_type:
            parser = Text()
            parser.feed(decoded)
            decoded = " ".join(parser.parts)
        normalized = re.sub(r"\s+", " ", decoded)
        assert excerpt in normalized, identifier
        assert len(excerpt.split()) <= 25
        records.append({"id": identifier, "url": url, "effective_url": effective,
                        "command_argv": command, "command": shlex.join(command), "exit_code": result.returncode,
                        "http_status": status, "content_type": content_type,
                        "retrieved_at_utc": datetime.now(UTC).isoformat(), "original_bytes": len(raw),
                        "original_sha256": hashlib.sha256(raw).hexdigest(), "locator": locator,
                        "excerpt": excerpt, "inspection_note": note,
                        "version": "arXiv:2502.05171v2 / 2025-02-17" if identifier == "huginn" else
                                   "無固定版號的官方現行文件；以讀取時間及原始回應 SHA-256 識別。"})
        print(identifier, status, len(raw), locator)
    return {"curl_version": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0],
            "storage": "僅保存收據、短摘錄與核對筆記；第三方全文未寫入檔案。", "sources": records}


if __name__ == "__main__":
    checks = check_example()
    save("independent-check.json", checks)
    save("source-inspection.json", sources())
    print(checks["result"])
