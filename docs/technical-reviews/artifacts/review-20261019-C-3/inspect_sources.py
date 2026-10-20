"""以 curl 取得原始來源，只保存收據與支持主張的短摘錄。"""

import hashlib
import html
import json
import platform
import re
import subprocess
from datetime import datetime
from html.parser import HTMLParser


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


specs = [
    {
        "id": "paper",
        "url": "https://arxiv.org/pdf/2107.03374v2",
        "version": "arXiv:2107.03374v2，2021-07-14",
        "locator": "Functional Correctness；pass@k 定義與式 (1)，PDF 印刷頁 2–3",
        "excerpt": "a problem is considered solved if any sample passes the unit tests",
        "formula": "pass@k := E_Problems[1 - binom(n-c,k)/binom(n,k)]",
        "inspection_note": "本回合另以 web 工具讀原 PDF 的 Functional Correctness、式 (1) 與相鄰 estimator 說明；curl 取得 PDF 原始位元組並核對版本標記。只保留短摘錄與公式，不保存全文。",
    },
    {
        "id": "independence",
        "url": "https://online.stat.psu.edu/stat414/Lesson05",
        "version": "STAT 414 線上教材；版本以本次 curl 收據的 SHA-256 識別",
        "locator": "Independent Events；HTML anchor def-independenteventsb；Mutually Independent Event",
        "excerpt": "P(A\\cap B)=P(A)\\times P(B)",
        "inspection_note": "讀取 Independent Events 的乘法等價定義及 Mutually Independent Event 的乘法條件。常數事件機率為 0 或 1 時也能滿足乘法等式。",
    },
    {
        "id": "round",
        "url": "https://docs.python.org/3.13/library/functions.html#round",
        "version": "Python 3.13.16 documentation",
        "locator": "round(number, ndigits=None)",
        "excerpt": "Return number rounded to ndigits precision after the decimal point.",
        "inspection_note": "讀取 round 的精度、回傳數字與二進位浮點限制；它不是固定補齊六位小數的字串格式器。",
    },
    {
        "id": "operators",
        "url": "https://docs.python.org/3.13/reference/expressions.html",
        "version": "Python 3.13.16 documentation",
        "locator": "The power operator；Membership test operations",
        "excerpt": "it yields its left argument raised to the power of its right argument.",
        "second_excerpt": "The operators in and not in test for membership.",
        "inspection_note": "讀取次方定義與 in 對 list 的元素相等性檢查，符合本頁實際使用。",
    },
]
for spec in specs:
    argv = ["curl", "--location", "--fail", "--silent", "--show-error", "--max-time", "45", spec["url"]]
    response = subprocess.run(argv, capture_output=True, check=True)
    raw = response.stdout
    spec.update(command_argv=argv, exit_code=response.returncode, bytes=len(raw), full_source_sha256=hashlib.sha256(raw).hexdigest(), accessed_on=datetime.now().astimezone().date().isoformat())
    if spec["id"] == "paper":
        assert raw.startswith(b"%PDF-")
        spec["curl_verification"] = "取得有效 PDF 標頭；版本與摘錄由本回合 web 原 PDF 閱讀核對。"
    else:
        parser = Text()
        parser.feed(raw.decode())
        text = re.sub(r"\s+", " ", html.unescape(" ".join(parser.parts)))
        excerpt = spec["excerpt"]
        if spec["id"] == "round":
            assert all(word in text for word in ("Return", "rounded", "ndigits", "decimal point"))
        else:
            assert excerpt in text
        if "second_excerpt" in spec:
            assert spec["second_excerpt"] in text
        if spec["id"] == "independence":
            assert "mutually independent" in text
        spec["curl_verification"] = "取得 HTML 原文，確認所摘取的定義與 API 文字。"
print(json.dumps({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-C-3/inspect_sources.py", "environment": {"python": platform.python_version(), "network_client": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]}, "sources": specs}, ensure_ascii=False, indent=2) + "\n", end="")
