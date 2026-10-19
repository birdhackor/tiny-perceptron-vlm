"""以 curl 讀原始網站；只保存收據與必要的短摘錄。"""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


sources = [
    {
        "id": "python_counter",
        "url": "https://docs.python.org/3.13/library/collections.html",
        "locators": ["collections.Counter", "collections.Counter.fromkeys", "collections.Counter.update"],
        "markers": ["A Counter is a", "usual dictionary methods are available", "Elements are counted from an iterable or added-in"],
        "short_quotes": ["not implemented", "adds counts instead of replacing them"],
    },
    {
        "id": "python_float",
        "url": "https://docs.python.org/3.13/tutorial/floatingpoint.html",
        "locators": ["Floating-Point Arithmetic: Issues and Limitations", "Representation Error"],
        "markers": ["most decimal fractions cannot be represented", "summing three values of"],
        "short_quotes": ["most decimal fractions cannot be represented exactly as binary fractions"],
    },
    {
        "id": "d2l_language_models",
        "url": "https://d2l.ai/chapter_recurrent-neural-networks/language-model.html",
        "locators": ["Markov Models and n-grams", "Word Frequency"],
        "markers": ["The probability formulae that involve", "The probability of words can be calculated", "are the number of occurrences of singletons"],
        "short_quotes": ["unigram"],
    },
    {
        "id": "psu_pmf",
        "url": "https://online.stat.psu.edu/stat414/Lesson07",
        "locators": ["Probability Mass Functions", "probability mass function requirements"],
        "markers": ["probability mass function", "sum"],
        "short_quotes": [],
    },
    {
        "id": "sklearn_baseline",
        "url": "https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html",
        "locators": ["DummyClassifier", "strategy"],
        "markers": ["makes predictions that ignore the input features", "simple baseline", "most_frequent"],
        "short_quotes": ["simple baseline"],
    },
]

receipts = []
for item in sources:
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", item["url"]]
    result = subprocess.run(command, capture_output=True, check=True)
    raw = result.stdout
    html = raw.decode("utf-8")
    parser = Text()
    parser.feed(html)
    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    title = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", html, re.S).group(1)).strip()
    print("原始來源：", item["url"])
    print("實際頁面標題：", title)
    for marker in item["markers"]:
        index = text.lower().find(marker.lower())
        if index < 0:
            raise ValueError("原文沒有定位字串：" + marker)
        print("定位：", marker)
        print(text[max(0, index - 100):index + 1200])
    for quote in item["short_quotes"]:
        assert quote in text
    receipts.append({
        "id": item["id"], "url": item["url"], "title": title,
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": command, "curl_exit_code": result.returncode,
        "original_sha256": hashlib.sha256(raw).hexdigest(), "original_bytes": len(raw),
        "locators": item["locators"], "short_quotes": item["short_quotes"],
        "full_original_saved": False,
    })

target = Path(__file__).with_name("sources-receipt.json")
with target.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-4/fetch_sources.py",
        "storage_policy": "全文僅在記憶體讀取；持久證據只保存 URL、實際標題、讀取時刻、原始回應 SHA-256 與短摘錄。",
        "sources": receipts,
    }, ensure_ascii=False, indent=2) + "\n")
print("來源收據：", target)
