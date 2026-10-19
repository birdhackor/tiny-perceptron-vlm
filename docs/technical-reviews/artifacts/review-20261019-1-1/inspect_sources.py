"""以 curl 讀官方原文，僅保存短摘錄、版本、定位與讀取收據。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess


ARTIFACTS = Path(__file__).resolve().parent


class Plain(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


PAGES = [
    ("python-builtins", "https://docs.python.org/3.13/library/functions.html", [
        ("enumerate", "enumerate ( iterable , start = 0 )", 1100),
        ("sorted", "sorted ( iterable , /", 1000),
        ("set", "class set ( iterable", 500),
    ], "Return a new sorted list from the items in iterable", "enumerate 預設從 0 計數；sorted 預設直接比較項目，reverse=True 反轉比較；set 從 iterable 取得元素。"),
    ("python-types", "https://docs.python.org/3.13/library/stdtypes.html", [
        ("set", "A set object is an unordered collection", 700),
        ("str.join", "str. join ( iterable", 700),
        ("dict.__getitem__", "Return the item of d with key k", 1000),
        ("sequence indexing", "s[i] ith item", 700),
    ], "A set object is an unordered collection of distinct hashable objects.", "set 元素互異；str.join 以接收者當分隔字串串接；d[key] 查不到鍵引發 KeyError，但 dict 子類的 __missing__ 可另定義；本例是普通 dict。序列索引從 0 起。"),
    ("python-expressions", "https://docs.python.org/3.13/reference/expressions.html", [
        ("value-comparisons", "Strings (instances of str ) compare", 700),
        ("comprehensions", "The comprehension consists", 900),
        ("list-displays", "When a comprehension is supplied", 700),
    ], "Strings (instances of str ) compare lexicographically using the numerical Unicode code points", "str 按 Unicode 碼點比較；comprehension 按 for 子句取值，每次產生一個新項目。本例每項只有單一碼點，所以由小到大的數值順序直接成立。"),
    ("python-assert", "https://docs.python.org/3.13/reference/simple_stmts.html", [
        ("assert", "The simple form, assert expression", 1200),
    ], "The current code generator emits no code for an assert statement when optimization is requested at compile time.", "assert 的普通執行等價於條件為假時 raise AssertionError，成立時繼續；-O 會省略 assert。本例執行沒有啟用最佳化。"),
    ("python-unicode", "https://docs.python.org/3.13/howto/unicode.html", [
        ("definitions", "The Unicode standard describes", 1250),
        ("string-type", "the str type is described", 800),
    ], "The Unicode standard describes how characters are represented by code points", "Unicode 以整數碼點表達字元；碼點不是字義或筆畫排序。Python 字串是 Unicode 碼點序列，視覺上的字元與碼點不必永遠一對一。"),
    ("hf-tokenizers", "https://huggingface.co/docs/tokenizers/v0.20.3/en/quicktour", [
        ("vocabulary ordering", "The order in which you write the special tokens list matters", 650),
        ("encoding", "We can now use our tokenizer", 1600),
        ("encoding ids", "output.ids", 900),
        ("decoding", "decode method", 1300),
    ], "The order in which you write the special tokens list matters", "原文將 token 對應到 vocabulary 內的 ID，ID 由表內位置決定；decode 將 ID 轉回 tokens 再合成文字。未知 token、normalization、特殊 token 的過濾可能影響一般 tokenizer 的還原，本節純字元表沒有這些處理。"),
]


def inspect(item):
    identifier, url, locators, excerpt, note = item
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    parser = Plain()
    parser.feed(raw.decode("utf-8"))
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    assert excerpt in text
    assert len(excerpt.split()) <= 25
    passages = []
    for locator, needle, width in locators:
        at = text.find(needle)
        passages.append({"locator": locator, "needle": needle, "found": at >= 0, "text": text[at:at + width] if at >= 0 else ""})
    title = re.search(r"<title>(.*?)</title>", raw.decode("utf-8"), re.S).group(1)
    receipt = {
        "id": identifier, "url": url, "command_argv": command,
        "exit_code": completed.returncode, "stderr": completed.stderr.decode(),
        "retrieved_at": datetime.now(UTC).isoformat(), "html_title": title,
        "original_bytes": len(raw), "original_sha256": hashlib.sha256(raw).hexdigest(),
        "version": "Tokenizers v0.20.3 文件；固定版本 URL" if identifier == "hf-tokenizers" else "Python 3.13.16 官方文件；Unicode HOWTO Release 1.12" if identifier == "python-unicode" else "Python 3.13.16 官方文件",
        "locators": [{"locator": p["locator"], "needle": p["needle"], "found": p["found"]} for p in passages],
        "short_excerpt": excerpt, "inspection_note": note,
        "checked_original": True,
    }
    return receipt, passages


with ThreadPoolExecutor(max_workers=6) as pool:
    inspected = list(pool.map(inspect, PAGES))
with (ARTIFACTS / "original-sources.json").open("x", encoding="utf-8") as stream:
    json.dump({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ARTIFACTS.parents[3])), "sources": [item[0] for item in inspected], "storage_policy": "第三方全文只在程序記憶體中讀取；保存 URL、版本、原檔 SHA-256、定位、短摘錄與自行撰寫的核對說明。"}, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
for receipt, passages in inspected:
    print(json.dumps({"receipt": receipt, "inspected_passages": passages}, ensure_ascii=False, indent=2))
