"""讀取官方原文，只保存來源收據與支持主張的小段摘錄。"""

import hashlib
import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"style", "script"}:
            self.hidden += 1
        self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"style", "script"}:
            self.hidden -= 1
        self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


specs = [
    {
        "id": "python_json",
        "url": "https://docs.python.org/3.13/library/json.html",
        "needles": ["Deserialize s", "JSONDecoder", "object dict", "true True", "false False"],
        "excerpts": ["JSON (JavaScript Object Notation)", "object dict", "number (int) int", "true True", "false False"],
    },
    {
        "id": "python_types",
        "url": "https://docs.python.org/3.13/library/stdtypes.html",
        "needles": ["bool is a subclass of int", "get ( key", "Return the value for key"],
        "excerpts": ["bool is a subclass of int", "Return the value for key if key is in the dictionary, else default"],
    },
    {
        "id": "python_builtins",
        "url": "https://docs.python.org/3.13/library/functions.html",
        "needles": ["isinstance (", "direct, indirect", "return the type of an object"],
        "excerpts": ["return the type of an object", "direct, indirect, or virtual"],
    },
    {
        "id": "rfc_json",
        "url": "https://www.rfc-editor.org/rfc/rfc8259.txt",
        "needles": ["An object is an unordered", "JSON text is a serialized value", "grammar"],
        "excerpts": ["JSON text is a serialized value.", "An object is an unordered collection of zero or more name/value pairs"],
    },
    {
        "id": "cross_validation",
        "url": "https://scikit-learn.org/stable/modules/cross_validation.html",
        "needles": ["When evaluating different settings", "validation set", "final evaluation"],
        "excerpts": ["there is still a risk of overfitting on the test set", "final evaluation can be done on the test set"],
    },
    {
        "id": "helm",
        "url": "https://crfm.stanford.edu/2022/11/17/helm.html",
        "needles": ["metrics", "trade", "prompts", "standard"],
        "excerpts": ["accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency"],
    },
]

receipts = []
for spec in specs:
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", spec["url"]]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    original = raw.decode("utf-8")
    if spec["url"].endswith(".txt"):
        body = original
        title = "RFC 8259, December 2017"
    else:
        parser = PlainText()
        parser.feed(original)
        body = unescape("".join(parser.parts))
        title_match = re.search(r"<title[^>]*>(.*?)</title>", original, re.S | re.I)
        title = unescape(re.sub("<[^>]+>", "", title_match[1])) if title_match else ""
    body = re.sub(r"\s+", " ", body).strip()
    excerpts = []
    for excerpt in spec["excerpts"]:
        if excerpt not in body:
            raise ValueError(f"摘錄未出現在原文：{spec['id']}: {excerpt}")
        excerpts.append(excerpt)
    receipt = {
        "id": spec["id"],
        "url": spec["url"],
        "title": title,
        "accessed_at": datetime.now(UTC).isoformat(),
        "command_argv": command,
        "exit_code": completed.returncode,
        "stderr": completed.stderr.decode(),
        "original_sha256": hashlib.sha256(raw).hexdigest(),
        "original_bytes": len(raw),
        "excerpts": excerpts,
        "checked_original": True,
        "storage_policy": "全文只在記憶體讀取；檔案只保存收據與短摘錄。",
    }
    receipts.append(receipt)
    print(json.dumps({"id": spec["id"], "title": title, "sha256": receipt["original_sha256"]}, ensure_ascii=False))
    for needle in spec["needles"]:
        index = body.find(needle)
        print(json.dumps({"needle": needle, "found": index >= 0, "inspection": body[max(0, index - 100):index + 750] if index >= 0 else ""}, ensure_ascii=False))

output = Path(__file__).with_name("source-excerpts.json")
with output.open("x", encoding="utf-8") as target:
    json.dump({"environment": {"python": platform.python_version()}, "sources": receipts}, target, ensure_ascii=False, indent=2)
    target.write("\n")
