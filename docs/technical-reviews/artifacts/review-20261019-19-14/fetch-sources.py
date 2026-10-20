"""用 curl 直接讀官方文件，只保存短摘錄與版本收據。"""

import hashlib
import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


specifications = [
    {
        "id": "grouped_evaluation",
        "url": "https://scikit-learn.org/stable/modules/cross_validation.html",
        "locator": "Cross-validation: evaluating estimator performance；開頭的 validation/test 說明與 Group K-fold",
        "excerpts": [
            "same group is not represented in both testing and training sets.",
            "knowledge about the test set can “leak” into the model",
        ],
        "inspection_note": "讀取整個相關段落：同群受試者的樣本需分組留出；反覆用 test 選參數會洩漏，應另留 validation。此原則支持以底層世界作群組與保留未用過的最後測試；不保證本專案家族定義等於語義等價類。",
    },
    {
        "id": "temperature_scaling",
        "url": "https://scikit-learn.org/stable/modules/calibration.html",
        "locator": "Temperature Scaling",
        "excerpts": [
            "The parameter \\(T\\) is learned by minimizing log_loss, i.e. cross-entropy loss, on a hold-out (calibration) set.",
        ],
        "inspection_note": "核對 softmax(z/T)、以 hold-out calibration set 最小化 CE 選 T，及 argmax 不變的條件。本文僅主張 calibration 的用途，未宣稱必然提升測試品質。",
    },
    {
        "id": "hashlib",
        "url": "https://docs.python.org/3.13/library/hashlib.html",
        "locator": "Hash algorithms；hash.hexdigest",
        "excerpts": [
            "use sha256() to create a SHA-256 hash object.",
            "Return the digest",
        ],
        "inspection_note": "核對 SHA-256 接收 bytes、digest/hexdigest 契約；64 位十六進位長度另由執行值與 256/4 推導核對。固定輸入得到固定 hash 不表示有限樣本桶數必然相等。",
    },
]

records = []
for spec in specifications:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", spec["url"]]
    raw = subprocess.check_output(command)
    html = raw.decode("utf-8")
    parser = Text()
    parser.feed(html)
    text = " ".join("".join(parser.parts).split())
    found = []
    for excerpt in spec["excerpts"]:
        normalized = " ".join(excerpt.split())
        if normalized not in text:
            raise ValueError("官方文件找不到預期摘錄：" + spec["id"] + " " + normalized)
        found.append(normalized)
    title = re.search(r"<title>(.*?)</title>", html, re.DOTALL).group(1)
    records.append({**spec, "title": title, "command": command, "accessed_at": datetime.now(UTC).isoformat(), "original_sha256": hashlib.sha256(raw).hexdigest(), "original_bytes": len(raw), "checked_original": True, "excerpts": found, "result": "curl exit 0；相關原文已讀取；全文僅在記憶體中解析，未寫入 repo。"})

print(json.dumps({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-14/fetch-sources.py", "environment": {"python": platform.python_version(), "transport": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]}, "sources": records}, ensure_ascii=False, indent=2))
