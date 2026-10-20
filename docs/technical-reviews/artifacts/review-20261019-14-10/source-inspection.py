"""以 curl 讀原始論文，只保存查核所需的短摘錄與收據。"""

import hashlib
import html
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(value):
    value = re.sub(
        r'<math\b[^>]*alttext="([^"]*)"[^>]*>.*?</math>',
        lambda match: " " + html.unescape(match[1]) + " ",
        value,
        flags=re.S,
    )
    parser = PlainText()
    parser.feed(value)
    return " ".join(" ".join(parser.parts).split())


base = "https://arxiv.org/"
urls = [base + "html/2309.00071v3", base + "pdf/2309.00071v3"]
received = []
payloads = []
for url in urls:
    command = ["curl", "-fsSL", url]
    raw = subprocess.check_output(command)
    payloads.append(raw)
    received.append(
        {
            "url": url,
            "command": command,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )

document = payloads[0].decode()
selected = {}
for first, following in [("S3.SS2", "S3.SS4"), ("S4.SS1", "S4.SS5"), ("A1.SS3", "A2")]:
    lo = document.index('<section id="' + first + '"')
    hi = document.index('<section id="' + following + '"', lo + 1)
    selected[first] = plain(document[lo:hi])

needles = {
    "boundaries": ("S3.SS2", "The values of \\alpha and \\beta should be tuned on a case-by-case basis."),
    "llama_boundaries": ("S3.SS2", "good values for \\alpha and \\beta are \\alpha=1 and \\beta=32"),
    "definition": ("S3.SS2", 'By the "YaRN method", we refer to a combination of the attention scaling'),
    "fit": ("S3.SS2", "on LLaMA 7b, 13b, 33b and 65b models without fine-tuning."),
    "documents": ("A1.SS3", "Here we fix 896 16 k-token documents from RedPajama"),
    "model": ("A1.SS3", "Fix s=8 , compare the LLaMA 7b perplexity on 896 16 k-token documents"),
    "comparison": ("A1.SS3", "for a suitable t , a sample may obtain better perplexity scores across the extended context window;"),
    "long_evaluation": ("S4.SS1", "4.2 Long Sequence Language Modeling"),
    "retrieval": ("S4.SS1", "4.3 Passkey Retrieval"),
    "short_evaluation": ("S4.SS1", "To test the degradation of models’ short context performance under context extension"),
}
excerpts = []
for name, (section, needle) in needles.items():
    if needle not in selected[section]:
        raise ValueError("原文未找到預期摘錄：" + name)
    excerpts.append({"name": name, "section_id": section, "text": needle})

equations = []
for number in (10, 11, 13, 14, 15):
    match = re.search(
        r'<(table|tbody)\b[^>]*id="S3\.E' + str(number) + r'"[^>]*>.*?</\1>', document, re.S
    )
    if match is None:
        raise ValueError("原文未找到預期公式：" + str(number))
    equations.append({"locator": "S3.E" + str(number), "text": plain(match[0])})

print(
    json.dumps(
        {
            "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-10/source-inspection.py",
            "environment": {"python": platform.python_version(), "device": "CPU；網路讀取原文，沒有模型訓練"},
            "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
            "version": "arXiv:2309.00071v3，2026-02-06",
            "received": received,
            "original_sections_inspected": ["S3.SS2", "S3.SS3", "S4.SS1–S4.SS4", "A1.SS3"],
            "excerpts": excerpts,
            "equations": equations,
            "result": "所有定位與摘錄皆出現在本次 curl 取得的原文；原始全文僅留記憶體。這是來源查核，沒有重做論文模型實驗。",
        },
        ensure_ascii=False,
        indent=2,
    )
)
