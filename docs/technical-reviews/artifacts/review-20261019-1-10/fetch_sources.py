"""讀取官方原文；全文只在記憶體，保存少量摘錄與取得收據。"""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


SOURCES = [
    ("autograd", "https://docs.pytorch.org/docs/2.14/notes/autograd.html", [
        ("How autograd encodes the history", "Autograd is a reverse automatic differentiation system."),
        ("Setting requires_grad", "During the forward pass, an operation is only recorded"),
    ]),
    ("backward", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.backward.html", [
        ("Tensor.backward", "The graph is differentiated using the chain rule."),
    ]),
    ("detach", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html", [
        ("Tensor.detach", "Returns a new Tensor, detached from the current graph."),
    ]),
    ("square", "https://docs.pytorch.org/docs/2.14/generated/torch.square.html", [
        ("torch.square", "Returns a new tensor with the square of the elements"),
    ]),
    ("item", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.item.html", [
        ("Tensor.item", "Returns the value of this tensor as a standard Python number."),
    ]),
    ("chain", "https://openstax.org/books/calculus-volume-1/pages/3-6-the-chain-rule", [
        ("Rule: The Chain Rule", "Rule: The Chain Rule"),
    ]),
    ("derivative", "https://openstax.org/books/calculus-volume-1/pages/3-1-defining-the-derivative", [
        ("The Derivative of a Function at a Point", "The Derivative of a Function at a Point"),
    ]),
    ("paths", "https://openstax.org/books/calculus-volume-3/pages/4-5-the-chain-rule", [
        ("Chain Rule for One Independent Variable", "Chain Rule for One Independent Variable"),
    ]),
]

records = []
for identifier, url, locations in SOURCES:
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=True)
    parser = Text()
    parser.feed(result.stdout.decode("utf-8"))
    body = " ".join(" ".join(parser.parts).split())
    snippets = []
    inspection = []
    for locator, search in locations:
        index = body.find(search)
        if index < 0:
            raise ValueError(f"原文定位不存在：{identifier} / {search}")
        words = body[index:].split()
        snippets.append({"locator": locator, "excerpt": " ".join(words[:12])})
        inspection.append({"locator": locator, "original_text_for_inspection": body[max(0, index - 100):index + 2200]})
    records.append({
        "id": identifier,
        "url": url,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "raw_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "raw_bytes": len(result.stdout),
        "command_argv": command,
        "curl_exit_code": result.returncode,
        "excerpts": snippets,
    })
    if "--inspect" in sys.argv:
        print(json.dumps({"id": identifier, "url": url, "inspection": inspection}, ensure_ascii=False, indent=2))

if "--inspect" not in sys.argv:
    print(json.dumps({"sources": records, "full_text_saved": False}, ensure_ascii=False, indent=2))
