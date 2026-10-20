"""以 curl 讀取原始來源，只保存定位摘錄與完整回應的雜湊。"""

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.ignore = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.ignore += 1
        if tag in {"p", "div", "section", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.ignore -= 1

    def handle_data(self, data):
        if not self.ignore:
            self.parts.append(data)


requests = [
    ("pytorch_functional", "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/functional.py", ["def relu(", "gelu = _add_docstr("], False),
    ("pytorch_activation", "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/activation.py", ["class ReLU(", "class GELU("], False),
    ("pytorch_tensor", "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/_torch_docs.py", ["torch.tensor,", "torch.square,"], False),
    ("gelu_paper", "https://arxiv.org/html/1606.08415v5", ["the standard normal", "3.1 MNIST Classification"], True),
    ("deep_learning_book", "https://www.deeplearningbook.org/contents/mlp.html", ["Clearly, we must use a nonlinear function"], True),
    ("pytorch_benchmark", "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/utils/benchmark/utils/timer.py", ["class Timer:"], False),
]

receipt = {"accessed_at_utc": datetime.now(UTC).isoformat(), "sources": []}
for name, url, needles, html in requests:
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "30", url]
    result = subprocess.run(command, capture_output=True, check=False)
    raw = result.stdout
    body = raw.decode("utf-8", errors="replace")
    if html:
        parser = PlainText()
        parser.feed(body)
        body = re.sub(r"[ \t\r\f\v]+", " ", "".join(parser.parts))
        body = re.sub(r"\n+", "\n", body)
    matches = []
    for needle in needles:
        start = body.lower().find(needle.lower())
        if name == "gelu_paper" and needle == "3.1 MNIST Classification":
            start = body.lower().find(needle.lower(), start + len(needle))
        if start >= 0:
            before = 550 if name == "deep_learning_book" else 100
            length = 1300 if name == "pytorch_benchmark" else 2000
            matches.append({"needle": needle, "offset_in_inspected_text": start, "excerpt": body[max(0, start - before):start + length]})
    receipt["sources"].append({"id": name, "url": url, "command": command, "curl_exit_code": result.returncode, "stderr": result.stderr.decode(), "response_sha256": hashlib.sha256(raw).hexdigest(), "response_bytes": len(raw), "excerpts": matches})
print(json.dumps(receipt, ensure_ascii=False, indent=2))
