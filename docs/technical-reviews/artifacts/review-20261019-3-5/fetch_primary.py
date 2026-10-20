"""用 curl 讀取原始來源，只保存定位、雜湊與短摘錄。"""

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.ignored += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.ignored -= 1

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def fetch(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=True)
    raw = result.stdout
    return raw.decode(), {
        "url": url,
        "command_argv": command,
        "read_at_utc": datetime.now(UTC).isoformat(),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw),
    }


records = []
for name in ["torch.randint", "torch.Tensor.float", "torch.mul", "torch.sum", "torch.std", "torch.Tensor.item", "torch.manual_seed", "torch.nn.functional.softmax"]:
    url = f"https://docs.pytorch.org/docs/2.14/generated/{name}.html"
    html, receipt = fetch(url)
    match = re.search(r'"articleBody":\s*("(?:\\.|[^"\\])*")', html)
    body = json.loads(match[1])
    print(name + "\n" + body[:4500] + "\n")
    fragments = {
        "torch.randint": "Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).",
        "torch.Tensor.float": "self.float() is equivalent to self.to(torch.float32).",
        "torch.mul": "Multiplies input by other.",
        "torch.sum": "Returns the sum of each row of the input tensor in the given dimension dim.",
        "torch.std": "Defaults to Bessel’s correction, correction=1.",
        "torch.Tensor.item": "Returns the value of this tensor as a standard Python number.",
        "torch.manual_seed": "Sets the seed for generating random numbers on all devices.",
        "torch.nn.functional.softmax": "It is applied to all slices along dim",
    }
    excerpt = fragments[name]
    if excerpt not in body:
        raise ValueError("來源短摘錄未出現在原文：" + name)
    receipt.update(title=name, version="PyTorch 2.14 documentation", locator=name, excerpt=excerpt)
    if name == "torch.std":
        receipt["inspection_note"] = "已讀標準差公式與 correction 參數：預設 correction=1，以 N-1 作樣本變異數分母。"
    records.append(receipt)

url = "https://arxiv.org/html/1706.03762v7"
html, receipt = fetch(url)
parser = Text()
parser.feed(html)
body = " ".join(" ".join(parser.parts).split())
start = body.index("We call our particular attention")
end = body.index("Applications of Attention in our Model", start)
print("Attention Is All You Need\n" + body[start:end] + "\n")
receipt.update(
    title="Attention Is All You Need",
    version="arXiv:1706.03762v7",
    locator="Scaled Dot-Product Attention, equation (1), footnote on variance; Multi-Head Attention",
    excerpts=["The input consists of queries and keys of dimension", "has mean", "and variance", "For each of these we use"],
    inspected_formula="Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V; d_k=d_model/h for each head in the paper's configuration",
    inspection_note="已讀輸入寬度、公式、獨立零平均單位變異數的假設與變異數推算，以及每頭投影後的寬度；未將作者對梯度的推測當成本頁實測證據。",
)
assert all(excerpt in body for excerpt in receipt["excerpts"])
records.append(receipt)

url = "https://www.probabilitycourse.com/chapter3/3_2_4_variance.php"
html, receipt = fetch(url)
parser = Text()
parser.feed(html)
body = " ".join(" ".join(parser.parts).split())
start = body.index("The variance of a random variable")
end = body.index("We will prove this theorem", start)
print("Variance\n" + body[start:end] + "\n")
receipt.update(
    title="Introduction to Probability, Statistics, and Random Processes: Variance",
    version="作者公開線上教材，讀取當日版本；response_sha256 固定實際回應",
    locator="Variance and standard deviation definitions; scaling and independent-sum theorems",
    excerpt="The standard deviation of a random variable",
    inspected_formula="Var(X)=E[(X-E[X])^2]; SD(X)=sqrt(Var(X)); Var(aX+b)=a^2 Var(X); Var(sum X_i)=sum Var(X_i) for independent X_i",
    inspection_note="已讀定義、尺度轉換的證明及獨立變數加總定理；同單位指 SD 與 X 的單位相同，不是一般誤差界。",
)
assert receipt["excerpt"] in body
records.append(receipt)

target = Path(__file__).with_name("primary-sources.json")
with target.open("x", encoding="utf-8") as stream:
    json.dump({"sources": records}, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
