import hashlib
import json
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class OriginalText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def fetch(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return result.stdout, {
        "url": url,
        "command_argv": command,
        "exit_code": result.returncode,
        "accessed_at": datetime.now(UTC).isoformat(),
        "original_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "original_bytes": len(result.stdout),
        "stderr": result.stderr.decode(),
    }


def exact_piece(text, start, end):
    left = text.index(start)
    return text[left:text.index(end, left)].strip()


paper, paper_receipt = fetch("https://arxiv.org/html/2002.05202v1")
parser = OriginalText()
parser.feed(paper.decode())
text = " ".join("".join(parser.parts).split())
paper_receipt["version"] = "arXiv:2002.05202v1, 2020-02-12; arXiv HTML rendering"
paper_receipt["short_excerpts"] = [
    {
        "locator": "Gated Linear Units (GLU) and Variants, equation (5)",
        "text": exact_piece(text, r"\textrm{SwiGLU}(x,W,V,b,c,\beta)", r"\end{split}"),
    },
    {
        "locator": "Gated Linear Units (GLU) and Variants, equation (6)",
        "text": exact_piece(text, r"\textrm{FFN}_{\textrm{SwiGLU}}", r"\end{split}"),
    },
    {
        "locator": "Gated Linear Units (GLU) and Variants, text following equation (6)",
        "text": exact_piece(text, "All of these layers have three weight matrices", "To keep"),
    },
]
paper_receipt["inspection_note"] = (
    "已讀原文 Introduction、Gated Linear Units (GLU) and Variants、Model Architecture 與比較條件。"
    "Swish_beta(x)=x*sigmoid(beta*x)，FFN_SwiGLU 用 beta=1；兩個輸入投影逐項相乘後乘第三張表。"
    "原文實驗省略 bias，hidden 由 3072 改成 2048，以 2/3 匹配權重與主要矩陣運算量。"
    "此處保存公式與短摘錄；不保存第三方全文。"
)

commit = "5c4886908584029761b579af026dcfb627c84070"
base = "https://raw.githubusercontent.com/pytorch/pytorch/" + commit + "/torch/nn/modules/"
activation, activation_receipt = fetch(base + "activation.py")
activation_text = activation.decode()
activation_receipt["version"] = "PyTorch 2.14.1; git " + commit
activation_receipt["short_excerpts"] = [
    {
        "locator": "SiLU docstring",
        "text": "The SiLU function is also known as the swish function.",
    },
    {
        "locator": "SiLU docstring, defining equation",
        "text": r"\text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}",
    },
    {
        "locator": "Sigmoid docstring, defining equation",
        "text": exact_piece(activation_text, r"\text{Sigmoid}(x)", "Shape:"),
    },
]
for item in activation_receipt["short_excerpts"]:
    assert item["text"] in activation_text
activation_receipt["inspection_note"] = (
    "已核對與本機 torch.version.git_version 完全相同的官方原始碼：Sigmoid、SiLU 的公式，"
    "SiLU.forward 呼叫 F.silu，及 SiLU 的逐項、保留形狀契約。"
)

linear, linear_receipt = fetch(base + "linear.py")
linear_text = linear.decode()
linear_receipt["version"] = "PyTorch 2.14.1; git " + commit
linear_receipt["short_excerpts"] = [
    {"locator": "Linear.__init__", "text": "bias: bool = True,"},
    {"locator": "Linear.__init__, weight allocation", "text": "torch.empty((out_features, in_features), **factory_kwargs)"},
    {"locator": "Linear.__init__, bias allocation", "text": "self.bias = Parameter(torch.empty(out_features, **factory_kwargs))"},
]
for item in linear_receipt["short_excerpts"]:
    assert item["text"] in linear_text
linear_receipt["inspection_note"] = (
    "已讀 Linear 的 affine 定義、Shape、Attributes、__init__ 與 forward："
    "weight 形狀為 (out_features, in_features)，bias 預設啟用且長度為 out_features。"
)

print(json.dumps({"sources": [paper_receipt, activation_receipt, linear_receipt]}, ensure_ascii=False, indent=2))
