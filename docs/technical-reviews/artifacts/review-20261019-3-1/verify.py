"""驗算加權讀取範例，保存執行收據與原始來源的短摘錄。"""

import ast
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
RUN = ROOT / "outputs/reviewer-tools/runs/20261020T073536900818Z"
COMMIT = "5c4886908584029761b579af026dcfb627c84070"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (ARTIFACTS / name).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())


def fetch(url, version, excerpts, note, html=False):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    text = raw.decode("utf-8")
    if html:
        parser = Text()
        parser.feed(text)
        text = "\n".join(parser.parts)
    for item in excerpts:
        assert item["text"] in text, "摘錄必須存在於實際讀取的原始來源"
    return {
        "url": url,
        "version": version,
        "accessed_on": "2026-10-20",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": command,
        "exit_code": completed.returncode,
        "stderr": completed.stderr.decode("utf-8"),
        "response_sha256": sha(raw),
        "response_bytes": len(raw),
        "excerpts": excerpts,
        "inspection_note": note,
        "full_text_saved": False,
    }


torch.set_default_device("cpu")
values = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
before = values.clone()
weights = torch.tensor([0.9, 0.1])
output = weights @ values
reverse_weights = torch.tensor([0.1, 0.9])
reverse_output = reverse_weights @ values
expected = torch.tensor([1.8, 0.4])
reverse_expected = torch.tensor([0.2, 3.6])
assert torch.allclose(output, expected)
assert torch.allclose(reverse_output, reverse_expected)
assert torch.equal(output, torch.matmul(weights, values))
assert torch.equal(values, before)
assert torch.allclose(output, (weights[:, None] * values).sum(dim=0))
assert output.shape == (2,)
assert not any(t.requires_grad for t in (values, weights, output))

decimal_original = [Decimal("0.9") * 2 + Decimal("0.1") * 0,
                    Decimal("0.9") * 0 + Decimal("0.1") * 4]
decimal_reverse = [Decimal("0.1") * 2 + Decimal("0.9") * 0,
                   Decimal("0.1") * 0 + Decimal("0.9") * 4]
contributions = [Decimal("0.9") * 1, Decimal("0.1") * 100]
assert decimal_original == [Decimal("1.8"), Decimal("0.4")]
assert decimal_reverse == [Decimal("0.2"), Decimal("3.6")]
assert contributions == [Decimal("0.9"), Decimal("10.0")]

layer = torch.nn.Linear(2, 1, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[-2.0, 4.0]]))
linear_output = layer(torch.tensor([1.0, 1.0]))
assert linear_output.item() == 2.0
assert layer.weight.sum().item() == 2.0

source = (ROOT / "course/chapters/03.md").read_text(encoding="utf-8")
start = source.index("## 3.2 ")
end = source.index("\n## ", start + 1)
next_section = source[start:end]
next_code = next_section.split("```python\n", 1)[1].split("```", 1)[0]
namespace = {"__name__": "__main__"}
exec(compile(next_code, "content-dependent-weights", "exec"), namespace)
assert namespace["weights"][0] > namespace["weights"][1] > namespace["weights"][2]
tree = ast.parse((RUN / "fence-1.py").read_text())
calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
assert set(calls) == {"torch.tensor", "print", "torch.allclose"}

save("verification.json", {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-1/verify.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version),
                    "device": str(values.device), "platform": platform.platform()},
    "original": {"values": values.tolist(), "weights": weights.tolist(),
                 "shape": list(output.shape), "dtype": str(output.dtype),
                 "output": output.tolist(), "printed": str(output),
                 "allclose": bool(torch.allclose(output, expected)),
                 "elementwise_difference": (output - expected).tolist(),
                 "exact_decimal": [str(x) for x in decimal_original]},
    "reversed": {"weights": reverse_weights.tolist(), "output": reverse_output.tolist(),
                 "allclose": bool(torch.allclose(reverse_output, reverse_expected)),
                 "exact_decimal": [str(x) for x in decimal_reverse],
                 "values_unchanged": bool(torch.equal(values, before))},
    "contribution_counterexample": {"exact_decimal": [str(x) for x in contributions]},
    "linear_counterexample": {"weight": layer.weight.tolist(),
                              "weight_sum": layer.weight.sum().item(),
                              "input": [1.0, 1.0], "output": linear_output.tolist()},
    "original_calls": calls,
    "original_requires_grad": {"values": values.requires_grad,
                               "weights": weights.requires_grad, "output": output.requires_grad},
    "next_section": {"source_file_sha256": sha((ROOT / "course/chapters/03.md").read_bytes()),
                     "code": next_code, "scores": namespace["scores"].tolist(),
                     "weights": namespace["weights"].tolist()},
    "result": "所有斷言成立；只驗證 CPU 的指定算例與程式行為。",
})

copied = {}
for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json",
             "extraction.json", "fence-1.py", "bootstrap.py"):
    raw = (RUN / name).read_bytes()
    copied[name] = {"sha256": sha(raw), "content_utf8": raw.decode("utf-8")}
save("section-execution.json", {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/03.md#3.1" --execute',
    "launcher_stdout": '{"artifact_directory": "/Users/yclin/project/tiny-perceptron-vlm/outputs/reviewer-tools/runs/20261020T073536900818Z", "source_sha256": "7bf9685d97502adcd7bacfddf20585dc33be84b0d1bf4aee09836674f6d95bdf", "python_fences": 1, "svg_references": 0, "execution_exit_code": 0}\n',
    "launcher_exit_code": 0,
    "copied_files": copied,
    "copy_policy": "文字內容逐位元組以 UTF-8 字串保存；每個成員另記原始 bytes 的 SHA-256。",
})

base = f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/"
sources = [
    fetch("https://arxiv.org/html/1706.03762v7", "arXiv:1706.03762v7, 2023-08-02", [
        {"locator": "Attention", "text": "The output is computed as a weighted sum of the values"},
        {"locator": "Scaled Dot-Product Attention, equation (1)", "text": r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^{T}}{\sqrt{d_{k}}})V"},
    ], "閱讀 Attention 與 Scaled Dot-Product Attention：輸出是 value 的加權和，權重由 query 與 key 決定。這裡只審讀取步驟，不把手動比例當作完整 attention。", html=True),
    fetch(base + "_torch_docs.py", f"PyTorch 2.14.1, git {COMMIT}", [
        {"locator": "torch.matmul", "text": "Matrix product of two tensors."},
        {"locator": "torch.allclose", "text": "allclose(input: Tensor, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool"},
    ], "閱讀 matmul 的一維左乘二維規則：暫補一個前軸、矩陣乘法後移除。閱讀 allclose 的逐元素 atol + rtol*abs(other) 條件；它是容忍差核對。"),
    fetch(base + "nn/modules/activation.py", f"PyTorch 2.14.1, git {COMMIT}", [
        {"locator": "Softmax definition", "text": r"\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}"},
    ], "閱讀 Softmax 類別文件：沿指定軸正規化為非負且總和為一；本節讀取比例是這種正規化權重，不能外推到 dropout 後或所有 attention 變體。"),
    fetch(base + "nn/modules/linear.py", f"PyTorch 2.14.1, git {COMMIT}", [
        {"locator": "Linear.forward", "text": "return F.linear(input, self.weight, self.bias)"},
        {"locator": "Linear.reset_parameters", "text": "init.kaiming_uniform_(self.weight, a=math.sqrt(5))"},
    ], "閱讀 Linear 類別文件、建構子、reset_parameters 與 forward：權重是可學習 Parameter，初始化跨負正值；forward 沒有非負投影或和為一的正規化限制。"),
    fetch("https://arxiv.org/html/1902.10186v2", "arXiv:1902.10186v2, NAACL 2019 author manuscript", [
        {"locator": "Adversarial Attention", "text": "quite different attention distributions over inputs would yield essentially the same"},
        {"locator": "Discussion and Conclusions, limitations", "text": "Alternative attention specifications may yield different conclusions"},
    ], "閱讀 Abstract、Preliminaries and Assumptions、Counterfactual Attention Weights、Adversarial Attention 與 Discussion and Conclusions：研究可構造不同權重而預測近同。實驗限特定注意力、分類/NLI/QA 等非 seq2seq 任務；僅支持不能單憑權重宣稱理解，並未證明所有 attention 皆無法解釋。", html=True),
]
save("primary-source-excerpts.json", {
    "sources": sources,
    "retrieval_note": "第三方全文僅在記憶體讀取，未存進 repo。",
    "unused_pdf_attempt": {
        "command": "curl --fail --silent --show-error --location https://aclanthology.org/N19-1357.pdf | pdftotext -layout - - | sed -n '1,120p'",
        "output": "zsh:1: command not found: pdftotext\ncurl: (23) Failure writing output to destination, passed 16384 returned 0\n",
        "resolution": "改用同篇作者 arXiv HTML 原文，沒有把 PDF 擷取失敗當作來源已讀的證據。",
    },
})
print(json.dumps({"result": "驗算與原始來源摘錄保存完成", "device": str(values.device),
                  "torch": str(torch.__version__)}, ensure_ascii=False))
