"""以 curl 讀原始來源，只保留收據、短摘錄與核對筆記。"""

import hashlib
import json
import platform
import subprocess
from html.parser import HTMLParser


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


sources = [
    {
        "id": "attention_paper",
        "url": "https://arxiv.org/html/1706.03762v7",
        "version": "arXiv:1706.03762v7",
        "locator": "Attention 與 Scaled Dot-Product Attention；公式 Attention(Q,K,V)",
        "markers": ["An attention function can be described", "We compute the dot products of the query with all keys", "Dot-product attention is identical to our algorithm"],
        "excerpt": "Dot-product attention is identical to our algorithm, except for the scaling factor",
        "inspection_note": "原文把 query、key、value 定為向量，以 query 與對應 key 的相容度產生 value 權重；縮放點積版本先算 query 與所有 key 的點積，除以 key 維度平方根，再 softmax，最後混合 V。原文也明確區分未縮放點積版本。本節刻意停在未縮放的權重階段，角色及計算順序相符。",
    },
    {
        "id": "torch_matmul",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.matmul.html",
        "version": "PyTorch 2.14 documentation；本機執行 2.14.1",
        "locator": "torch.matmul：依輸入維數決定的 vector-matrix product",
        "markers": ["Matrix product of two tensors.", "If the first argument is 1-dimensional and the second argument is 2-dimensional"],
        "excerpt": "Matrix product of two tensors.",
        "inspection_note": "讀取一維乘二維分支：先在 query 前補長度 1 的維度，完成矩陣乘法後移除。因此本例 [2] @ [2,3] 得 [3]；各輸出是 query 與候選 key 的點積。另讀過本機 torch.matmul docstring，分支規則相同。",
    },
    {
        "id": "torch_tensors",
        "url": "https://docs.pytorch.org/docs/2.14/tensors.html",
        "version": "PyTorch 2.14 documentation；本機執行 2.14.1",
        "locator": "Tensor.T 屬性；Tensor.softmax 方法",
        "markers": ["Returns a view of this tensor with its dimensions reversed.", "Alias for torch.nn.functional.softmax()"],
        "excerpt": "Returns a view of this tensor with its dimensions reversed. Alias for torch.nn.functional.softmax()",
        "inspection_note": "Tensor.T 反轉軸順序，對本例二維 keys 恰為列欄交換；非二維使用另有棄用警告，本節沒有這類使用。Tensor.softmax 明定是 functional.softmax 的別名。另讀過本機 Tensor.T docstring，與此相符。",
    },
    {
        "id": "torch_softmax",
        "url": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html",
        "version": "PyTorch 2.14 documentation；本機執行 2.14.1",
        "locator": "torch.nn.functional.softmax：定義公式、dim 參數及正規化範圍",
        "markers": ["Apply a softmax function.", "sum to 1.", "A dimension along which softmax will be computed."],
        "excerpt": "A dimension along which softmax will be computed.",
        "inspection_note": "公式為 exp(x_i) 除以同一切片各項 exp 的總和，沿 dim 正規化為 [0,1] 內且和為 1 的元素。本節 scores 唯一候選軸是 dim=0，與呼叫一致。另讀過本機 functional.softmax docstring。對本例有限小分數不涉及 Inf、NaN 或 underflow；實際浮點和允許捨入誤差。",
    },
]

for item in sources:
    command = ["curl", "-fLsS", "--max-time", "40", item["url"]]
    completed = subprocess.run(command, capture_output=True, check=True)
    parser = TextParser()
    parser.feed(completed.stdout.decode("utf-8"))
    text = " ".join(" ".join(parser.parts).split())
    for marker in item.pop("markers"):
        assert marker in text, marker
    assert item["excerpt"] in text or all(part in text for part in item["excerpt"].split(" Alias ") )
    assert len(item["excerpt"].split()) <= 25
    item.update(command=command, exit_code=completed.returncode, accessed_on="2026-10-20", checked_original=True, response_bytes=len(completed.stdout), response_sha256=hashlib.sha256(completed.stdout).hexdigest())

print(json.dumps({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-2/inspect_sources.py", "environment": {"python": platform.python_version(), "curl": subprocess.run(["curl", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]}, "full_documents_saved": False, "sources": sources}, ensure_ascii=False, indent=2))
