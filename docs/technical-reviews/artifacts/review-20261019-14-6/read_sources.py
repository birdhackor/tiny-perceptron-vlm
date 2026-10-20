"""以 curl 讀取原始來源，只保存短摘錄、雜湊與版本收據。"""

import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COMMIT = "5c4886908584029761b579af026dcfb627c84070"


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def read(url):
    command = ["curl", "-fLsS", "--max-time", "30", url]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return raw, command


specifications = [
    ("paper", "https://arxiv.org/html/1608.05859v3", "arXiv:1608.05859v3, 2017-02-21", "In tied NNLMs, we set", "Weight Tying；核對 U=V=S 與兩種角色梯度相加；Introduction 的 Vh 與其後 softmax。"),
    ("linear", "torch/nn/modules/linear.py", COMMIT, "Applies an affine linear transformation to the incoming data", "Linear 的 y=xA^T+b、權重形狀與 bias=False 分支；forward 呼叫 F.linear。"),
    ("parameter", "torch/nn/parameter.py", COMMIT, "A kind of Tensor that is to be considered a module parameter.", "Parameter 的 Tensor 子類別、Module 自動註冊與 requires_grad 預設值。"),
    ("embedding", "torch/nn/modules/sparse.py", COMMIT, "A simple lookup table that stores embeddings of a fixed dictionary and size.", "Embedding 的索引查表、weight 形狀與輸入輸出形狀。"),
    ("no_grad", "torch/autograd/grad_mode.py", COMMIT, "Context-manager that disables gradient calculation.", "no_grad 停用 reverse-mode 梯度記錄；factory 與 forward-mode 例外不影響本節直接改值用途。"),
    ("module", "torch/nn/modules/module.py", COMMIT, "remove_duplicate: bool = True", "parameters 使用 named_parameters；後者預設去除同一 Parameter 的重複引用。"),
]
receipts = []
for identifier, location, version, excerpt, note in specifications:
    url = location if location.startswith("https://") else f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/{location}"
    raw, command = read(url)
    text = raw.decode()
    if identifier == "paper":
        parser = PlainText()
        parser.feed(text)
        text = " ".join("".join(parser.parts).split())
    assert excerpt in text
    receipts.append({
        "id": identifier, "url": url, "version": version,
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "command": command, "response_sha256": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw), "excerpt": excerpt, "inspection_note": note,
    })

path = "scripts/course_experiments/architecture.py"
blob_id = "183d41508632aa2c18a53a84c917a110f1349baa"
command = ["git", "cat-file", "blob", blob_id]
historical = subprocess.run(command, cwd=ROOT, check=True, capture_output=True).stdout
expected = "1ad0b0789318236d5e1a8fc1117cb65758bbd6e3ad62dc2d646a111ae1ea37a3"
assert hashlib.sha256(historical).hexdigest() == expected
current = (ROOT / path).read_bytes()
old_tree, new_tree = ast.parse(historical), ast.parse(current)
checked_names = ("_copy_matching", "_clone_config", "_nll", "_train", "run_modern")
comparisons = {}
for name in checked_names:
    old = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    new = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    comparisons[name] = ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    assert comparisons[name]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-6/read_sources.py",
    "sources": receipts,
    "historical_implementation": {
        "path": path, "blob_id": blob_id, "sha256": expected,
        "command": command, "matches_recorded_code_sha256": True,
        "functions_ast_equal_to_current": comparisons,
        "inspection_note": "實驗紀錄所指 commit 不在本機，但找到 SHA-256 完全相符的 Git blob；直接讀取並比對與本節相關的函式，語法樹皆相同。",
    },
    "storage_policy": "第三方全文只在記憶體讀取，未寫入檔案；保存的摘錄均為支持主張的小段原文。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
