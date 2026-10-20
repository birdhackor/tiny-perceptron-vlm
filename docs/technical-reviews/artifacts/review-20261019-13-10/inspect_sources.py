"""以 curl 讀原始來源，只保存支撐主張的短摘錄與下載收據。"""

import ast
import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime

REVISION = "5c4886908584029761b579af026dcfb627c84070"


def fetch(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    return raw.decode(), {
        "url": url,
        "command": command,
        "exit_code": completed.returncode,
        "bytes": len(raw),
        "download_sha256": hashlib.sha256(raw).hexdigest(),
        "accessed_on": datetime.now(UTC).date().isoformat(),
    }


def text(raw):
    raw = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", raw, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))


receipts = []
dpo, receipt = fetch("https://arxiv.org/html/2305.18290v3")
t = text(dpo)
assert "arXiv:2305.18290v3" in t
assert "Reward Modelling Phase" in t and "negative log-likelihood loss" in t
assert "avoids fitting an explicit, standalone reward model" in t
maths = [html.unescape(value) for value in re.findall(r'<math\b[^>]*\balttext="([^"]+)"', dpo)]
chosen_maths = [value for value in maths if (r"\mathcal{L}_{R}" in value or r"\mathcal{L}_{\text{DPO}}" in value or r"p^{*}(y_{1}" in value) and "=" in value]
receipt.update(
    version="arXiv:2305.18290v3, 2024-07-29",
    locators=["Preliminaries: Reward Modelling Phase, equations (1) and (2)", "Direct Preference Optimization, equation (7)"],
    short_excerpt="avoids fitting an explicit, standalone reward model",
    equations=chosen_maths[:2] + [value for value in chosen_maths if r"\mathcal{L}_{\text{DPO}}" in value][:1],
    inspection_note="偏好分布是同題兩回答的比較；reward 差經 logistic 得勝出機率，以負 log likelihood 擬合。DPO 將 reward 改以 policy/reference 的 log 機率比參數化，直接更新 policy；不是答案事實正確率。",
)
assert len(chosen_maths) >= 2
receipts.append(receipt)

instruct, receipt = fetch("https://arxiv.org/html/2203.02155v1")
t = text(instruct)
start = t.rfind("Step 2: Collect comparison data")
end = t.find("3.2 Dataset", start)
method = t[start:end]
assert "human-preferred" in method
assert "scalar reward" in method and "PPO" in method
receipt.update(
    version="arXiv:2203.02155v1, 2022-03-04",
    locators=["High-level methodology: Step 2 and Step 3"],
    short_excerpt="We use the output of the RM as a scalar reward.",
    inspection_note="先收集對同一輸入的回答比較並訓練 reward model，之後以該模型的 scalar reward 更新另一個 policy。此流程使用人的比較；本專案規則標籤沒有提供同等的人類回饋證據。",
)
receipts.append(receipt)

module, receipt = fetch(f"https://raw.githubusercontent.com/pytorch/pytorch/{REVISION}/torch/nn/modules/module.py")
tree = ast.parse(module)
module_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Module")
methods = {node.name: node for node in module_node.body if isinstance(node, ast.FunctionDef)}
buffer_doc = ast.get_docstring(methods["register_buffer"])
assert "not be considered a model parameter" in re.sub(r"\s+", " ", buffer_doc)
assert "state_dict" in buffer_doc
lines = module.splitlines()
call_node = methods["_call_impl"]
receipt.update(
    version="PyTorch 2.14.1; git " + REVISION,
    locators=["Module.register_buffer", "Module._call_impl", "Module.__call__"],
    short_excerpt=buffer_doc[:buffer_doc.find("Args:")].strip(),
    call_excerpt="\n".join(lines[call_node.lineno - 1:call_node.lineno + 12]),
    inspection_note="register_buffer 預設 persistent=True，buffer 不登錄成 parameter，會進 state_dict；Module 呼叫鏈會執行 forward。摘錄不含完整第三方檔案。",
)
receipts.append(receipt)

tensor_docs, receipt = fetch(f"https://raw.githubusercontent.com/pytorch/pytorch/{REVISION}/torch/_tensor_docs.py")
snippets = {}
for name in ("expand", "numel", "squeeze"):
    match = re.search(r'add_docstr_all\(\s*"' + name + r'",\s*r?"""(.*?)"""', tensor_docs, re.S)
    assert match is not None
    body = match.group(1)
    snippets[name] = body[:min(1800, body.find("Example") if "Example" in body else len(body))].strip()
receipt.update(
    version="PyTorch 2.14.1; git " + REVISION,
    locators=["Tensor.expand", "Tensor.numel", "Tensor.squeeze"],
    snippets=snippets,
    inspection_note="expand 對 singleton 維度建立 view，-1 保留原維度；沒有額外複製儲存空間。numel 計算元素總數，squeeze 指定維度長度為 1 才去除。本節只談展開後邏輯形狀，未宣稱複製記憶體。",
)
receipts.append(receipt)

activation, receipt = fetch(f"https://raw.githubusercontent.com/pytorch/pytorch/{REVISION}/torch/nn/modules/activation.py")
tree = ast.parse(activation)
tanh_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Tanh")
tanh_doc = ast.get_docstring(tanh_node)
assert "element-wise" in tanh_doc and "exp" in tanh_doc
tanh_forward = next(node for node in tanh_node.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
receipt.update(
    version="PyTorch 2.14.1; git " + REVISION,
    locators=["Tanh: mathematical definition", "Tanh.forward"],
    short_excerpt=tanh_doc[:tanh_doc.find("Shape:")].strip(),
    forward_excerpt=ast.get_source_segment(activation, tanh_forward),
    inspection_note="Tanh 逐元素呼叫 torch.tanh；公式為 (exp(x)-exp(-x))/(exp(x)+exp(-x))，實數輸出介於 -1 和 1，有限精度可能飽和到端點。",
)
receipts.append(receipt)

python_docs, receipt = fetch("https://docs.python.org/3.13/reference/compound_stmts.html")
t = text(python_docs)
pos = t.find("A string literal appearing as the first statement in the function body")
assert pos >= 0
receipt.update(
    version="Python 3.13.16 documentation",
    locators=["Function definitions", "Footnotes: function and class docstrings"],
    short_excerpt=t[pos:pos + 310],
    inspection_note="函式／類別最前面的字串成為 __doc__。本節函式未讀取 __doc__，所以修改它不改 loss 計算；不外推成所有三引號字串都沒有運算作用。",
)
receipts.append(receipt)

curl_version = subprocess.run(["curl", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-10/inspect_sources.py", "environment": {"retrieval": curl_version, "parser": "Python standard library; ast, re, html"}, "sources": receipts}
from pathlib import Path

target = Path(__file__).with_name("sources.json")
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
