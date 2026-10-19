"""核對凍結小節的查表、初始化、梯度記錄與局部改值行為。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[4]
checks = []


def check(name, expected, observed):
    if expected != observed:
        raise AssertionError(f"{name}：預期 {expected!r}，實際 {observed!r}")
    checks.append({"name": name, "expected": expected, "observed": observed})


frozen_path = ROOT / "docs/course-review-20261019/freeze-01/sources/1.6.md"
frozen = frozen_path.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", frozen.decode(), re.S)
check("凍結小節的 Python 區塊數", 1, len(blocks))
namespace = {}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(blocks[0], str(frozen_path), "exec"), namespace)
scores, inputs, output = (namespace[name] for name in ["scores", "inputs", "output"])
check("神經網路工具箱", "torch.nn", namespace["nn"].__name__)
check("初始整張分數表", [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], scores.weight.tolist())
check("查出貓與狗兩列", [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]], output.tolist())
check("查表輸出形狀", [2, 3], list(output.shape))
check("候選最高分的位置", 1, int(output[0].argmax()))
check("貓與狗候選得分相等", True, bool(output[0, 0] == output[0, 2]))
check("輸入索引型別", "torch.int64", str(inputs.dtype))
check("預設分數型別", "torch.float32", str(output.dtype))
check("分數表參數數量", 9, scores.weight.numel())
check("參數型別", True, isinstance(scores.weight, nn.Parameter))
check("參數可求梯度", True, scores.weight.requires_grad)
check("參數登記名稱", ["weight"], list(dict(scores.named_parameters())))
check("手動初始化後仍是葉節點", True, scores.weight.is_leaf)
check("初始化參數沒有運算來源", True, scores.weight.grad_fn is None)
check("尚未反向傳播時沒有梯度", True, scores.weight.grad is None)
check("查表留下梯度記錄", "EmbeddingBackward0", type(output.grad_fn).__name__)
check("參數的顯示標記", True, "Parameter containing" in captured.getvalue())
check("參數的梯度顯示標記", True, "requires_grad=True" in captured.getvalue())
before_lookup = scores.weight.detach().clone()
scores(inputs)
check("預設查表不改表", True, torch.equal(before_lookup, scores.weight))
with torch.no_grad():
    check("填表區內關閉梯度記錄", False, torch.is_grad_enabled())
    unrecorded = scores(inputs)
    check("區內查表沒有運算記錄", True, unrecorded.grad_fn is None)
check("離開填表區恢復梯度記錄", True, torch.is_grad_enabled())
check("不記錄模式保留參數可求梯度屬性", True, scores.weight.requires_grad)
temporary = torch.tensor([4.0, -3.0])
identity = id(temporary)
returned = temporary.zero_()
check("清零直接修改原物件", True, returned is temporary and id(temporary) == identity)
check("清零的數值", [0.0, 0.0], temporary.tolist())
with torch.no_grad():
    scores.weight[2, 0] = 3.0
edited = scores(inputs)
check("改狗列後貓題保持原值", [0.0, 2.0, 0.0], edited[0].tolist())
check("改狗列後狗題的新值", [3.0, 0.0, 0.0], edited[1].tolist())
check("只有指定列改變", [False, False, True], (before_lookup != scores.weight).any(dim=1).tolist())
with torch.no_grad():
    scores.weight[1] = torch.tensor([-2.0, 0.0, 5.0])
check("分數可為負且不須歸一化", [-2.0, 0.0, 5.0], scores(torch.tensor([1]))[0].tolist())
text = "貓看狗。狗吃魚。貓看鳥。"
vocabulary = sorted(set(text))
check("前一接字表示例的字表大小", 7, len(vocabulary))
check("七種字的完整分數表格數", 49, nn.Embedding(len(vocabulary), len(vocabulary)).weight.numel())
extended_vocabulary = ["貓", "看", "狗", "="]
table = nn.Embedding(len(extended_vocabulary), len(extended_vocabulary))
with torch.no_grad():
    table.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))
contexts = ["顏色=", "形狀="]
last_ids = [extended_vocabulary.index(context[-1]) for context in contexts]
context_outputs = table(torch.tensor(last_ids))
check("以末字選列時兩段前文給相同 ID", [3, 3], last_ids)
check("以末字選列時無法分辨兩段前文", True, torch.equal(context_outputs[0], context_outputs[1]))
parameters_before_backward = table.weight.detach().clone()
context_outputs.sum().backward()
check("求出梯度並不自動更新參數", True, torch.equal(parameters_before_backward, table.weight))
receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-6/verification.py",
    "environment": {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "platform": platform.platform(),
        "device": str(output.device),
        "dtype": str(output.dtype),
    },
    "source_sha256": hashlib.sha256(frozen).hexdigest(),
    "checks": checks,
    "frozen_code_stdout": captured.getvalue(),
    "limitations": [
        "等號示意另建含等號的字表；凍結範例的三字字表不含等號。",
        "只核對預設 max_norm=None 的查表；設定 max_norm 的前向計算可能原地改權重。",
        "未執行最佳化器更新或訓練；梯度方向只表達指定目標的局部一階變化。",
        "浮點數只使用可精確表示的小整數；本次不驗證其他裝置與版本。",
    ],
}
print(json.dumps(receipt, ensure_ascii=False, indent=2))
