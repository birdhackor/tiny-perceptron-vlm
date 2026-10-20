"""核對 Linear、梯度模式、凍結參數與 optimizer 的實際行為。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn


torch.set_default_device("cpu")
torch.set_num_threads(1)
torch.manual_seed(1019)
root = Path(__file__).resolve().parents[4]
snapshot = root / "docs/course-review-20261019/freeze-01/sources/5.17.md"
raw = snapshot.read_bytes()
results = {}


def values(tensor):
    return None if tensor is None else tensor.detach().tolist()


fences = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
for index, code in enumerate(fences, 1):
    namespace = {}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(code, str(snapshot), "exec"), namespace)
    layer = namespace["layer"]
    output = namespace["output"]
    item = {
        "獨立執行": True,
        "程式雜湊": hashlib.sha256(code.encode()).hexdigest(),
        "標準輸出": captured.getvalue(),
        "輸入": values(namespace["x"]),
        "輸入形狀": list(namespace["x"].shape),
        "輸出形狀": list(output.shape),
        "dtype": str(output.dtype),
        "輸出可求導": output.requires_grad,
        "grad_fn": type(output.grad_fn).__name__,
        "參數可求導": {name: p.requires_grad for name, p in layer.named_parameters()},
        "參數梯度": {name: values(p.grad) for name, p in layer.named_parameters()},
        "權重": values(layer.weight),
        "偏移": values(layer.bias),
        "輸入梯度": values(namespace["x"].grad),
    }
    if index == 1:
        assert torch.equal(layer.weight.grad, torch.ones(1, 2)), "權重梯度應為兩個 1"
        assert torch.equal(layer.bias.grad, torch.ones(1)), "偏移梯度應為 1"
        assert namespace["other"].grad_fn is None, "no_grad 結果應沒有求導關係"
        item["no_grad輸出可求導"] = namespace["other"].requires_grad
        item["no_grad輸出grad_fn"] = namespace["other"].grad_fn
    else:
        assert all(p.grad is None for p in layer.parameters()), "凍結的新參數應沒有累積梯度"
        assert torch.equal(namespace["x"].grad, layer.weight), "輸入梯度應精確等於同一次計算的權重"
    results[f"原始程式區塊{index}"] = item

layer = nn.Linear(2, 1)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[0.25, -0.75]]))
    layer.bias.fill_(0.5)
x = torch.ones(1, 2)
training_output = layer(x)
layer.eval()
eval_output = layer(x)
assert torch.equal(training_output, eval_output), "Linear 在 train 與 eval 的結果應相同"
assert torch.equal(eval_output, torch.tensor([[0.0]])), "仿射計算結果應為零"
results["Linear仿射與模式"] = {
    "輸入": values(x), "權重": values(layer.weight), "偏移": values(layer.bias),
    "train輸出": values(training_output), "eval輸出": values(eval_output),
    "子模組": [type(m).__name__ for m in layer.modules()],
}

dropout = nn.Dropout(0.5)
x = torch.ones(64, requires_grad=True)
training_output = dropout(x)
dropout.eval()
eval_output = dropout(x)
eval_output.sum().backward()
dropout.train()
with torch.no_grad():
    no_grad_output = dropout(x)
assert torch.equal(eval_output, x), "eval 的 Dropout 應為恆等映射"
assert eval_output.requires_grad, "eval 應保留輸入的可求導性"
assert torch.equal(x.grad, torch.ones_like(x)), "恆等映射的輸入梯度應為 1"
assert not no_grad_output.requires_grad and no_grad_output.grad_fn is None, "train 配合 no_grad 應不記錄運算"
results["Dropout與梯度模式"] = {
    "seed": 1019, "元素數": x.numel(), "省略機率": dropout.p,
    "train輸出不同值": values(training_output.unique()),
    "eval等於輸入": torch.equal(eval_output, x),
    "eval輸出可求導": eval_output.requires_grad,
    "eval輸入梯度不同值": values(x.grad.unique()),
    "no_grad時training": dropout.training,
    "no_grad輸出可求導": no_grad_output.requires_grad,
    "no_grad輸出grad_fn": no_grad_output.grad_fn,
}

front = nn.Linear(2, 2, bias=False)
reference = nn.Linear(2, 1)
head = nn.Linear(1, 1, bias=False)
with torch.no_grad():
    front.weight.copy_(torch.eye(2))
    reference.weight.copy_(torch.tensor([[0.25, -0.75]]))
    reference.bias.fill_(0.5)
    head.weight.fill_(1.0)
reference.requires_grad_(False)
reference.eval()
data = torch.tensor([[2.0, 1.0]])
head(reference(front(data))).sum().backward()
gradient = front.weight.grad.clone()
assert torch.equal(gradient, torch.tensor([[0.5, 0.25], [-1.5, -0.75]])), "梯度應通過固定參考層傳到前層"
assert all(p.grad is None for p in reference.parameters()), "固定參考層不應累積參數梯度"
front.zero_grad(set_to_none=True)
head.zero_grad(set_to_none=True)
with torch.no_grad():
    detached = reference(front(data))
head(detached).sum().backward()
assert front.weight.grad is None, "no_grad 應切斷通往前層的路徑"
assert head.weight.grad is not None, "後層仍可獨立累積梯度"
results["固定參考層與前層訓練"] = {
    "資料": values(data), "參考層權重": values(reference.weight),
    "正常模式前層梯度": values(gradient),
    "no_grad後前層梯度": values(front.weight.grad),
    "no_grad後後層梯度": values(head.weight.grad),
    "參考層參數梯度": {name: values(p.grad) for name, p in reference.named_parameters()},
}

parameter = nn.Parameter(torch.tensor([1.0]))
optimizer = torch.optim.SGD([parameter], lr=0.1, foreach=False)
(parameter * 2).sum().backward()
parameter.requires_grad_(False)
before = parameter.detach().clone()
retained_gradient = parameter.grad.clone()
optimizer.step()
after = parameter.detach().clone()
assert not torch.equal(before, after), "保留的梯度仍會讓 SGD 更新參數"
optimizer.zero_grad(set_to_none=True)
cleared_before = parameter.detach().clone()
optimizer.step()
assert torch.equal(parameter, cleared_before), "梯度為 None 時 SGD 應略過參數"
results["凍結後保留舊梯度"] = {
    "requires_grad": parameter.requires_grad,
    "學習率": 0.1, "原先梯度": values(retained_gradient),
    "step前": values(before), "step後": values(after),
    "清成None後梯度": values(parameter.grad),
    "清成None後再step": values(parameter),
}

torch.manual_seed(1019)
first = nn.Linear(2, 1).weight.detach().clone()
torch.manual_seed(1020)
second = nn.Linear(2, 1).weight.detach().clone()
assert not torch.equal(first, second), "兩個指定 seed 應產生不同的初始化係數"
results["初始化"] = {"seed1019權重": values(first), "seed1020權重": values(second)}

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-17/verify-grad-controls.py",
    "執行時間UTC": datetime.now(timezone.utc).isoformat(),
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
        "torch_git_version": torch.version.git_version, "platform": platform.platform(),
        "device": "cpu", "dtype": str(torch.get_default_dtype()), "seed": "1019；初始化比較另用 1020"},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "結果": results,
    "結論": "所有斷言通過；requires_grad=False 本身不能保證 optimizer 不更新保留舊梯度的參數。",
}, ensure_ascii=False, indent=2))
