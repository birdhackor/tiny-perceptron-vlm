"""核對參數凍結、既有梯度及不同 optimizer 的更新條件。"""

import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch
from torch import nn


torch.set_default_device("cpu")
torch.set_num_threads(1)
torch.manual_seed(1019)
root = Path(__file__).resolve().parents[4]
source = root / "docs/course-review-20261019/freeze-05/sources/5.17.md"
raw = source.read_bytes()
results = {}


def values(tensor):
    return None if tensor is None else tensor.detach().tolist()


for index, block in enumerate(re.findall(r"```python\n(.*?)```", raw.decode(), re.S), 1):
    namespace = {}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(block, str(source), "exec"), namespace)
    layer, x = namespace["layer"], namespace["x"]
    if index == 1:
        assert torch.equal(layer.weight.grad, torch.ones(1, 2)), "權重梯度應為兩個 1"
        assert torch.equal(layer.bias.grad, torch.ones(1)), "偏移梯度應為 1"
    else:
        assert torch.equal(x.grad, layer.weight), "輸入梯度應等於固定係數"
        assert all(p.grad is None for p in layer.parameters()), "新建並凍結的參數不應累積梯度"
    results[f"原始程式區塊{index}"] = {
        "程式雜湊": hashlib.sha256(block.encode()).hexdigest(),
        "標準輸出": captured.getvalue(), "獨立執行": True,
        "輸入形狀": list(x.shape), "輸出形狀": list(namespace["output"].shape),
        "權重": values(layer.weight), "輸入梯度": values(x.grad),
        "參數梯度": {name: values(p.grad) for name, p in layer.named_parameters()},
    }

parameter = nn.Parameter(torch.tensor([1.0]))
(parameter * 2).sum().backward()
before_freeze = parameter.grad.clone()
parameter.requires_grad_(False)
after_freeze = parameter.grad.clone()
x = torch.tensor([1.0], requires_grad=True)
(parameter * x * 3).sum().backward()
after_backward = parameter.grad.clone()
assert torch.equal(before_freeze, after_freeze), "凍結不應清除既有梯度"
assert torch.equal(before_freeze, after_backward), "之後 backward 不應對凍結參數累積新梯度"
optimizer = torch.optim.SGD([parameter], lr=0.1, foreach=False)
before_step = parameter.detach().clone()
optimizer.step()
assert not torch.equal(before_step, parameter), "保留的梯度仍可被 SGD 使用"
results["保留梯度與停止新累積"] = {
    "凍結前梯度": values(before_freeze), "凍結後梯度": values(after_freeze),
    "再次backward後梯度": values(after_backward), "新輸入梯度": values(x.grad),
    "SGD更新前": values(before_step), "SGD更新後": values(parameter),
}

for name, cls, kwargs in [
    ("SGD", torch.optim.SGD, {"momentum": 0.9, "weight_decay": 0.2}),
    ("Adam", torch.optim.Adam, {"weight_decay": 0.2}),
    ("AdamW", torch.optim.AdamW, {"weight_decay": 0.2}),
]:
    p = nn.Parameter(torch.tensor([1.0]))
    optimizer = cls([p], lr=0.1, foreach=False, **kwargs)
    p.square().sum().backward()
    optimizer.step()
    p.requires_grad_(False)
    p.grad = None
    before = p.detach().clone()
    optimizer.step()
    assert torch.equal(before, p), "此 optimizer 應略過梯度為 None 的參數"
    results[name + "清成None"] = {
        "配置": {"lr": 0.1, "foreach": False, **kwargs}, "已先更新步數": 1,
        "requires_grad": p.requires_grad, "grad": values(p.grad),
        "清成None後step前": values(before), "清成None後step後": values(p),
    }

p = nn.Parameter(torch.tensor([1.0]))
q = nn.Parameter(torch.tensor([2.0]))
optimizer = torch.optim.LBFGS([p, q], lr=0.1, max_iter=1)


def closure():
    optimizer.zero_grad(set_to_none=True)
    loss = (p + q).square().sum() + q.square().sum()
    loss.backward()
    return loss


history = []
for _ in range(2):
    optimizer.step(closure)
    history.append({"p": values(p), "q": values(q)})
p.requires_grad_(False)
p.grad = None
before = p.detach().clone()
optimizer.step(closure)
after = p.detach().clone()
assert p.grad is None, "凍結參數的梯度應一直為 None"
assert not torch.equal(before, after), "LBFGS 的歷史方向仍會更新此參數"
results["LBFGS清成None"] = {
    "配置": {"lr": 0.1, "max_iter": 1, "line_search_fn": None},
    "代價": "(p+q)^2+q^2", "初值": {"p": 1.0, "q": 2.0},
    "凍結前兩步": history, "requires_grad": p.requires_grad,
    "grad": values(p.grad), "凍結且清成None後step前": values(before),
    "凍結且清成None後step後": values(after),
}

excluded_results = {}
for name in ("SGD", "LBFGS"):
    p = nn.Parameter(torch.tensor([1.0]))
    q = nn.Parameter(torch.tensor([2.0]))
    (p * 2).sum().backward()
    p.requires_grad_(False)
    if name == "SGD":
        optimizer = torch.optim.SGD([q], lr=0.1)
        q.square().sum().backward()
        optimizer.step()
    else:
        optimizer = torch.optim.LBFGS([q], lr=0.1, max_iter=1)

        def closure_excluded():
            optimizer.zero_grad(set_to_none=True)
            loss = (p + q).square().sum() + q.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure_excluded)
    assert p.item() == 1.0, "沒有交給 optimizer 的參數不應被它更新"
    assert q.item() != 2.0, "交給 optimizer 的參數應真的更新"
    excluded_results[name] = {"p": values(p), "p.grad": values(p.grad), "q": values(q)}
results["參數不交給optimizer"] = excluded_results

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-17/verify-freezing-current-01.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
        "torch_git_version": torch.version.git_version, "platform": platform.platform(),
        "device": "cpu", "dtype": str(torch.get_default_dtype()), "seed": "1019"},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "結果": results,
    "結論": "所有斷言通過；新梯度累積與舊梯度說明成立。None 可讓 SGD、Adam、AdamW 略過參數，但不能保證 LBFGS 不更新仍在參數清單中的凍結參數。",
}, ensure_ascii=False, indent=2))
