"""核對凍結參數在指定 optimizer 下的略過與排除行為。"""

import ast
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
source = root / "docs/course-review-20261019/freeze-06/sources/5.17.md"
raw = source.read_bytes()
results = {}


def values(tensor):
    return None if tensor is None else tensor.detach().tolist()


for index, block in enumerate(re.findall(r"```python\n(.*?)```", raw.decode(), re.S), 1):
    namespace = {}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(block, str(source), "exec"), namespace)
    layer = namespace["layer"]
    if index == 1:
        assert torch.equal(layer.weight.grad, torch.ones(1, 2)), "權重梯度應為兩個 1"
        assert torch.equal(layer.bias.grad, torch.ones(1)), "偏移梯度應為 1"
    else:
        assert torch.equal(namespace["x"].grad, layer.weight), "輸入梯度應等於固定係數"
        assert all(p.grad is None for p in layer.parameters()), "凍結參數不應累積新梯度"
    results[f"原始程式區塊{index}"] = {
        "程式雜湊": hashlib.sha256(block.encode()).hexdigest(),
        "標準輸出": captured.getvalue(), "獨立執行": True,
        "輸入形狀": list(namespace["x"].shape),
        "輸出形狀": list(namespace["output"].shape),
        "權重": values(layer.weight), "輸入梯度": values(namespace["x"].grad),
        "參數梯度": {name: values(p.grad) for name, p in layer.named_parameters()},
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
    state_before = {k: v.detach().clone() for k, v in optimizer.state[p].items() if torch.is_tensor(v)}
    optimizer.step()
    assert torch.equal(before, p), "梯度為 None 的參數應被此 optimizer 略過"
    for key, value in state_before.items():
        assert torch.equal(value, optimizer.state[p][key]), "略過的參數不應更新其 optimizer tensor 狀態"
    results[name + "清成None"] = {
        "配置": {"lr": 0.1, "foreach": False, **kwargs},
        "已先更新步數": 1, "requires_grad": p.requires_grad, "grad": values(p.grad),
        "step前": values(before), "step後": values(p),
        "略過時tensor狀態不變": True,
    }

p = nn.Parameter(torch.tensor([1.0]))
q = nn.Parameter(torch.tensor([2.0]))
(p * 2).sum().backward()
p.requires_grad_(False)
optimizer = torch.optim.LBFGS([q], lr=0.1, max_iter=1)


def closure():
    optimizer.zero_grad(set_to_none=True)
    loss = (p + q).square().sum() + q.square().sum()
    loss.backward()
    return loss


optimizer.step(closure)
assert p.item() == 1.0, "未交給 optimizer 的參數不應被它更新"
assert q.item() != 2.0, "optimizer 應真的更新自己持有的參數"
results["排除凍結參數"] = {
    "optimizer": "LBFGS", "配置": {"lr": 0.1, "max_iter": 1},
    "只交給optimizer的參數": "q", "p": values(p), "p.grad": values(p.grad), "q": values(q),
}

train_path = root / "scripts/train.py"
train_raw = train_path.read_bytes()
constructors = [node for node in ast.walk(ast.parse(train_raw))
                if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.optim.AdamW"]
assert constructors, "訓練入口應使用 AdamW"
results["訓練入口optimizer"] = {
    "path": train_path.relative_to(root).as_posix(),
    "sha256": hashlib.sha256(train_raw).hexdigest(),
    "建構式": [ast.get_source_segment(train_raw.decode(), node) for node in constructors],
    "查核範圍": "讀取並解析實際建構式；沒有啟動訓練。",
}

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-17/verify-freezing-current-02.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
        "torch_git_version": torch.version.git_version, "platform": platform.platform(),
        "device": "cpu", "dtype": str(torch.get_default_dtype()), "seed": "1019"},
    "source_sha256": hashlib.sha256(raw).hexdigest(), "結果": results,
    "結論": "所有斷言通過；AdamW、SGD、Adam 略過 None 梯度參數；從 LBFGS 的參數清單排除凍結參數可避免它被更新。",
}, ensure_ascii=False, indent=2))
