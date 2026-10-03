"""1.13 獨立審查：原例、指定練習與有界 CPU 診斷；不訓練模型。"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "docs/technical-reviews/artifacts"
torch.set_num_threads(1)
torch.set_default_device("cpu")


def section(path, lesson):
    raw = (ROOT / path).read_bytes()
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, h in enumerate(headers) if h[0].startswith(f"## {lesson} ".encode()))
    end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
    return raw[headers[index].start():end]


def program(raw):
    return "\n".join(re.findall(r"```python\n(.*?)```", raw.decode(), re.S))


def execute(code):
    stream = io.StringIO()
    namespace = {"__name__": "__main__"}
    with contextlib.redirect_stdout(stream):
        exec(compile(code, "1.13-reviewed-example", "exec"), namespace)
    return namespace, stream.getvalue()


raw = section("course/chapters/01.md", "1.13")
(ART / "1.13-reviewed-section.txt").write_bytes(raw)
code = program(raw)
exercise = code.replace("w.grad = None", "# w.grad = None")
assert exercise != code and exercise.count("# w.grad = None") == 1
original_ns, original_out = execute(code)
exercise_ns, exercise_out = execute(exercise)
original_values = [original_ns[k].item() for k in ("first", "second")] + [original_ns["w"].grad.item(), original_ns["w"].item()]
exercise_values = [exercise_ns[k].item() for k in ("first", "second")] + [exercise_ns["w"].grad.item(), exercise_ns["w"].item()]
assert original_values == [4.0, 8.0, 4.0, 2.0]
assert exercise_values == [4.0, 8.0, 12.0, 2.0]

# clone 的快照與未 clone 的別名。
p = nn.Parameter(torch.tensor(2.0))
(p.square()).backward()
alias = p.grad
snapshot = p.grad.clone()
is_copy = snapshot.data_ptr() != p.grad.data_ptr()
(p.square()).backward()
clone_probe = {"different_storage": is_copy, "snapshot": snapshot.item(), "alias_after_second": alias.item(), "current_grad": p.grad.item(), "requires_grad": p.requires_grad, "is_leaf": p.is_leaf, "parameter_before_step": p.item(), "dtype": str(p.dtype), "shape": str(tuple(p.shape))}
assert clone_probe["snapshot"] == 4.0 and clone_probe["alias_after_second"] == 8.0
p.grad = None
none_before = p.grad is None
(p.square()).backward()
none_probe = {"before_backward_is_none": none_before, "after_backward": p.grad.item()}
assert none_before and p.grad.item() == 4.0
p.grad.zero_()
zero_before = p.grad.item()
(p.square()).backward()
zero_probe = {"before_backward": zero_before, "after_backward": p.grad.item()}
assert zero_before == 0.0 and p.grad.item() == 4.0

# zero_grad 只處理自己管理的參數；backward 與 step 分開。
a = nn.Parameter(torch.tensor(2.0))
b = nn.Parameter(torch.tensor(3.0))
unmanaged = nn.Parameter(torch.tensor(2.0))
opt = torch.optim.SGD([a, b], lr=0.125)
(a.square() + b.square() + unmanaged.square()).backward()
opt.zero_grad()
managed_reset = [a.grad is None, b.grad is None]
unmanaged_grad = unmanaged.grad.item()
(a.square()).backward()
before_step = a.item()
grad_before_step = a.grad.item()
opt.step()
step_probe = {"managed_reset": managed_reset, "unmanaged_gradient": unmanaged_grad, "before_step": before_step, "gradient_before_step": grad_before_step, "after_step": a.item(), "unused_b_after_step": b.item()}
assert managed_reset == [True, True] and unmanaged_grad == 4.0
assert before_step == 2.0 and a.item() == 1.5
opt.zero_grad(set_to_none=False)
zero_optimizer_probe = {"a_grad": a.grad.item(), "b_grad_is_none": b.grad is None}
assert a.grad.item() == 0.0

# 同一個平方 loss 需要的保存值已釋放；保留圖時可重用。
r = nn.Parameter(torch.tensor(2.0))
loss = r.square()
loss.backward()
try:
    loss.backward()
except RuntimeError as error:
    same_loss = {"error_type": type(error).__name__, "message": str(error), "gradient_after_first": 4.0}
else:
    raise AssertionError("Expected saved-tensor reuse RuntimeError")
r = nn.Parameter(torch.tensor(2.0))
loss = r.square()
loss.backward(retain_graph=True)
loss.backward()
retained_probe = {"gradient": r.grad.item()}
assert r.grad.item() == 8.0

# 普通一輪與下一輪都在每次 backward 前清除；scalar SGD 只做兩次教學更新。
t = nn.Parameter(torch.tensor(2.0))
sgd = torch.optim.SGD([t], lr=0.125)
training_sequence = []
for _ in range(2):
    sgd.zero_grad()
    loss = t.square()
    loss.backward()
    before, grad = t.item(), t.grad.item()
    sgd.step()
    training_sequence.append({"before": before, "loss": loss.item(), "gradient": grad, "after": t.item()})
assert [q["gradient"] for q in training_sequence] == [4.0, 3.0]

# 必要前置只親跑其短教學程式，並未重跑 16.6 的正式實驗。
prerequisites = {}
for path, lesson in [("course/chapters/01.md", "1.10"), ("course/chapters/16.md", "16.6")]:
    pre_raw = section(path, lesson)
    _, output = execute(program(pre_raw))
    prerequisites[lesson] = {"source": f"{path}#{lesson}", "source_sha256": hashlib.sha256(pre_raw).hexdigest(), "stdout": output}

result = {
    "reviewed_on": "2026-10-03",
    "command": "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python docs/technical-reviews/artifacts/1.13-cpu.py",
    "environment": {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "torch_threads": str(torch.get_num_threads()), "OMP_NUM_THREADS": str(os.environ.get("OMP_NUM_THREADS")), "MKL_NUM_THREADS": str(os.environ.get("MKL_NUM_THREADS")), "cuda_build": str(torch.version.cuda)},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "original_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "exercise_change": "只將 w.grad = None 改成 # w.grad = None；原 assert 未變",
    "original": {"stdout": original_out, "values": original_values, "assert_passed": True},
    "exercise": {"stdout": exercise_out, "values": exercise_values, "original_assert_passed": True},
    "clone_probe": clone_probe,
    "none_reset_probe": none_probe,
    "zero_reset_probe": zero_probe,
    "optimizer_step_probe": step_probe,
    "optimizer_zero_probe": zero_optimizer_probe,
    "same_loss_probe": same_loss,
    "retained_graph_probe": retained_probe,
    "ordinary_training_sequence": training_sequence,
    "prerequisites": prerequisites,
    "result": "全部有界 CPU 檢查通過；無 GPU、網路、資料下載或正式訓練。"
}
(ART / "1.13-cpu-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
