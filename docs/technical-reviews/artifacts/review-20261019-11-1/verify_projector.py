"""直接核對接頭算例、清零位置與語意反例。"""

import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch
from torch import nn


def execute(code):
    namespace = {"__name__": "__main__"}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(code, "projector_example", "exec"), namespace)
    return namespace, captured.getvalue()


root = Path(__file__).resolve().parents[4]
snapshot = root / "docs/course-review-20261019/freeze-01/sources/11.1.md"
raw = snapshot.read_bytes()
baseline, exercise = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
torch.set_default_device("cpu")
torch.set_num_threads(1)
original, baseline_stdout = execute(baseline)
projector = original["projector"]
a, b = original["a"], original["b"]
original_difference = original["difference"]
assert tuple(projector(a).shape) == (1, 8), "接頭輸出形狀不符"
assert original_difference.item() > 0, "指定隨機種子的差向量應為非零"
torch.testing.assert_close(projector(a), a @ projector.weight.T + projector.bias)
torch.testing.assert_close(projector(b), b @ projector.weight.T + projector.bias)
torch.testing.assert_close(projector(a) - projector(b), (a - b) @ projector.weight.T)

marker = "difference = (projector(a) - projector(b)).norm()"
assert baseline.count(marker) == 1, "無法唯一定位差向量計算"
inserted = baseline.replace(marker, exercise + "\n" + marker)
zeroed, inserted_stdout = execute(inserted)
assert zeroed["difference"].item() == 0, "清零後差向量應為零"
assert torch.equal(zeroed["projector"](a), zeroed["projector"].bias.unsqueeze(0)), "輸出應等於偏移"
assert torch.equal(zeroed["projector"](a), zeroed["projector"](b)), "兩個輸出應相同"

before = original_difference.detach().clone()
bias_before = projector.bias.detach().clone()
with torch.no_grad():
    grad_enabled_during_zero = torch.is_grad_enabled()
    projector.weight.zero_()
assert not grad_enabled_during_zero, "清零時不應啟用梯度記錄"
assert projector.weight.requires_grad, "no_grad 不應凍結參數"
assert projector.weight.grad is None, "範例不應累積梯度"
assert torch.equal(projector.bias, bias_before), "偏移應保持相同"
assert torch.count_nonzero(projector.weight).item() == 0, "權重應全部清零"
assert torch.equal(original_difference.detach(), before), "既有結果不應重算"
recomputed = (projector(a) - projector(b)).norm()
assert recomputed.item() == 0, "重算的差向量應為零"

single = nn.Linear(4, 1)
with torch.no_grad():
    single.weight.copy_(torch.tensor([[1.0, 2.0, 0.0, 0.0]]))
    single.bias.fill_(0.5)
single_zero, single_one = single(a).item(), single(b).item()
assert single_zero == 0.5, "全零輸入的算例不符"
assert single_one == 3.5, "全一輸入的算例不符"

counterexample = {"紅圓": "藍", "藍圓": "紅"}
assert counterexample["紅圓"] != counterexample["藍圓"], "反例必須對輸入敏感"
assert counterexample["紅圓"] != "紅" and counterexample["藍圓"] != "藍", "反例必須答錯顏色"

result = {
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": str(a.device),
        "dtype": str(a.dtype),
        "threads": str(torch.get_num_threads()),
    },
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "baseline": {
        "stdout": baseline_stdout,
        "input_shape": list(a.shape),
        "output_shape": [1, 8],
        "difference_norm": before.item(),
        "affine_formula_verified": True,
    },
    "insert_before_difference": {
        "stdout": inserted_stdout,
        "difference_norm": zeroed["difference"].item(),
        "bias_only": True,
        "output_shape": list(zeroed["projector"](a).shape),
    },
    "zero_after_difference": {
        "stored_difference_norm": original_difference.item(),
        "recomputed_difference_norm": recomputed.item(),
        "stored_result_unchanged": True,
    },
    "no_grad_mutation": {
        "grad_enabled_during_mutation": grad_enabled_during_zero,
        "weight_requires_grad_after_mutation": projector.weight.requires_grad,
        "weight_grad_is_none": projector.weight.grad is None,
        "bias_unchanged": True,
        "nonzero_weights_after_mutation": torch.count_nonzero(projector.weight).item(),
        "backward_calls": 0,
        "optimizer_step_calls": 0,
    },
    "scalar_example": {"zero_input": single_zero, "one_input": single_one},
    "semantic_counterexample": {
        "answers": counterexample,
        "different_answers": True,
        "correct_answers": 0,
        "cases": 2,
        "scope": "人工構造的邏輯反例，並非實測模型品質。",
    },
    "result": "所有斷言通過",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
