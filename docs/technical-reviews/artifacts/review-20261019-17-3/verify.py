"""以精確分數及 CPU tensor 核對權重還原與輸出誤差。"""

import contextlib
import hashlib
import io
import json
import platform
import re
from fractions import Fraction
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
source = ROOT / "docs/course-review-20261019/freeze-01/sources/17.3.md"
raw = source.read_bytes()
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
namespace = {"__name__": "__main__"}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, str(source), "exec"), namespace)

w = namespace["w"]
restored = namespace["restored"]
x = namespace["x"]
error = w - restored
delta = namespace["original_output"] - namespace["compressed_output"]
direct_delta = x @ error.T
scaled_x = x * 10
scaled_delta = scaled_x @ w.T - scaled_x @ restored.T

torch.testing.assert_close(restored, torch.tensor([[0.5, 0.0], [-0.5, 1.0]]), rtol=0, atol=0)
torch.testing.assert_close(error, torch.tensor([[0.2, 0.2], [-0.2, 0.2]]), rtol=0, atol=1e-7)
torch.testing.assert_close(delta, torch.tensor([[0.4, 0.0], [0.0, -0.4]]), rtol=0, atol=1e-7)
torch.testing.assert_close(delta, direct_delta, rtol=0, atol=1e-7)
torch.testing.assert_close(scaled_delta, torch.tensor([[4.0, 0.0], [0.0, -4.0]]), rtol=0, atol=1e-6)
assert restored.dtype == torch.float32, "還原 tensor 應維持浮點型別"
assert round(error.abs().mean().item(), 4) == 0.2, "平均絕對誤差應符合手算"
assert round((w - restored).abs().mean().item(), 4) == 0.2, "改變輸入不應改變權重誤差"

q = Fraction
exact_w = [[q("0.7"), q("0.2")], [q("-0.7"), q("1.2")]]
exact_restored = [[q("0.5"), q(0)], [q("-0.5"), q(1)]]
exact_error = [[a - b for a, b in zip(row, restored_row)] for row, restored_row in zip(exact_w, exact_restored)]
exact_x = [[1, 1], [1, -1]]
exact_delta = [[sum(a * b for a, b in zip(row, err)) for err in exact_error] for row in exact_x]
exact_mae = sum(abs(v) for row in exact_error for v in row) / 4
assert exact_mae == q("0.2"), "精確分數的平均絕對誤差應為五分之一"
assert exact_delta == [[q("0.4"), q(0)], [q(0), q("-0.4")]], "精確分數的點積應符合手算"

# 加上固定的 ReLU 與讀出，展示權重誤差本身無法決定最後答案是否改變。
def readout(output):
    return torch.relu(output)[:, 0] - 0.7


original_scores = readout(namespace["original_output"])
restored_scores = readout(namespace["compressed_output"])
assert (original_scores > 0).tolist() == [True, False], "原權重的二元判斷應符合直接計算"
assert (restored_scores > 0).tolist() == [False, False], "還原權重的二元判斷應符合直接計算"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-3/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": str(w.device), "default_dtype": str(torch.get_default_dtype())},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "original_fence_stdout": stdout.getvalue(),
    "w": w.tolist(),
    "restored": restored.tolist(),
    "restored_dtype": str(restored.dtype),
    "weight_error": error.tolist(),
    "weight_mae": error.abs().mean().item(),
    "original_output": namespace["original_output"].tolist(),
    "compressed_output": namespace["compressed_output"].tolist(),
    "output_delta": delta.tolist(),
    "direct_error_product": direct_delta.tolist(),
    "maximum_identity_roundoff": (delta - direct_delta).abs().max().item(),
    "scaled_input": scaled_x.tolist(),
    "scaled_output_delta": scaled_delta.tolist(),
    "scaled_printed_delta": scaled_delta.round(decimals=4).tolist(),
    "exact_error": [[str(v) for v in row] for row in exact_error],
    "exact_mae": str(exact_mae),
    "exact_output_delta": [[str(v) for v in row] for row in exact_delta],
    "readout_definition": "relu(output)[:, 0] - 0.7 > 0",
    "original_readout_scores": original_scores.tolist(),
    "restored_readout_scores": restored_scores.tolist(),
    "original_decisions": (original_scores > 0).tolist(),
    "restored_decisions": (restored_scores > 0).tolist(),
    "scope": "兩筆固定輸入、四個權重、CPU FP32。讀出是邏輯反例，不是實際模型品質實驗。",
    "result": "所有精確分數與 tensor 核對均通過。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
