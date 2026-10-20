"""重算注意力算例，並以不同輸入展示加權混合不能唯一反推來源。"""

import ast
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
torch.set_default_device("cpu")
q = torch.tensor([[1.0, 0.0]])
k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
v = torch.tensor([[10.0, 20.0, 30.0], [-1.0, -2.0, -3.0]])
scores = q @ k.T
weights = scores.softmax(dim=-1)
output = weights @ v
a = math.exp(1) / (math.exp(1) + math.exp(0))
b = 1 / (math.exp(1) + math.exp(0))
exact_output = [a * x + b * y for x, y in zip(v[0].tolist(), v[1].tolist())]
rounded_output = [0.731 * x + 0.269 * y for x, y in zip(v[0].tolist(), v[1].tolist())]
five_digit_output = [0.73106 * x + 0.26894 * y for x, y in zip(v[0].tolist(), v[1].tolist())]
changed = v.clone()
changed[0] = torch.tensor([20.0, 40.0, 60.0])
changed_output = weights @ changed
exact_changed = [a * x + b * y for x, y in zip(changed[0].tolist(), changed[1].tolist())]
swapped_output = weights @ v.flip(0)
paired_weights = (q @ k.flip(0).T).softmax(dim=-1)
paired_output = paired_weights @ v.flip(0)
alternative = v.to(torch.float64).clone()
delta = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
alternative[0] += delta
alternative[1] -= (a / b) * delta
w64 = torch.tensor([[a, b]], dtype=torch.float64)
original64 = w64 @ v.to(torch.float64)
alternative64 = w64 @ alternative
general_q = torch.arange(8, dtype=torch.float64).reshape(4, 2)
general_k = torch.eye(2, dtype=torch.float64)
general_v = torch.arange(6, dtype=torch.float64).reshape(2, 3)
general_weights = (general_q @ general_k.T).softmax(dim=-1)
general_output = general_weights @ general_v
code_path = Path(__file__).parent / "section-execution.json"
code = json.loads(code_path.read_text())["copied_files"]["fence-1.py"]["text"]
tree = ast.parse(code)
calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
checks = {
    "分數": scores.tolist() == [[1.0, 0.0]],
    "比例": torch.allclose(weights, torch.tensor([[a, b]]), atol=1e-7, rtol=0),
    "原例三位小數": [round(x, 3) for x in exact_output] == [7.042, 14.083, 21.125],
    "取整比例三位小數": [round(x, 3) for x in rounded_output] == [7.041, 14.082, 21.123],
    "五位比例三位小數": [round(x, 3) for x in five_digit_output] == [7.042, 14.083, 21.125],
    "程式原例與公式": max(abs(x - y) for x, y in zip(output[0].tolist(), exact_output)) < 3e-6,
    "輸出形狀": list(output.shape) == [1, 3],
    "練習三位小數": [round(x, 3) for x in exact_changed] == [14.352, 28.704, 43.057],
    "練習與公式": max(abs(x - y) for x, y in zip(changed_output[0].tolist(), exact_changed)) < 5e-6,
    "只換內容保持比例": torch.equal(weights, (q @ k.T).softmax(dim=-1)),
    "只換內容列改變答案": not torch.allclose(swapped_output, output),
    "鍵值成對交換保持答案": torch.allclose(paired_output, output),
    "不同來源相同混合": torch.allclose(original64, alternative64, atol=1e-12, rtol=0),
    "一般形狀": list(general_weights.shape) == [4, 2] and list(general_output.shape) == [4, 3],
    "程式沒有線性層或訓練呼叫": all(
        name not in " ".join(calls) for name in ("Linear", "backward", "step", "optim")
    ),
}
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-4/verify.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "q": q.tolist(), "k": k.tolist(), "v": v.tolist(), "dtype": str(q.dtype),
    "mask": "無遮罩，未縮放，softmax 沿最後一軸",
    "scores": scores.tolist(), "weights_float32": weights.tolist(), "weights_float64": [a, b],
    "output_float32": output.tolist(), "output_float64": exact_output,
    "output_from_three_digit_weights": rounded_output,
    "output_from_five_digit_weights": five_digit_output,
    "changed_output_float32": changed_output.tolist(), "changed_output_float64": exact_changed,
    "swapped_values_output": swapped_output.tolist(), "paired_swap_output": paired_output.tolist(),
    "non_unique_values": alternative.tolist(), "non_unique_mixture": alternative64.tolist(),
    "non_unique_max_error": (original64 - alternative64).abs().max().item(),
    "general_shapes": {"weights": list(general_weights.shape), "values": list(general_v.shape), "output": list(general_output.shape)},
    "original_code_calls": calls, "checks": checks, "all_passed": all(checks.values()),
}
print(json.dumps(result, ensure_ascii=False, indent=2))
if not all(checks.values()):
    raise SystemExit("算例核對未通過")
