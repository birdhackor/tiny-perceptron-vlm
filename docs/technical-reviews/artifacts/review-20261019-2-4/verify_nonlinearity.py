"""核對凍結範例、練習修改與圖中座標；不訓練模型。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


root = Path(__file__).resolve().parents[4]
source = root / "docs/course-review-20261019/freeze-01/sources/2.4.md"
raw = source.read_bytes()
body = raw.decode("utf-8")
code = re.search(r"```python\n(.*?)```", body, re.S).group(1)
namespace = {}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(code, str(source), "exec"), namespace)

a, b = namespace["a"], namespace["b"]
x = namespace["x"]
hidden = namespace["hidden"]
linear = namespace["linear"]
curved = namespace["curved"]
combined = namespace["combined"]
relu = namespace["F"].relu
expected_hidden = torch.tensor([[-2.0, 2.0], [0.0, 0.0], [2.0, -2.0]])
expected_relu = torch.tensor([[0.0, 2.0], [0.0, 0.0], [2.0, 0.0]])
assert torch.equal(hidden, expected_hidden)
assert torch.equal(relu(hidden), expected_relu)
assert torch.equal(linear, torch.zeros(3, 1))
assert torch.equal(curved, torch.tensor([[2.0], [0.0], [2.0]]))
assert torch.equal(combined, linear)
assert torch.equal(b.weight @ a.weight, torch.zeros(1, 1))
assert a.bias is None and b.bias is None
assert tuple(a.weight.shape) == (2, 1)
assert tuple(b.weight.shape) == (1, 2)
assert torch.equal(a.weight.T, a.weight.transpose(0, 1))
assert torch.equal(x @ (b.weight @ a.weight).T, torch.matmul(x, (b.weight @ a.weight).T))
assert tuple(curved.squeeze(-1).shape) == (3,)
assert tuple(torch.zeros(1, 3, 1).squeeze(-1).shape) == (1, 3)
assert tuple(torch.zeros(3, 2).squeeze(-1).shape) == (3, 2)

before = [parameter.detach().clone() for parameter in [a.weight, b.weight]]
changed_x = torch.tensor([[-2.0], [0.0], [3.0]])
changed_hidden = a(changed_x)
changed_linear = b(changed_hidden)
changed_curved = b(relu(changed_hidden))
removed_relu = b(changed_hidden)
assert torch.equal(changed_linear, torch.zeros(3, 1))
assert torch.equal(changed_curved, torch.tensor([[2.0], [0.0], [3.0]]))
assert torch.equal(removed_relu, torch.zeros(3, 1))
assert all(torch.equal(old, new) for old, new in zip(before, [a.weight, b.weight]))
assert a.weight.grad is None and b.weight.grad is None
assert hidden.grad_fn is not None and linear.squeeze(-1).grad_fn is not None
calls = [node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)]
assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in {"backward", "step"} for node in calls)

svg_path = root / "course/figures/rewrite-02-nonlinearity.svg"
svg = ET.fromstring(svg_path.read_bytes())
circles = [(float(item.attrib["cx"]), float(item.attrib["cy"])) for item in svg.findall(".//{http://www.w3.org/2000/svg}circle")]
assert circles == [(165.0, 360.0), (320.0, 360.0), (475.0, 360.0), (165.0, 645.0), (320.0, 770.0), (475.0, 645.0)]
receipt_path = root / "docs/course-review-20261019/figure-renders-04.json"
receipt = json.loads(receipt_path.read_bytes())
record = next(item for item in receipt["records"] if item["page_id"] == "2.4")
png_path = root / record["png"]["path"]
assert hashlib.sha256(png_path.read_bytes()).hexdigest() == record["png"]["sha256"]
assert hashlib.sha256(svg_path.read_bytes()).hexdigest() == record["frozen_svg"]["sha256"]

result = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": str(x.device), "dtype": str(x.dtype)},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "original_stdout": captured.getvalue(),
    "weights": {"a": a.weight.tolist(), "b": b.weight.tolist(), "biases": [a.bias, b.bias]},
    "values": {"hidden": hidden.tolist(), "relu_hidden": relu(hidden).tolist(), "linear": linear.tolist(), "curved": curved.tolist(), "combined_weight": (b.weight @ a.weight).tolist(), "combined": combined.tolist()},
    "shapes": {"input": list(x.shape), "hidden": list(hidden.shape), "output": list(curved.shape), "squeezed": list(curved.squeeze(-1).shape)},
    "exercise": {"input": changed_x.tolist(), "linear": changed_linear.tolist(), "curved": changed_curved.tolist(), "removed_relu": removed_relu.tolist()},
    "autograd": {"hidden_grad_fn": type(hidden.grad_fn).__name__, "squeezed_grad_fn": type(linear.squeeze(-1).grad_fn).__name__, "parameter_gradients": [a.weight.grad, b.weight.grad], "weights_unchanged_by_forward": True, "backward_or_step_calls_in_original": False},
    "figure_receipt": record,
    "figure_circle_coordinates": circles,
    "result": "所有確定性數值、形狀、練習修改與圖檔雜湊斷言通過；未執行訓練。"
}
print(json.dumps(result, ensure_ascii=False, indent=2))
