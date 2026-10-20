"""核對位置查表、逐格相加、梯度與因果混合的具體行為。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.attention import manual_attention


def execute(code):
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, "frozen-example", "exec"), namespace)
    return namespace, stdout.getvalue()


torch.manual_seed(1041)
torch.set_default_device("cpu")
torch.set_num_threads(1)
frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/4.1.md"
raw = frozen.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "4419cb0a3525d6dab1e584809ade34245cde44b3113621df67628a77cd945d11"
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
ns, original_stdout = execute(code)
e, p, ids, positions, x = (ns[name] for name in ("e", "p", "ids", "positions", "x"))
expected = torch.tensor([[1.0, 0.0, 0.0], [0.1, 1.2, 0.3], [1.2, 0.4, 0.6]])
assert positions.tolist() == [0, 1, 2]
assert x.shape == (3, 3)
assert torch.allclose(x, expected, rtol=0, atol=1e-7)
assert torch.equal(e(ids)[0], e(ids)[2])
assert not torch.equal(x[0], x[2])
assert torch.equal(p(torch.tensor([2])), p(torch.tensor([0, 2]))[1:])
baseline = {
    "positions": positions.tolist(),
    "token_vectors": e(ids).detach().tolist(),
    "position_vectors": p(positions).detach().tolist(),
    "sum": x.detach().tolist(),
    "maximum_absolute_error": (x - expected).abs().max().item(),
    "different_cat_vectors": not torch.equal(x[0], x[2]),
    "dtype": str(x.dtype),
    "grad_fn": type(x.grad_fn).__name__,
    "stdout": original_stdout,
}
assert baseline["grad_fn"] == "AddBackward0"
zero_code = code.replace("\nids =", "\n    p.weight.zero_()\nids =").replace(
    "assert not torch.equal(x[0], x[2])", "assert torch.equal(x[0], x[2])"
)
assert zero_code != code
zero_ns, zero_stdout = execute(zero_code)
assert torch.equal(zero_ns["x"], zero_ns["e"](zero_ns["ids"]))
zeroed = {"sum": zero_ns["x"].detach().tolist(), "equal_cat_vectors": True, "stdout": zero_stdout}
assert p(torch.arange(5)).shape == (5, 3)
try:
    p(torch.tensor([5]))
except IndexError as error:
    bounds = {"valid_positions": list(range(5)), "invalid_position": 5, "exception_type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("超出表格的索引應拒絕查詢")
assert e.weight.requires_grad and p.weight.requires_grad
assert torch.is_grad_enabled()
x.sum().backward()
assert e.weight.grad[1].tolist() == [2.0, 2.0, 2.0]
assert e.weight.grad[2].tolist() == [1.0, 1.0, 1.0]
assert p.weight.grad[:3].tolist() == [[1.0, 1.0, 1.0]] * 3
assert torch.equal(p.weight.grad[3:], torch.zeros(2, 3))
e_before, p_before = e.weight.detach().clone(), p.weight.detach().clone()
optimizer = torch.optim.SGD([e.weight, p.weight], lr=0.1)
optimizer.step()
assert not torch.equal(e_before, e.weight)
assert not torch.equal(p_before, p.weight)
gradients = {
    "token_table_gradient": e.weight.grad.tolist(),
    "position_table_gradient": p.weight.grad.tolist(),
    "both_tables_updated": True,
    "unqueried_position_rows_unchanged": torch.equal(p.weight[3:], p_before[3:]),
    "loss": "x.sum()；只測梯度路徑，不代表語言任務訓練效果",
    "optimizer": "SGD, lr=0.1, one update",
}
assert gradients["unqueried_position_rows_unchanged"]
q = torch.zeros(1, 1, 3, 3)
k = torch.zeros_like(q)
allowed = torch.ones(3, 3, dtype=torch.bool).tril()[None, None]
cat = torch.tensor([1.0, 0.0, 0.0])
look = torch.tensor([0.0, 1.0, 0.0])
mixed_v = torch.stack([cat, look, cat])[None, None]
mixed_out, mixed_weights = manual_attention(q, k, mixed_v, allowed)
same_v = cat.repeat(3, 1)[None, None]
same_out, same_weights = manual_attention(q, k, same_v, allowed)
assert not torch.equal(mixed_out[0, 0, 0], mixed_out[0, 0, 2])
assert torch.equal(same_out, same_v)
assert torch.allclose(mixed_out[0, 0, 2], torch.tensor([2 / 3, 1 / 3, 0.0]), rtol=0, atol=1e-7)
causal = {
    "query_and_key": "全零，所有允許位置的分數相同",
    "allowed": allowed[0, 0].tolist(),
    "mixed_values": mixed_v[0, 0].tolist(),
    "weights": mixed_weights[0, 0].tolist(),
    "mixed_output": mixed_out[0, 0].tolist(),
    "same_values_output": same_out[0, 0].tolist(),
    "scope": "確定性且無 dropout 的加權和；一般相同 value 的證明另用 softmax 權重和為一的代數推導。",
}
ignored = expected @ torch.zeros(3, 2)
assert torch.equal(ignored[0], ignored[2])
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-4-1/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": "torch.float32", "seed": "1041"},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "attention_source_sha256": hashlib.sha256((ROOT / "tiny_perceptron/attention.py").read_bytes()).hexdigest(),
    "baseline": baseline,
    "zeroed_position_table": zeroed,
    "bounds": bounds,
    "gradients": gradients,
    "causal_mixing": causal,
    "downstream_ignores_position": ignored.tolist(),
    "result": "所有斷言通過；未進行語言模型訓練或品質／速度測量。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
