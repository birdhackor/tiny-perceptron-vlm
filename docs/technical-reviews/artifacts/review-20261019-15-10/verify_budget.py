"""獨立核對參數計數、查表範圍與額外 tensor；不訓練模型。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from scripts.course_experiments.architecture import _parameter_budget
from tiny_perceptron.model import ModelConfig, TinyLM


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(1)
torch.manual_seed(20261020)
snapshot = ROOT / "docs/course-review-20261019/freeze-04/sources/15.10.md"
body = snapshot.read_text()
current = (ROOT / "course/chapters/15.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", current))
index = next(i for i, h in enumerate(headers) if h[0].startswith(b"## 15.10 "))
end = headers[index + 1].start() if index + 1 < len(headers) else len(current)
assert current[headers[index].start():end] == snapshot.read_bytes()
code = re.search(r"```python\n(.*?)```", body, re.S).group(1)
toy = []
for k in (2, 1):
    selected_code = code.replace("width, hidden, k = 4, 8, 2", f"width, hidden, k = 4, 8, {k}")
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(selected_code, "frozen_section_python", "exec"), {})
    expected = {2: [(4, 272, 144, 1088), (8, 544, 160, 2176)],
                1: [(4, 272, 80, 1088), (8, 544, 96, 2176)]}[k]
    pattern = r"expert數 (\d+) 總權重 (\d+) 每token使用估計 (\d+) FP32權重bytes (\d+)"
    actual = [tuple(map(int, re.fullmatch(pattern, line).groups()))
              for line in captured.getvalue().splitlines()]
    assert actual == expected
    toy.append({"k": k, "stdout": captured.getvalue(), "rows": actual})
expert = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 4, bias=False))
assert sum(p.numel() for p in expert.parameters()) == 64

record_path = ROOT / "docs/course-experiments/results/moe.json"
record = json.loads(record_path.read_bytes())
models = []
for name, variant in record["results"]["variants"].items():
    if not variant["model"]["config"]["experts"]:
        continue
    model = TinyLM(ModelConfig(**variant["model"]["config"]))
    named = list(model.named_parameters())
    expert_count = sum(p.numel() for n, p in named if ".ffn.experts." in n)
    router_count = sum(p.numel() for n, p in named if ".ffn.router." in n)
    total = sum(p.numel() for _, p in named)
    others = total - expert_count - router_count
    independent = {"total_parameters": total, "expert_parameters": expert_count,
                   "router_parameters": router_count,
                   "active_parameter_proxy_per_token": total - expert_count +
                   sum(sum(p.numel() for p in block.ffn.experts[0].parameters()) *
                       block.ffn.top_k for block in model.blocks)}
    assert independent == _parameter_budget(model) == variant["budget"]
    assert (expert_count, router_count, others, total) == (264704, 512, 75392, 340608)
    k = model.config.top_k
    assert independent["active_parameter_proxy_per_token"] == {1: 142080, 2: 208256}[k]
    per_expert = sum(p.numel() for p in model.blocks[0].ffn.experts[0].parameters())
    assert per_expert == 64 * 256 + 256 + 256 * 64 + 64 == 33088
    assert model.embedding.weight.numel() == 264 * 64 == 16896
    assert model.position.weight.numel() == 128 * 64 == 8192
    models.append({"variant": name, "config": variant["model"]["config"],
                   "independent": independent, "other_parameters": others,
                   "one_expert_with_biases": per_expert,
                   "embedding_shape": list(model.embedding.weight.shape),
                   "position_shape": list(model.position.weight.shape),
                   "one_embedding_row_elements": model.config.width,
                   "other_breakdown": {"token_embedding": 16896, "position": 8192,
                                       "output": 16896, "attention": 32768, "norms": 640}})

model.eval()
ids = torch.tensor([[4, 5, 6]])
with torch.no_grad():
    prefill = model(ids)
    cached = model(torch.tensor([[7]]), cache=prefill["cache"])
    full = model(torch.tensor([[4, 5, 6, 7]]))
cache_error = (cached["logits"][:, -1] - full["logits"][:, -1]).abs().max().item()
assert torch.allclose(cached["logits"][:, -1], full["logits"][:, -1], atol=2e-6, rtol=2e-5)
cache = {"prefill_shapes": [[list(t.shape) for t in pair] for pair in prefill["cache"]],
         "decode_shapes": [[list(t.shape) for t in pair] for pair in cached["cache"]],
         "last_token_logits_max_abs_error": cache_error,
         "tolerance": {"atol": 2e-6, "rtol": 2e-5}}

parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
optimizer = torch.optim.Adam([parameter], lr=0.01)
parameter.square().sum().backward()
assert parameter.grad.item() == 4.0
gradient = parameter.grad.detach().clone()
optimizer.step()
state = optimizer.state[parameter]
assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
adam = {"gradient_before_step": gradient.tolist(),
        "parameter_bytes": parameter.numel() * parameter.element_size(),
        "gradient_bytes": parameter.grad.numel() * parameter.grad.element_size(),
        "state": {n: {"value": t.tolist(), "shape": list(t.shape),
                      "bytes": t.numel() * t.element_size()} for n, t in state.items()}}
assert torch.finfo(torch.float32).bits == 32
assert parameter.element_size() == 4
paths = ["tiny_perceptron/model.py", "tiny_perceptron/modern.py", "tiny_perceptron/attention.py",
         "scripts/course_experiments/architecture.py", "docs/course-experiments/results/moe.json"]
hashes = {name: sha(ROOT / name) for name in paths}
historical = {name: {"recorded": record["code_sha256"][name], "current": hashes[name],
                     "matches": record["code_sha256"][name] == hashes[name]}
              for name in paths if name in record["code_sha256"]}
historical_command = ["git", "show", f"{record['revision']}:scripts/course_experiments/architecture.py"]
historical_process = subprocess.run(historical_command, cwd=ROOT, capture_output=True, text=True)


def budget_function(source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_parameter_budget")
    return ast.dump(node, include_attributes=False)


budget_function_matches = None
if historical_process.returncode == 0:
    budget_function_matches = budget_function(historical_process.stdout) == budget_function(
        (ROOT / "scripts/course_experiments/architecture.py").read_text())
result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-10/verify_budget.py",
          "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                          "torch_git_version": str(torch.version.git_version), "device": "cpu",
                          "platform": platform.platform(), "seed": "20261020"},
          "result": "全部斷言通過；僅核對計數與小型 tensor 行為。",
          "source_sha256": sha(snapshot), "freeze_sha256": sha(ROOT / "docs/course-review-20261019/freeze-04.json"),
          "repository_sha256": hashes, "historical_code_comparison": historical,
          "historical_budget_function": {"command_argv": historical_command,
                                         "exit_code": historical_process.returncode,
                                         "stderr": historical_process.stderr,
                                         "ast_matches_current": budget_function_matches},
          "fp32": {"bits": 32, "element_bytes": 4}, "toy": toy,
          "model_budgets": models, "cache": cache, "adam": adam,
          "scope": "未載入已訓練權重、未重訓、未量測效能或 GPU 記憶體；參數數量與權重值無關。"}
print(json.dumps(result, ensure_ascii=False, indent=2))
