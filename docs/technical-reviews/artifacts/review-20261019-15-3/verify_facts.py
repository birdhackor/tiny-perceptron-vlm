"""核對凍結算例、張量維度、路由統計與固定驗證資料的分母。"""

import ast
import contextlib
import hashlib
import io
import json
import math
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from scripts.course_experiments.architecture import _routing
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from tiny_perceptron.data import pad_batch
from tiny_perceptron.modern import MoEFFN


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check_close(actual, expected, tolerance):
    a = torch.as_tensor(actual, dtype=torch.float64)
    b = torch.as_tensor(expected, dtype=torch.float64)
    assert a.shape == b.shape, "實際值與預期值的形狀不同"
    error = float((a - b).abs().max())
    assert error <= tolerance, "實際值超出預期容忍差"
    return error


torch.set_num_threads(1)
torch.set_default_device("cpu")
source = ROOT / "docs/course-review-20261019/freeze-01/sources/15.3.md"
body = source.read_text()
code = body.split("```python\n", 1)[1].split("```", 1)[0]
namespace = {"__name__": "__main__"}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(code, str(source), "exec"), namespace)
x, weight, scores, prob, experts, output = (
    namespace[name] for name in ["x", "router_weight", "scores", "prob", "expert_outputs", "output"]
)
assert list(x.shape) == [1, 2], "輸入形狀不符"
assert list(weight.shape) == [2, 3], "路由矩陣形狀不符"
assert list(scores.shape) == [1, 3], "分數形狀不符"
assert list(experts.shape) == [1, 3, 2], "expert 堆疊形狀不符"
assert list(prob[:, :, None].shape) == [1, 3, 1], "混合係數形狀不符"
assert list(output.shape) == [1, 2], "混合輸出形狀不符"
prob_error = check_close(prob.tolist(), [[0.25, 0.25, 0.5]], 1e-7)
output_error = check_close(output.tolist(), [[2.25, 4.5]], 1e-6)
assert abs(math.exp(math.log(2)) - 2) <= 1e-15, "自然對數與指數的對應不符"
assert abs(math.log(2) - 0.6931) <= 5e-5, "自然對數近似值不符"
assert abs(math.e - 2.718) <= 5e-4, "自然對數底數近似值不符"

zero_weight = weight.clone()
zero_weight[0, 2] = 0
uniform = (x @ zero_weight).softmax(-1)
uniform_output = (uniform[:, :, None] * experts).sum(1)
check_close(uniform.tolist(), [[1 / 3] * 3], 1e-7)
check_close(uniform_output.tolist(), [[2, 4]], 1e-6)

learnable = weight.clone().requires_grad_()
mixed = ((x @ learnable).softmax(-1)[:, :, None] * experts).sum(1)
mixed.square().sum().backward()
assert float(learnable.grad.norm()) > 0, "可學矩陣未收到任務梯度"


class CountingExpert(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        return self.scale * features


counting = [CountingExpert(i) for i in (1, 2, 3)]
tiny_probability = torch.tensor([[-80.0, 0.0, 0.0]]).softmax(-1)
dense_outputs = torch.stack([expert(x) for expert in counting], dim=1)
_ = (tiny_probability[:, :, None] * dense_outputs).sum(1)
assert [expert.calls for expert in counting] == [1, 1, 1], "連續小係數未略過 expert"


class RoutingFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(experts=4, top_k=2)
        self.blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            block.ffn = MoEFFN(2, experts=4, top_k=2, hidden=4)
            block.ffn.experts = nn.ModuleList([nn.Identity() for _ in range(4)])
            with torch.no_grad():
                block.ffn.router.weight.copy_(
                    torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
                )
            self.blocks.append(block)

    def forward(self, ids, valid=None):
        features = torch.stack([ids.float(), torch.ones_like(ids).float()], dim=-1)
        for block in self.blocks:
            features = block.ffn(features)[0]
        return features


torch.manual_seed(42)
fixture = RoutingFixture()
examples = [
    (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
    (torch.tensor([4]), torch.tensor([5])),
]
fixture_result = _routing(fixture, examples, SimpleNamespace(device="cpu"))
ids, _, valid = pad_batch(examples)
features = torch.stack([ids.float(), torch.ones_like(ids).float()], dim=-1)[valid]
manual_probabilities = features @ fixture.blocks[0].ffn.router.weight.detach().T
manual_probabilities = manual_probabilities.softmax(-1)
manual_counts = torch.bincount(manual_probabilities.topk(2, dim=-1).indices.flatten(), minlength=4)
assert fixture_result["effective_input_tokens"] == 4, "統計未排除 padding"
for layer in fixture_result["layers"]:
    assert layer["dispatch_denominator"] == 8, "分派次數分母不符"
    assert layer["dispatch_counts"] == manual_counts.tolist(), "分派次數不同"
    check_close(layer["load_fraction"], (manual_counts.double() / 8).tolist(), 1e-15)
    check_close(layer["mean_router_probability"], manual_probabilities.double().mean(0).tolist(), 1e-7)

record_path = ROOT / "docs/course-experiments/results/moe.json"
record = json.loads(record_path.read_text())
variant = record["results"]["variants"]["top2_aux0.01"]
routing = variant["validation_routing"]
second = routing["layers"][1]
assert variant["model"]["config"]["top_k"] == 2, "實驗 top-k 設定不符"
assert variant["model"]["config"]["experts"] == 4, "實驗 expert 數量不符"
assert variant["training"]["auxiliary_weight"] == 0.01, "實驗輔助係數不符"
assert second["dispatch_denominator"] == 2 * routing["effective_input_tokens"] == 78512, "實驗分母不符"
assert sum(second["dispatch_counts"]) == 78512, "分派次數總和不符"
check_close(
    second["load_fraction"],
    [n / second["dispatch_denominator"] for n in second["dispatch_counts"]],
    1e-15,
)
assert [round(n, 4) for n in second["mean_router_probability"]] == [0.2875, 0.1772, 0.2652, 0.2701], "平均比例四捨五入不符"
assert [round(n, 4) for n in second["load_fraction"]] == [0.2065, 0.2692, 0.2260, 0.2983], "分派份額四捨五入不符"

archive = ROOT / "assets/training/tinystories-v1.tar.gz"
archive_raw = archive.read_bytes()
archive_receipt = {"local_sha256": sha(archive_raw), "local_bytes": len(archive_raw)}
if archive_raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
    archive_receipt["local_kind"] = "Git LFS 指標"
    archive_receipt["local_pointer"] = archive_raw.decode()
    archive_receipt["command_argv"] = [
        "curl", "-fsSL", "--max-time", "20",
        "https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/main/assets/training/tinystories-v1.tar.gz",
    ]
    archive_raw = subprocess.check_output(archive_receipt["command_argv"])
    archive_receipt["retrieval_exit_code"] = 0
assert sha(archive_raw) == record["assets"][0]["archive_sha256"], "讀取的資料包與實驗綁定版本不符"
archive_receipt["archive_sha256"] = sha(archive_raw)
archive_receipt["archive_bytes"] = len(archive_raw)
archive_receipt["scope"] = "資料包僅在記憶體中讀取，未寫入或改動 repo 的 LFS 指標。"
with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:gz") as tar:
    member = next(m for m in tar.getmembers() if m.name.endswith("tinystories-train-512.jsonl"))
    records = [json.loads(line) for line in tar.extractfile(member).read().decode().splitlines()]
for row in records:
    row["family"] = row.get("text_sha256", records_sha256([{"text": row["text"]}]))
dataset = split_records(records, record["seed"])
validation = dataset["validation"]
validation_examples = text_examples(validation, max_length=variant["model"]["config"]["max_length"])
validation_tokens = sum(len(pair[0]) for pair in validation_examples)
assert records_sha256(validation) == record["results"]["dataset"]["validation"]["sha256"], "重建驗證切分 SHA 不符"
assert validation_tokens == routing["effective_input_tokens"], "重建有效輸入 token 數不符"

architecture = "scripts/course_experiments/architecture.py"
expected_sha = record["code_sha256"][architecture]
commits = subprocess.check_output(["git", "log", "--all", "--format=%H", "--", architecture], cwd=ROOT, text=True).splitlines()
matched = None
for commit in commits:
    historical = subprocess.run(["git", "show", commit + ":" + architecture], cwd=ROOT, capture_output=True)
    if historical.returncode == 0 and sha(historical.stdout) == expected_sha:
        matched = commit
        break
assert matched is not None, "本機歷史未找到實驗記錄所綁定的程式"
old_text = historical.stdout.decode()
current_text = (ROOT / architecture).read_text()
same_functions = {}
for name in ["_routing", "run_moe", "_train", "_text_dataset"]:
    def function_ast(text):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.dump(node, include_attributes=False)

    same_functions[name] = function_ast(old_text) == function_ast(current_text)
    assert same_functions[name], "當前程式與實驗程式的關鍵函式不一致"

repository_files = {}
for path in [architecture, "tiny_perceptron/modern.py", "tiny_perceptron/data.py", "scripts/course_experiments/common.py"]:
    repository_files[path] = {"current_sha256": sha((ROOT / path).read_bytes()), "experiment_sha256": record["code_sha256"][path]}
    if path != architecture:
        assert repository_files[path]["current_sha256"] == repository_files[path]["experiment_sha256"], "相關程式版本不符"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-3/verify_facts.py",
    "result": "全部斷言通過；未執行訓練或載入模型權重。",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "dtype": str(x.dtype)},
    "source_sha256": sha(source.read_bytes()),
    "frozen_code_stdout": captured.getvalue(),
    "math": {"e": math.e, "log_2": math.log(2), "exp_0": math.exp(0), "exp_log_2": math.exp(math.log(2))},
    "toy": {"scores": scores.tolist(), "probability": prob.tolist(), "output": output.tolist(), "probability_max_absolute_error": prob_error, "output_max_absolute_error": output_error, "expert_outputs": experts.tolist(), "expanded_probability_shape": list(prob[:, :, None].shape), "product_shape": list((prob[:, :, None] * experts).shape)},
    "exercise": {"scores": (x @ zero_weight).tolist(), "probability": uniform.tolist(), "output": uniform_output.tolist()},
    "learning_signal": {"router_gradient": learnable.grad.tolist(), "router_gradient_norm": float(learnable.grad.norm()), "scope": "反向傳播可把任務損失的訊號傳給連續路由矩陣；未測試收斂或品質。"},
    "dense_compute": {"tiny_probability": tiny_probability.tolist(), "expert_call_counts": [expert.calls for expert in counting]},
    "scale_derivation": {"formula": "y=(1-p)*E1+p*E2；令 E1=1、目標 y=3，則 p=2/(E2-1)", "E2_4": 2 / 3, "E2_8": 2 / 7, "scope": "只證明混合係數受 expert 輸出尺度影響；不能據此判定實驗偏好成因。"},
    "routing_fixture": fixture_result,
    "record": {"path": str(record_path.relative_to(ROOT)), "sha256": sha(record_path.read_bytes()), "declared_revision": record["revision"], "seed": record["seed"], "device": record["device"], "torch": record["torch_version"], "python": record["python_version"], "model_config": variant["model"]["config"], "auxiliary_weight": variant["training"]["auxiliary_weight"], "training_steps": variant["training"]["steps"], "validation_routing": routing},
    "validation_reconstruction": {"archive_receipt": archive_receipt, "records": len(validation), "examples": len(validation_examples), "max_length": variant["model"]["config"]["max_length"], "records_sha256": records_sha256(validation), "input_tokens": validation_tokens, "dispatch_denominator": 2 * validation_tokens, "padding_excluded": True},
    "repository_provenance": {"matching_historical_code_commit": matched, "historical_code_sha256": expected_sha, "same_function_asts": same_functions, "files": repository_files, "scope": "以實驗所記錄的 SHA 找到原程式；關鍵函式與本次實跑的程式相同。未以紀錄的舊 revision 可讀性代替內容核對。"},
    "limitations": ["未重跑 CUDA 訓練。", "未載入原權重重算平均機率；該值核對的是引用實驗紀錄、四捨五入與統計方法。", "不由單次小型實驗外推模型品質、速度或普遍的 expert 專長。"],
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
