"""核對正規化算例、模型接線、比較 API 與已保存實驗數值。"""

import ast
import contextlib
import hashlib
import inspect
import io
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import RMSNorm
from tiny_perceptron.training import seed_everything


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(1)
torch.manual_seed(42)
source = ROOT / "docs/course-review-20261019/freeze-01/sources/14.2.md"
fences = re.findall(r"```python\n(.*?)```", source.read_text(), re.S)
namespace = {}
original_stdout = io.StringIO()
with contextlib.redirect_stdout(original_stdout):
    for fence in fences:
        exec(compile(fence, str(source), "exec"), namespace)
excerpt = ast.parse(fences[1]).body[0]
library = ast.parse(inspect.getsource(RMSNorm)).body[0]
assert ast.dump(excerpt) == ast.dump(library), "節錄類別必須與函式庫相同"
x = namespace["x"]
eps = namespace["eps"]
ln = namespace["ln"]
rms = namespace["rms"]
reference_ln = torch.tensor([[0, 0, 0], [-1.2247, 0, 1.2247]])
reference_rms = torch.tensor([[1, 1, 1], [0.4629, 0.9258, 1.3887]])
assert torch.allclose(ln, reference_ln, atol=5e-5, rtol=0), "LN 的四位小數不符"
assert torch.allclose(rms, reference_rms, atol=5e-5, rtol=0), "RMS 的四位小數不符"
assert x.mean(-1, keepdim=True).shape == (2, 1), "保留軸後的形狀不符"
assert torch.equal(F.layer_norm(x, (3,), eps=eps), torch.nn.LayerNorm(3, eps=eps)(x))
norm = RMSNorm(3)
norm(x).sum().backward()
assert norm.weight.grad is not None and torch.all(norm.weight.grad != 0)
assert list(norm.state_dict()) == ["weight"]
assert list(dict(norm.named_parameters())) == ["weight"]
assert torch.equal(x.float(), x)
scale = (x.float().square().mean(-1, keepdim=True) + eps).rsqrt()
assert torch.equal(scale.to(x.dtype), scale)
assert torch.equal(norm(torch.zeros_like(x)), torch.zeros_like(x))

exercise_stdout = io.StringIO()
exercise_namespace = {}
with contextlib.redirect_stdout(exercise_stdout):
    for index, fence in enumerate(fences):
        if index == 0:
            fence = fence.replace("eps = 1e-5", "x = x + 10\neps = 1e-5")
        exec(compile(fence, str(source), "exec"), exercise_namespace)
assert torch.allclose(exercise_namespace["ln"], ln, atol=1e-5, rtol=0)
assert torch.allclose(
    exercise_namespace["rms"][1], torch.tensor([0.9146, 0.9977, 1.0808]), atol=5e-5, rtol=0
)
assert torch.allclose(exercise_namespace["norm"](exercise_namespace["x"]), exercise_namespace["rms"])

models = {kind: TinyLM(ModelConfig(width=64, layers=2, heads=4, norm=kind)) for kind in ("layer", "rms")}
model_results = {}
for kind, model in models.items():
    normalizers = [(name, module) for name, module in model.named_modules() if isinstance(module, (torch.nn.LayerNorm, RMSNorm))]
    assert len(normalizers) == 5
    model_results[kind] = {
        "description": model.description(),
        "normalizers": [{"name": name, "type": type(module).__name__, "parameters": {n: p.numel() for n, p in module.named_parameters()}} for name, module in normalizers],
    }
assert model_results["layer"]["description"]["parameters"] == 141568
assert model_results["rms"]["description"]["parameters"] == 141248

positive_zero = torch.tensor([0.0], dtype=torch.float32)
negative_zero = torch.tensor([-0.0], dtype=torch.float32)
equal_counterexamples = {
    "same_dtype_signed_zero": {
        "torch_equal": torch.equal(positive_zero, negative_zero),
        "positive_bits": positive_zero.view(torch.int32).tolist(),
        "negative_bits": negative_zero.view(torch.int32).tolist(),
    },
    "different_dtypes": {
        "torch_equal": torch.equal(torch.tensor([1], dtype=torch.int32), torch.tensor([1.0], dtype=torch.float32)),
        "left_dtype": "torch.int32",
        "right_dtype": "torch.float32",
    },
}
assert equal_counterexamples["same_dtype_signed_zero"]["torch_equal"]
assert equal_counterexamples["different_dtypes"]["torch_equal"]
assert positive_zero.view(torch.int32).item() != negative_zero.view(torch.int32).item()

experiment_path = ROOT / "docs/course-experiments/results/modern.json"
experiment = json.loads(experiment_path.read_text())
experiment_results = {}
for name in ("baseline", "rmsnorm"):
    variant = experiment["results"]["variants"][name]
    training = variant["training"]
    heldout = {}
    for split in ("validation", "test"):
        saved = variant["heldout"][split]
        recomputed = saved["nll_sum"] / saved["effective_tokens"]
        assert abs(recomputed - saved["nll"]) < 1e-12
        heldout[split] = {key: saved[key] for key in ("nll", "nll_sum", "effective_tokens", "examples", "records")}
        heldout[split]["recomputed_nll"] = recomputed
    experiment_results[name] = {
        "model": variant["model"],
        "training": {key: training[key] for key in ("requested_steps", "steps", "optimizer_updates", "skipped_updates", "batch_size", "effective_tokens", "learning_rate", "gradient_clip_norm", "warm_step_median_seconds")},
        "warm_step_median_milliseconds": 1000 * training["warm_step_median_seconds"],
        "heldout": heldout,
    }
assert [round(experiment_results[n]["heldout"]["validation"]["nll"], 5) for n in ("baseline", "rmsnorm")] == [2.26331, 2.25913]
assert [round(experiment_results[n]["heldout"]["test"]["nll"], 5) for n in ("baseline", "rmsnorm")] == [2.29163, 2.28631]
assert [round(experiment_results[n]["warm_step_median_milliseconds"], 3) for n in ("baseline", "rmsnorm")] == [13.062, 13.952]
paths = ["tiny_perceptron/model.py", "tiny_perceptron/modern.py", "scripts/course_experiments/architecture.py", "scripts/course_experiments/common.py", "tiny_perceptron/training.py"]
code_match = {p: sha(ROOT / p) == experiment["code_sha256"][p] for p in paths}
architecture_path = "scripts/course_experiments/architecture.py"
assert all(matches for path, matches in code_match.items() if path != architecture_path)
historical_url = "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/" + experiment["revision"] + "/" + architecture_path
historical_command = ["curl", "-fLsS", "--max-time", "30", historical_url]
historical_raw = subprocess.run(historical_command, capture_output=True, check=True).stdout
assert hashlib.sha256(historical_raw).hexdigest() == experiment["code_sha256"][architecture_path]


def function_text(raw, name):
    tree = ast.parse(raw)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(raw, function)


current_architecture = (ROOT / architecture_path).read_text()
historical_architecture = historical_raw.decode()
function_comparison = {}
for name in ("_sync", "_train", "run_modern"):
    previous = function_text(historical_architecture, name)
    current = function_text(current_architecture, name)
    assert previous == current, "實驗相關函式必須與記錄版本相同"
    function_comparison[name] = {"identical": True, "sha256": hashlib.sha256(previous.encode()).hexdigest()}

seed_results = []
for value in (42, 42, 43):
    seed_everything(value)
    seed_results.append({"seed": value, "initialization": torch.randn(5).tolist(), "samples": random.Random(value).choices(list(range(10)), k=8)})
assert seed_results[0] == seed_results[1]
assert seed_results[0]["initialization"] != seed_results[2]["initialization"]
assert seed_results[0]["samples"] != seed_results[2]["samples"]

doc_objects = {"float": torch.Tensor.float, "allclose": torch.allclose, "equal": torch.equal, "Parameter": torch.nn.Parameter, "state_dict": torch.nn.Module.state_dict, "manual_seed": torch.manual_seed}
doc_excerpts = {name: (obj.__doc__ or "")[:750] for name, obj in doc_objects.items()}
normalization_file = Path(inspect.getfile(torch.nn.LayerNorm))
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu"},
    "source_sha256": sha(source),
    "code_sha256": {p: sha(ROOT / p) for p in paths},
    "original_stdout": original_stdout.getvalue(),
    "exercise_stdout": exercise_stdout.getvalue(),
    "numeric": {
        "mean": x.mean(-1).tolist(), "population_variance": x.var(-1, correction=0).tolist(),
        "ideal_rms": [2, math.sqrt(14 / 3)], "stabilized_rms_denominator": (x.square().mean(-1) + eps).sqrt().tolist(),
        "ln": ln.tolist(), "rms": rms.tolist(), "shifted_ln": exercise_namespace["ln"].tolist(), "shifted_rms": exercise_namespace["rms"].tolist(),
        "gamma_gradient": norm.weight.grad.tolist(), "shape_with_keepdim": list(x.mean(-1, keepdim=True).shape),
        "multiply_rsqrt_max_abs_difference": float((norm(x) - rms).detach().abs().max()),
        "float32_casts_unchanged": True,
    },
    "models": model_results,
    "torch_equal_counterexamples": equal_counterexamples,
    "historical_experiment_code": {"url": historical_url, "command_argv": historical_command, "downloaded_sha256": hashlib.sha256(historical_raw).hexdigest(), "functions": function_comparison, "inspection_note": "全檔 hash 不同，但計時、同步及 modern 實驗入口函式逐字相同。歷史版本用實驗記錄中的完整 SHA-256 核對。"},
    "runtime_official_docstrings": doc_excerpts,
    "installed_normalization_source": {"path": str(normalization_file), "sha256": sha(normalization_file)},
    "experiment": {"path": str(experiment_path.relative_to(ROOT)), "sha256": sha(experiment_path), "seed": experiment["seed"], "device": experiment["device"], "gpu": experiment["gpu"], "torch": experiment["torch_version"], "python": experiment["python_version"], "runtime": experiment["results"]["runtime"], "dataset": experiment["results"]["dataset"], "variants": experiment_results, "current_code_matches_recorded_hashes": code_match, "median_measurement_count_per_variant": 237, "median_scope": "保存值由每步同步後的前向、loss、反向、梯度有限檢查、clip、optimizer/scaler 更新及資料搬移計時產生；240 步略去前 3 步，未取得逐步原始 latency，未重新訓練。"},
    "seed_replay": seed_results,
    "initial_attempt": {"exit_code": 1, "stderr": "AssertionError: 當前相關程式必須與實驗記錄的版本相符", "cause": "原核對程式要求全部相關檔的全檔 hash 相同；architecture.py 另有新增函式。後續以 curl 取回與記錄 SHA 完全一致的原始碼，確認三個相關函式逐字相同。"},
    "result": "所有數值、接線與保存實驗值核對通過；torch.equal 的逐位元比較描述有直接反例。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
