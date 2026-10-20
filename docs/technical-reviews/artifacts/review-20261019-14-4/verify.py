"""核對固定算例、模型結構與既有實驗紀錄；不重新訓練。"""

import ast
import hashlib
import io
import json
import math
import platform
import random
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import DenseFFN


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


torch.set_num_threads(1)
torch.manual_seed(42)
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-4/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "scalar": {"inputs": [1, 2], "linear": [3 * (2 * x) for x in [1, 2]], "square_between": [3 * (2 * x) ** 2 for x in [1, 2]], "square_ratios": [3 * (2 * x) ** 2 / x for x in [1, 2]]},
}
xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
x = torch.tensor(xs)
phi = [0.5 * (1 + math.erf(v / math.sqrt(2))) for v in xs]
expected = [v * p for v, p in zip(xs, phi)]
gelu = F.gelu(x)
relu2 = F.relu(x).square()
error = max(abs(a - b) for a, b in zip(expected, gelu.tolist()))
assert error < 2e-7, "GELU 與 erf 公式不相符"
assert relu2.tolist() == [0, 0, 0, 1, 4], "ReLU 平方與預期不相符"
output["activation_values"] = {"x": xs, "phi_math_erf": phi, "gelu_math_erf": expected, "gelu_torch": gelu.tolist(), "gelu_rounded": gelu.round(decimals=4).tolist(), "relu2": relu2.tolist(), "max_abs_formula_error": error, "three_inputs_relu": F.relu(torch.tensor([-1.0, 0.0, 2.0])).tolist(), "three_inputs_relu2": F.relu(torch.tensor([-1.0, 0.0, 2.0])).square().tolist()}
shapes = []
for shape in [(), (5,), (2, 3, 4)]:
    value = torch.randn(shape)
    assert F.gelu(value).shape == value.shape
    assert F.relu(value).square().shape == value.shape
    shapes.append(list(shape))
output["preserved_shapes"] = shapes
exercise = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
output["exercise"] = {"dtype": str(exercise.dtype), "gelu": F.gelu(exercise).tolist(), "rounded_gelu": F.gelu(exercise).round(decimals=4).tolist(), "relu2": F.relu(exercise).square().tolist()}
integer = torch.tensor([-4, -2, 0, 2, 4])
try:
    F.gelu(integer)
except (RuntimeError, NotImplementedError) as error:
    output["integer_gelu"] = {"dtype": str(integer.dtype), "exception": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("整數 GELU 未拒絕輸入")

models = {}
state_keys = {}
for activation in ["gelu", "relu2"]:
    config = ModelConfig(width=64, layers=2, heads=4, activation=activation)
    model = TinyLM(config)
    models[activation] = {"parameters": sum(p.numel() for p in model.parameters()), "block_activations": [b.ffn.activation for b in model.blocks], "ffn_linear_shapes": [[list(b.ffn.up.weight.shape), list(b.ffn.down.weight.shape)] for b in model.blocks]}
    state_keys[activation] = {k: list(v.shape) for k, v in model.state_dict().items()}
assert models["gelu"]["parameters"] == models["relu2"]["parameters"] == 141568
assert state_keys["gelu"] == state_keys["relu2"]
output["models"] = models
left = DenseFFN(8)
right = DenseFFN(8, activation="relu2")
right.load_state_dict(left.state_dict())
sample = torch.randn(2, 3, 8)
hidden = left.up(sample)
assert torch.equal(left(sample), left.down(F.gelu(hidden)))
assert torch.equal(right(sample), right.down(F.relu(hidden).square()))
output["activation_swap_same_weights"] = {"state_shapes_identical": True, "gelu_activation_mean": float(F.gelu(hidden).mean().detach()), "relu2_activation_mean": float(F.relu(hidden).square().mean().detach()), "gelu_negative_count": int((F.gelu(hidden) < 0).sum()), "relu2_negative_count": int((F.relu(hidden).square() < 0).sum()), "output_max_abs_difference": float((left(sample) - right(sample)).abs().max().detach())}

record_path = ROOT / "docs/course-experiments/results/modern.json"
record_raw = record_path.read_bytes()
record = json.loads(record_raw)
result = record["results"]
archive_path = ROOT / "assets/training/tinystories-v1.tar.gz"
archive_raw = archive_path.read_bytes()
archive_receipt = {"local_path": archive_path.relative_to(ROOT).as_posix(), "local_sha256": sha(archive_raw)}
if archive_raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
    archive_receipt["local_lfs_pointer"] = archive_raw.decode()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    url = "https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/" + revision + "/assets/training/tinystories-v1.tar.gz"
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "30", url]
    archive_raw = subprocess.check_output(command)
    archive_receipt.update(url=url, command=command, note="從 Git LFS 的 HTTPS 端點讀入記憶體；未修改本機資料包。")
assert sha(archive_raw) == record["assets"][0]["archive_sha256"]
with tarfile.open(fileobj=io.BytesIO(archive_raw)) as archive:
    member = next(m for m in archive.getmembers() if m.name.endswith("tinystories-train-512.jsonl"))
    dataset_raw = archive.extractfile(member).read()
records = [json.loads(line) for line in dataset_raw.decode().splitlines() if line.strip()]
for row in records:
    row["family"] = row.get("text_sha256", records_sha256([{"text": row["text"]}]))
data = split_records(records, record["seed"])
data_verification = {}
for name, rows in data.items():
    examples = text_examples(rows)
    actual = {"records": len(rows), "sha256": records_sha256(rows), "examples": len(examples), "effective_tokens": sum(int((y != IGNORE).sum()) for _, y in examples)}
    assert actual["records"] == result["dataset"][name]["records"]
    assert actual["sha256"] == result["dataset"][name]["sha256"]
    data_verification[name] = actual
examples = text_examples(data["train"])
sampling = []
for _ in range(2):
    sampler = random.Random(record["seed"])
    batches = [sampler.choices(range(len(examples)), k=16) for step in range(240)]
    tokens = sum(int((examples[i][1] != IGNORE).sum()) for batch in batches for i in batch)
    sampling.append({"batch_indices_sha256": sha(json.dumps(batches).encode()), "effective_tokens": tokens, "steps": len(batches), "batch_size": 16})
assert sampling[0] == sampling[1]
selected = {}
for name in ["baseline", "relu2"]:
    variant = result["variants"][name]
    training = variant["training"]
    assert training["effective_tokens"] == sampling[0]["effective_tokens"]
    assert training["optimizer_updates"] == 240 and training["skipped_updates"] == 0
    heldout = {}
    for split, metrics in variant["heldout"].items():
        assert metrics["effective_tokens"] == data_verification[split]["effective_tokens"]
        assert metrics["examples"] == data_verification[split]["examples"]
        reconstructed = metrics["nll_sum"] / metrics["effective_tokens"]
        assert abs(reconstructed - metrics["nll"]) < 1e-12
        heldout[split] = {k: metrics[k] for k in ["nll", "nll_sum", "effective_tokens", "examples", "records"]}
        heldout[split]["nll_rounded"] = round(metrics["nll"], 5)
    first = variant["heldout"]["test"]["samples"][0]
    assert ByteTokenizer().decode(first["generated_ids"]) == first["generated"]
    selected[name] = {"model": variant["model"], "training": {k: v for k, v in training.items() if k != "history"}, "heldout": heldout, "first_test_sample": first, "step_median_ms": training["warm_step_median_seconds"] * 1000, "step_median_ms_rounded": round(training["warm_step_median_seconds"] * 1000, 3)}

source_hashes = {}
for relative in ["tiny_perceptron/modern.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py", "scripts/course_experiments/common.py", "scripts/course_experiments/architecture.py"]:
    raw = (ROOT / relative).read_bytes()
    source_hashes[relative] = {"current_sha256": sha(raw), "recorded_sha256": record["code_sha256"][relative]}
target_hash = record["code_sha256"]["scripts/course_experiments/architecture.py"]
revisions = subprocess.check_output(["git", "log", "--format=%H", "--", "scripts/course_experiments/architecture.py"], cwd=ROOT, text=True).splitlines()
historical = []
current_tree = ast.parse((ROOT / "scripts/course_experiments/architecture.py").read_bytes())
identifiers = ["_clone_config", "_text_dataset", "_nll", "_train", "run_modern", "_sync"]
current_functions = {n.name: ast.dump(n, include_attributes=False) for n in current_tree.body if isinstance(n, ast.FunctionDef) and n.name in identifiers}
for revision in revisions:
    old = subprocess.check_output(["git", "show", revision + ":scripts/course_experiments/architecture.py"], cwd=ROOT)
    old_tree = ast.parse(old)
    old_functions = {n.name: ast.dump(n, include_attributes=False) for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name in identifiers}
    historical.append({"revision": revision, "sha256": sha(old), "matches_recorded_full_file": sha(old) == target_hash, "compared_functions_ast_equal": {name: current_functions[name] == old_functions.get(name) for name in identifiers}})
    if sha(old) == target_hash:
        break
output["experiment_record_audit"] = {"path": record_path.relative_to(ROOT).as_posix(), "sha256": sha(record_raw), "runtime": {k: record[k] for k in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status"]}, "archive_receipt": archive_receipt, "archive_sha256": sha(archive_raw), "dataset_member": member.name, "dataset_member_sha256": sha(dataset_raw), "data_verification": data_verification, "reconstructed_sampling": sampling, "selected": selected, "source_hashes": source_hashes, "historical_architecture_inspection": historical, "timing_scope": "每步計時包含 forward、backward、clip 與 optimizer step；CUDA 同步；排除前 3 步後的 237 步中位數。既有紀錄未保留原始 latency 陣列，本次沒有重跑 GPU 訓練或重新量測中位數。", "support_limit": "核對保存的實驗數字、來源、分母與可重建配置，不把本次 CPU 查核當成 GPU 效益重現。"}
output["result"] = "所有斷言通過"
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
