"""獨立核對算例、模型參數與已保存量測紀錄，僅寫入本目錄。"""

import concurrent.futures
import hashlib
import html
import inspect
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import DenseFFN


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


receipt = json.loads((HERE / "section-run-receipt.json").read_text())
run = Path(receipt["artifact_directory"])
bundle = {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/15.md#15.1" --execute',
    "receipt": receipt,
    "extraction": json.loads((run / "extraction.json").read_text()),
    "environment": json.loads((run / "environment.json").read_text()),
    "execution": json.loads((run / "execution.json").read_text()),
    "stdout": (run / "stdout.txt").read_text(),
    "stderr": (run / "stderr.txt").read_text(),
    "original_fence": (run / "fence-1.py").read_text(),
}
assert bundle["execution"]["exit_code"] == 0
assert bundle["receipt"]["source_sha256"] == "08209b8a0bcb4586de9d24243c264bcf5dba65b6d7705edd7a02fa9b1ea85f87"
save("section-execution.json", bundle)

up = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
down = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
x = torch.tensor([[1.0, 2.0], [2.0, 1.0]])


def ffn(value):
    return torch.relu(value @ up) @ down


toy = {}
for copies in (1, 2, 3):
    value = torch.cat([x] * copies, dim=0)
    result = ffn(value)
    assert result.tolist() == [[4.0, 5.0], [5.0, 4.0]] * copies
    toy[str(copies)] = {"input_shape": list(value.shape), "output": result.tolist(), "parameters": up.numel() + down.numel()}
negative = torch.tensor([[1.0, -2.0], [2.0, 1.0]])
toy["negative"] = {"up": (negative @ up).tolist(), "relu": torch.relu(negative @ up).tolist(), "output": ffn(negative).tolist()}
assert toy["negative"]["up"][0] == [1.0, -2.0, -1.0]
assert toy["negative"]["output"][0] == [1.0, 0.0]
small = DenseFFN(2, hidden=3)
assert small.activation == "gelu" and small.up.bias is not None and small.down.bias is not None
assert sum(p.numel() for p in small.parameters()) == 17
toy["library"] = {"activation": small.activation, "up_weight_shape": list(small.up.weight.shape), "down_weight_shape": list(small.down.weight.shape), "bias_parameters": small.up.bias.numel() + small.down.bias.numel(), "total_parameters": sum(p.numel() for p in small.parameters())}
toy["work"] = {"multiplications_per_row": 2 * 3 + 3 * 2, "multiplications_two_rows": 24, "multiplications_four_rows": 48, "scope": "逐項計數矩陣乘法的純量乘法；沒有量測牆鐘時間或宣稱加速。"}

record_path = ROOT / "docs/course-experiments/results/moe.json"
record = json.loads(record_path.read_text())
dense = record["results"]["variants"]["dense_active_top1"]
config = ModelConfig(**dense["model"]["config"])
model = TinyLM(config)
parameters = {name: p.numel() for name, p in model.named_parameters()}
total = sum(parameters.values())
ffn_counts = [sum(p.numel() for p in block.ffn.parameters()) for block in model.blocks]
assert total == dense["model"]["parameters"] == 141568
assert ffn_counts == [33088, 33088]
assert config.width == 64 and config.layers == 2
assert [block.ffn.up.out_features for block in model.blocks] == [256, 256]
heldout = {}
for split, rounded, records in (("validation", "2.29999", 51), ("test", "2.32093", 52)):
    measured = dense["heldout"][split]
    calculated = measured["nll_sum"] / measured["effective_tokens"]
    assert calculated == measured["nll"] and format(calculated, ".5f") == rounded
    assert measured["records"] == record["results"]["dataset"][split]["records"] == records
    heldout[split] = {key: value for key, value in measured.items() if key != "samples"}
    heldout[split]["recomputed_nll"] = calculated
    heldout[split]["rounded_nll"] = rounded
assert dense["training"]["optimizer_updates"] == dense["training"]["steps"] == 180
assert dense["training"]["skipped_updates"] == 0

repository_paths = [
    "tiny_perceptron/model.py", "tiny_perceptron/modern.py",
    "tiny_perceptron/attention.py", "tiny_perceptron/data.py",
    "scripts/course_experiments/architecture.py", "scripts/course_experiments/common.py",
    "scripts/course_experiments/run.py", "course/training.md", "course/chapters/04.md",
    "course/chapters/15.md", "docs/course-experiments/results/moe.json",
]
repository = {}
for path in repository_paths:
    sha = digest((ROOT / path).read_bytes())
    repository[path] = {"sha256": sha, "matches_experiment_code_hash": sha == record["code_sha256"].get(path) if path in record["code_sha256"] else None}

requests = [
    ("transformer", "https://arxiv.org/html/1706.03762v7", "arXiv:1706.03762v7", "which is applied to each position separately and identically", "原文的 Position-wise Feed-Forward Networks、式 (2)、Embeddings and Softmax，以及 Encoder and Decoder Stacks；同一層的 FFN 對各位置使用相同參數，兩次線性變換之間有 ReLU；token 轉成模型寬度的向量。"),
    ("switch", "https://arxiv.org/html/2101.03961v3", "arXiv:2101.03961v3", "we route to only a single expert", "閱讀 Simplifying Sparse Routing 的 Mixture of Expert Routing 與 Switch Routing，式 (1)–(2) 及架構說明；路由在多個 FFN 專家中為各 token 選 top-k，Switch 選一個。論文的效益數字不拿來證明本專案加速。"),
    ("matmul", "https://docs.pytorch.org/docs/2.14/generated/torch.matmul.html", "PyTorch 2.14", "If both arguments are 2-dimensional, the matrix-matrix product is returned.", "閱讀二維輸入矩陣相乘的契約，並與已安裝 torch.matmul.__doc__ 核對。"),
    ("cat", "https://docs.pytorch.org/docs/2.14/generated/torch.cat.html", "PyTorch 2.14", "Concatenates the given sequence", "閱讀 dim 指定拼接維度及形狀條件，並與已安裝 torch.cat.__doc__ 核對；操作對象只有輸入 tensor。"),
    ("numel", "https://docs.pytorch.org/docs/2.14/generated/torch.numel.html", "PyTorch 2.14", "Returns the total number of elements", "閱讀輸入 tensor 元素總數的契約，並與已安裝 torch.numel.__doc__ 核對；沒有對模型、中間值或輸出自動求和。"),
    ("relu", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.ReLU.html", "PyTorch 2.14", "Applies the rectified linear unit function element-wise", "閱讀 ReLU(x)=max(0,x) 公式，與已安裝 nn.ReLU.__doc__ 及實際 torch.relu 計算核對。"),
    ("linear", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html", "PyTorch 2.14", "Default: True", "閱讀 y=xA^T+b、權重與 bias 形狀、bias 預設 True，以及只改最後一維的形狀契約；與已安裝 nn.Linear.__doc__ 核對。"),
]


def fetch(item):
    identifier, url, version, excerpt, note = item
    command = ["curl", "-fLsS", "--max-time", "40", url]
    process = subprocess.run(command, capture_output=True, check=True)
    raw = process.stdout
    text = html.unescape(re.sub(r"<[^>]*>", " ", raw.decode()))
    text = re.sub(r"\s+", " ", text)
    assert excerpt in text, identifier
    return {"id": identifier, "url": url, "version": version, "accessed_on": datetime.now(timezone.utc).date().isoformat(), "command_argv": command, "exit_code": process.returncode, "response_sha256": digest(raw), "response_bytes": len(raw), "excerpt": excerpt, "inspection_note": note}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    originals = list(pool.map(fetch, requests))
save("original-source-excerpts.json", {"sources": originals, "storage_scope": "只保存支持主張的短摘錄與全文雜湊，不保存第三方全文。"})

out = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-1/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform()},
    "toy": toy,
    "dense": {"config": dense["model"]["config"], "total_parameters": total, "ffn_parameters": ffn_counts, "parameter_breakdown": parameters, "recorded_training": {key: dense["training"][key] for key in ("steps", "optimizer_updates", "skipped_updates", "effective_tokens", "batch_size", "learning_rate", "auxiliary_weight")}, "heldout": heldout},
    "recorded_environment": {key: record[key] for key in ("revision", "seed", "device", "torch_version", "python_version", "gpu", "evidence_status")},
    "recorded_dataset": record["results"]["dataset"],
    "variant_names": list(record["results"]["variants"]),
    "repository_sources": repository,
    "local_documentation": {"matmul": inspect.getdoc(torch.matmul).splitlines()[2], "numel": "輸入 tensor 的元素總數", "linear_bias_default": str(inspect.signature(torch.nn.Linear))},
    "limitations": ["未重新訓練或重新評估 NVIDIA L4 checkpoint；歷史 loss 依保存的實驗紀錄核對，並重算其 nll_sum / effective_tokens 與四捨五入。", "本機 TinyStories archive 為 Git LFS pointer，未還原資料集、重新切分或獨立重數各切分的故事數。", "本次實跑只證實 CPU、float32、小型算例，不代表其他平台的效能或訓練品質。"],
    "result": "算例、練習、目前實作的參數量與紀錄中數值的算術核對均符合；全部官方原文擷取成功。",
}
save("verification.json", out)
print(json.dumps({"result": out["result"], "parameters": total, "ffn_parameters": ffn_counts, "heldout": heldout, "original_sources_read": len(originals)}, ensure_ascii=False))
