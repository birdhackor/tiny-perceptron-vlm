"""核對快取算例、模型表形狀及保存實驗的分母與中位數。"""

import hashlib
import html
import json
import math
import platform
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.architecture import _clone_config
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.attention import CausalAttention
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


torch.set_num_threads(1)
torch.manual_seed(42)
torch.set_default_device("cpu")
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-4b/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "cache_examples": [],
}
for length in (3, 6):
    for kv in (4, 2, 1):
        model = TinyLM(ModelConfig(width=16, heads=4, kv_heads=kv)).eval()
        with torch.no_grad():
            cache = model(torch.arange(1, length + 1)[None])["cache"]
        size = sum(t.numel() * t.element_size() for pair in cache for t in pair)
        require(size == 2 * kv * length * 4 * 4, "快取的元素儲存量不符")
        require(tuple(cache[0][0].shape) == (1, kv, length, 4), "K 的軸長不符")
        result["cache_examples"].append({"length": length, "kv_heads": kv, "k_shape": list(cache[0][0].shape), "bytes": size, "dtype": str(cache[0][0].dtype)})
try:
    CausalAttention(width=16, heads=4, kv_heads=3)
except ValueError as error:
    result["invalid_head_count"] = {"exception": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("不可整除的頭數應拒絕")

source = TinyLM(ModelConfig(width=64, layers=2)).eval()
ctx = SimpleNamespace(seed=42, device="cpu")
result["model_tables"] = {}
for name, kv in (("mha", 4), ("gqa", 1)):
    model, copied = _clone_config(source, ctx, heads=4, kv_heads=kv, experts=0, backend="manual")
    original, current = source.state_dict(), model.state_dict()
    changed = {key: {"source": list(original[key].shape), "target": list(value.shape)} for key, value in current.items() if key not in copied}
    require(all(torch.equal(current[key], original[key]) for key in copied), "複製表的數字不同")
    model2, copied2 = _clone_config(source, ctx, heads=4, kv_heads=kv, experts=0, backend="manual")
    require(copied == copied2 and all(torch.equal(value, model2.state_dict()[key]) for key, value in current.items()), "固定種子初始化不一致")
    with torch.no_grad():
        ids = torch.arange(1, 26)[None]
        output = model(ids)
        cache = output["cache"]
        next_output = model(torch.tensor([[26]]), cache=cache)
        reference = model(torch.arange(1, 27)[None])
        cache_error = float((next_output["logits"][:, -1] - reference["logits"][:, -1]).abs().max())
    result["model_tables"][name] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "copied_initial_tables": copied,
        "unmatched_tables": changed,
        "prefill_cache_bytes": sum(t.numel() * t.element_size() for pair in cache for t in pair),
        "prefill_k_shape": list(cache[0][0].shape),
        "decode_k_shape": list(next_output["cache"][0][0].shape),
        "cached_logit_max_error": cache_error,
    }
    require(cache_error < 1e-5, "快取與重算的浮點差過大")
with torch.no_grad():
    four, _ = _clone_config(source, ctx, heads=4, kv_heads=4, backend="manual")
    ids = torch.tensor([[1, 2, 3]])
    result["head_partition_logit_difference"] = float((source(ids)["logits"] - four(ids)["logits"]).abs().max())
require(result["head_partition_logit_difference"] > 1e-5, "切成不同頭數後未觀察到計算改變")

tok = ByteTokenizer()
parts = split_records(generate_records("attributes-sft"), seed=42)
examples = text_examples(parts["train"], mode="sft")
sampler = random.Random(42)
sampled_tokens = sum(sum(int((y != IGNORE).sum()) for _, y in sampler.choices(examples, k=16)) for _ in range(100))
result["dataset_reconstruction"] = {
    name: {"records": len(rows), "sha256": records_sha256(rows), "effective_tokens": sum(int((y != IGNORE).sum()) for _, y in text_examples(rows, mode="sft"))}
    for name, rows in parts.items()
}
result["sampled_training_targets"] = sampled_tokens
require(sampled_tokens == 11278, "訓練有效目標數不符")

report_path = ROOT / "docs/course-experiments/results/efficiency.json"
report = json.loads(report_path.read_bytes())
result["saved_experiment"] = {
    "path": report_path.relative_to(ROOT).as_posix(), "sha256": sha(report_path),
    "environment": {key: report[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu")},
    "models": {},
}
for name, saved in report["results"]["models"].items():
    own = result["model_tables"][name]
    require(own["parameters"] == saved["model"]["parameters"], "總參數數量不符")
    require(own["copied_initial_tables"] == saved["copied_initial_tables"], "複製的表清單不符")
    require(own["prefill_cache_bytes"] == saved["cache"]["prefill_cache_bytes"], "正式形狀的快取 bytes 不符")
    own_saved = {"heldout": {}, "training_steps": saved["training"]["steps"], "training_targets": saved["training"]["effective_tokens"]}
    for split, held in saved["heldout"].items():
        samples = held["samples"]
        matches = sum(sample["generated_ids"][:-1] == tok.encode(sample["expected"]) for sample in samples)
        eos = sum(sample["generated_ids"][-1] == tok.eos_id for sample in samples)
        targets = sum(len(tok.encode(sample["expected"])) + 1 for sample in samples)
        nll = held["nll_sum"] / targets
        require(matches == held["matches"] and eos == len(samples), "完整答案或 EOS 統計不符")
        require(targets == held["effective_tokens"] and math.isclose(nll, held["nll"], abs_tol=1e-14), "平均代價的分母不符")
        require(result["dataset_reconstruction"][split]["sha256"] == report["results"]["dataset"][split]["sha256"], "重建切分雜湊不符")
        own_saved["heldout"][split] = {"matches": matches, "records": len(samples), "eos": eos, "effective_tokens": targets, "nll": nll, "rounded_nll": f"{nll:.5f}"}
    timing = saved["cache"]["cached_decode"]
    median = statistics.median(timing["samples_seconds"])
    require(median == timing["median_seconds"], "計時中位數不符")
    own_saved["timing"] = {"prompt_tokens": saved["cache"]["prompt_tokens"], "generated_tokens": saved["cache"]["generated_tokens"], "warmup_calls": timing["warmup_calls"], "measured_calls": len(timing["samples_seconds"]), "samples_seconds": timing["samples_seconds"], "median_ms": median * 1000}
    result["saved_experiment"]["models"][name] = own_saved
for name, item in result["dataset_reconstruction"].items():
    require(item["sha256"] == report["results"]["dataset"][name]["sha256"], "資料切分不同")
result["parameter_reduction"] = 4 * (64 * 64 - 16 * 64)
result["half_probability_cost"] = -math.log(0.5)
logits = torch.zeros(1, 5, 2, dtype=torch.float64)
labels = torch.tensor([[0, 1, 0, IGNORE, IGNORE]])
total, count = loss_sum(logits, labels)
result["masked_loss_example"] = {"effective_tokens": int(count), "mean": float(total / count)}
require(int(count) == 3 and abs(float(total / count) + math.log(0.5)) < 1e-14, "有效目標平均不符")

result["primary_sources"] = []
specs = [
    ("gqa", "https://arxiv.org/html/2305.13245v3", "arXiv:2305.13245v3, 2023-12-23", "Method / Uptraining 與 Grouped-query attention", "The projection matrices for key and value heads are mean pooled into single projection matrices", "核對分組共享、MQA 與 MHA 端點、快取縮減，以及投影取平均後追加訓練；論文實驗使用 T5、TPU，不能轉借為本專案品質或 GPU 效益。"),
    ("sdpa", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html", "PyTorch 2.14", "GQA note、Constraints for GQA 及參考演算法", "Memory-efficient attention also supports GQA on NVIDIA CUDA.", "核對 CUDA 上的 FlashAttention、cuDNN、math 與 NVIDIA memory-efficient 支援；GQA 仍為實驗功能，Q 頭數須可被 KV 頭數整除、K 與 V 頭數相等，不支援 Nested tensor；本地手寫路徑不據此宣稱後端驗證。"),
    ("cross_entropy", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html", "PyTorch 2.14", "class-index target 的 unreduced loss 與 mean/sum 定義", "this case is equivalent to applying LogSoftmax on an input, followed by NLLLoss", "無類別權重、無 label smoothing 時，單項為正確類別 softmax 機率的負自然對數；ignore_index 排除分子與平均分母。"),
    ("element_size", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.element_size.html", "PyTorch 2.14", "Tensor.element_size", "Returns the size in bytes of an individual element.", "API 回傳每元素 bytes，FP32 範例回傳 4；不包含配置器或 tensor 物件開銷。"),
    ("numel", "https://docs.pytorch.org/docs/2.14/generated/torch.numel.html", "PyTorch 2.14", "torch.numel", "Returns the total number of elements in the input tensor.", "API 計 tensor 元素個數；乘上 element_size 得到本文比較的元素儲存量。"),
]
for key, url, version, locator, excerpt, note in specs:
    command = ["curl", "-fsSL", url]
    fetched = subprocess.run(command, capture_output=True, check=True)
    raw = fetched.stdout
    plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw.decode())))
    require(excerpt in plain, "原始來源找不到保存摘錄")
    result["primary_sources"].append({"id": key, "url": url, "version": version, "accessed_on": "2026-10-20", "command_argv": command, "returncode": fetched.returncode, "response_sha256": hashlib.sha256(raw).hexdigest(), "locator": locator, "excerpt": excerpt, "inspection_note": note})
result["repository_inputs"] = {name: sha(ROOT / name) for name in ("tiny_perceptron/attention.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py", "scripts/course_experiments/architecture.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "scripts/prepare_data.py")}
result["scope"] = "本次只在 CPU 實跑結構、算例與實驗紀錄重算，未重新訓練或重新量測 L4；外部來源全文僅在記憶體讀取，保存短摘錄及雜湊。"
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
