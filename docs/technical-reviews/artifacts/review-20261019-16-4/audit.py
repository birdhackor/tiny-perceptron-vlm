"""核對快取結構、固定實驗紀錄與原始來源；不執行訓練。"""

import ast
import hashlib
import json
import math
import platform
import random
import re
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.architecture import _clone_config
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.attention import CausalAttention
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def function_text(raw, name):
    tree = ast.parse(raw)
    return ast.get_source_segment(raw, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))


def originals():
    specs = [
        ("gqa", "https://arxiv.org/html/2305.13245v3", "arXiv:2305.13245v3, 23 Dec 2023", "Uptraining; Grouped-query attention", "The projection matrices for key and value heads are mean pooled into single projection matrices",
         "讀過 Uptraining 與 Grouped-query attention 全文、圖說及 Limitations：每組 Query 共用一對 K/V；單組等於 MQA；平均投影後額外預訓練，沒有功能恆等承諾。"),
        ("sdpa", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html", "PyTorch 2.14", "API 定義、等價偽碼與 GQA Note", "Memory-efficient attention also supports GQA on NVIDIA CUDA. GQA does not support Nested tensors.",
         "已核對 QK 縮放、softmax、乘 V 的偽碼；GQA 段列出 Flash、cuDNN、math 與 NVIDIA CUDA memory-efficient，要求頭數整除、K/V 頭數相同且不接受 Nested tensors。這是文件契約，沒有在本機驗證 CUDA kernels。"),
        ("sft", "https://arxiv.org/html/2203.02155v1", "arXiv:2203.02155v1, 4 Mar 2022", "Models / Supervised fine-tuning (SFT)", "We fine-tune GPT-3 on our labeler demonstrations using supervised learning.",
         "讀過 Models 的 SFT 段及方法概述；示範資料上的監督式微調支持用示範回答續訓的定義，不支持本課屬性問答以外的能力結論。"),
        ("numel", "https://docs.pytorch.org/docs/2.14/generated/torch.numel.html", "PyTorch 2.14", "torch.numel", "Returns the total number of elements in the input tensor.",
         "已讀函式定義與例子：回傳 tensor 元素總數。"),
        ("element_size", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.element_size.html", "PyTorch 2.14", "Tensor.element_size", "Returns the size in bytes of an individual element.",
         "已讀定義與 float tensor 範例，單個元素以 bytes 計。"),
        ("cross_entropy", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html", "PyTorch 2.14", "class-index target 的 unreduced loss、ignore_index 與 reduction", "the loss is averaged over non-ignored targets.",
         "已讀 class-index 目標公式及 ignore_index：無 class weighting、無 label smoothing 時，每個有效目標取負自然對數機率；用 sum 後除以未忽略目標數得到本課平均。"),
    ]
    result = []
    for key, url, version, locator, excerpt, note in specs:
        process = subprocess.run(["curl", "-fLsS", "--max-time", "30", url], capture_output=True, check=True)
        plain = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", process.stdout.decode())))
        require(excerpt in plain, "原文摘錄必須出現在當次下載內容中")
        result.append({"id": key, "url": url, "version": version, "locator": locator,
                       "accessed_at": datetime.now(UTC).isoformat(), "response_sha256": sha(process.stdout),
                       "response_bytes": len(process.stdout), "command": ["curl", "-fLsS", "--max-time", "30", url],
                       "exit_code": process.returncode, "excerpt": excerpt, "inspection_note": note})
    return result


def run():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    result = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "device": "cpu", "default_dtype": str(torch.get_default_dtype())},
              "scope": "結構及紀錄核算；未重新執行 GPU 訓練、GPU 計時或任何品質評測。"}
    frozen = "docs/course-review-20261019/freeze-01/sources/16.4.md"
    figure = "course/figures/rewrite-16-query-kv-sharing.svg"
    result["inputs"] = {p: sha((ROOT / p).read_bytes()) for p in (
        frozen, figure, "docs/course-review-20261019/freeze-04.json", "docs/course-experiments/results/efficiency.json",
        "docs/course-experiments/results/sft.json", "tiny_perceptron/attention.py", "tiny_perceptron/model.py",
        "tiny_perceptron/data.py", "scripts/course_experiments/architecture.py", "scripts/course_experiments/common.py",
        "scripts/course_experiments/text.py", "scripts/prepare_data.py")}
    result["cache_cases"] = []
    with torch.no_grad():
        for length in (3, 6):
            for kv_heads in (4, 2, 1):
                model = TinyLM(ModelConfig(width=16, heads=4, kv_heads=kv_heads)).eval()
                ids = torch.arange(1, length + 1)[None]
                cache = model(ids)["cache"]
                count = sum(t.numel() * t.element_size() for pair in cache for t in pair)
                expected = 2 * kv_heads * length * 4 * 4
                require(len(cache) == 1 and count == expected, "快取層數與張量數字大小應符合公式")
                require(list(cache[0][0].shape) == [1, kv_heads, length, 4], "K 的軸應為 batch、KV 頭、位置、每頭特徵")
                require(all(t.dtype == torch.float32 for pair in cache for t in pair), "示範快取應使用 FP32")
                next_cache = model(torch.tensor([[7]]), cache=cache)["cache"]
                require(next_cache[0][0].shape[2] == length + 1, "沿用快取後應多保存一個位置")
                result["cache_cases"].append({"length": length, "kv_heads": kv_heads,
                    "k_shape": list(cache[0][0].shape), "v_shape": list(cache[0][1].shape),
                    "bytes": count, "element_bytes": cache[0][0].element_size(),
                    "after_one_position_shape": list(next_cache[0][0].shape)})
    try:
        CausalAttention(16, heads=4, kv_heads=3)
    except ValueError as error:
        result["indivisible_heads"] = {"exception": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError("Query 頭數不能被 KV 頭數整除時應拒絕設定")

    record = read_json("docs/course-experiments/results/efficiency.json")
    base = TinyLM(ModelConfig(width=64, layers=2)).eval()
    ctx = SimpleNamespace(seed=42, device="cpu")
    result["model_structure"] = {}
    for name, kv_heads in (("mha", 4), ("gqa", 1)):
        model, copied = _clone_config(base, ctx, heads=4, kv_heads=kv_heads, experts=0, backend="manual")
        observed = record["results"]["models"][name]
        require(copied == observed["copied_initial_tables"], "可複製參數表應與實驗列舉完全相同")
        count = sum(p.numel() for p in model.parameters())
        require(count == observed["model"]["parameters"], "模型總參數應等於固定紀錄")
        changed = {k: {"base": list(base.state_dict()[k].shape), "target": list(v.shape)}
                   for k, v in model.state_dict().items() if k not in copied}
        require(all(torch.equal(base.state_dict()[k], model.state_dict()[k]) for k in copied), "形狀相同的表應逐值複製")
        with torch.no_grad():
            cache = model(torch.ones(1, 25, dtype=torch.long))["cache"]
            cache_bytes = sum(t.numel() * t.element_size() for pair in cache for t in pair)
        require(cache_bytes == observed["cache"]["prefill_cache_bytes"], "正式模型的 prefill 快取應與公式及紀錄相同")
        result["model_structure"][name] = {"config": model.description()["config"], "parameters": count,
            "copied_count": len(copied), "changed_tables": changed, "cache_bytes": cache_bytes,
            "cache_shape_per_layer": [list(pair[0].shape) for pair in cache]}
        if name == "mha":
            with torch.no_grad():
                sample = torch.tensor([[1, 2, 3]])
                result["single_head_vs_four_heads_logit_max_difference"] = float((base(sample)["logits"] - model(sample)["logits"]).abs().max())

    parts = split_records(generate_records("attributes-sft"), seed=42)
    result["data_reconstruction"] = {}
    tok = ByteTokenizer()
    result["heldout_record_audit"] = {}
    for split, rows in parts.items():
        spec = record["results"]["dataset"][split]
        examples = text_examples(rows, mode="sft")
        tokens = sum(int((y != IGNORE).sum()) for _, y in examples)
        require(records_sha256(rows) == spec["sha256"], "重建資料的指紋應等於固定實驗")
        result["data_reconstruction"][split] = {"records": len(rows), "sha256": records_sha256(rows), "effective_tokens": tokens}
    sampler = random.Random(42)
    train_examples = text_examples(parts["train"], mode="sft")
    training_tokens = sum(sum(int((y != IGNORE).sum()) for _, y in sampler.choices(train_examples, k=16)) for _ in range(100))
    result["reconstructed_training_tokens"] = training_tokens
    for name, model_record in record["results"]["models"].items():
        training = model_record["training"]
        require(training_tokens == training["effective_tokens"], "抽樣後的有效目標數應與固定實驗相同")
        require(training["steps"] == training["optimizer_updates"] == 100 and training["skipped_updates"] == 0, "固定紀錄應完成所有更新")
        model_result = {"training": {k: training[k] for k in ("steps", "optimizer_updates", "effective_tokens", "batch_size", "learning_rate")}, "heldout": {}}
        for split in ("validation", "test"):
            held = model_record["heldout"][split]
            matches, eos = 0, 0
            per_sample = []
            require(len(held["samples"]) == len(parts[split]), "應逐題核對全部留出題")
            for row, sample in zip(parts[split], held["samples"], strict=True):
                ids = sample["generated_ids"]
                raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
                exact = raw == tok.encode(row["messages"][-1]["content"])
                ended = tok.eos_id in ids
                require(sample["messages"] == row["messages"][:-1] and sample["expected"] == row["messages"][-1]["content"], "留出樣本內容應對應重建資料")
                require(exact == sample["exact"] and ended == sample["eos"], "匹配與結束標記應由原始 ID 重算")
                require(len(ids) <= 32, "品質評測生成不得超過設定上限")
                matches += int(exact)
                eos += int(ended)
                per_sample.append({"expected": sample["expected"], "generated_ids": ids, "exact_recomputed": exact, "eos_recomputed": ended})
            tokens = result["data_reconstruction"][split]["effective_tokens"]
            mean = held["nll_sum"] / tokens
            require(tokens == held["effective_tokens"] and mean == held["nll"], "NLL 應使用有效目標數為分母")
            require(matches == held["matches"] and eos == len(parts[split]), "逐題答對數與 EOS 應與紀錄相同")
            model_result["heldout"][split] = {"nll_sum": held["nll_sum"], "effective_tokens": tokens, "nll_recomputed": mean,
                "rounded_nll": f"{mean:.5f}", "matches_recomputed": matches, "records": len(parts[split]), "eos_count": eos, "samples": per_sample}
        timing = model_record["cache"]["cached_decode"]
        median = statistics.median(timing["samples_seconds"])
        require(median == timing["median_seconds"], "計時中位數應由當次九個樣本重算")
        model_result["cached_generation_timing"] = {**timing, "median_ms_recomputed": median * 1000, "rounded_ms": f"{median * 1000:.3f}"}
        result["heldout_record_audit"][name] = model_result

    x, y = text_examples([{"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}], mode="sft")[0]
    require(y[y != IGNORE].tolist() == tok.encode("answer") + [tok.eos_id], "有效目標應只有助手回答及 EOS")
    _, padded_y, _ = pad_batch([(x, y), (x[:-1], y[:-1])])
    require(padded_y[1, -1].item() == IGNORE, "PAD 不應計分")
    logits = torch.zeros(1, 3, 2, dtype=torch.float64)
    labels = torch.tensor([[0, 1, 0]])
    summed, count = loss_sum(logits, labels)
    result["nll_example"] = {"probability": 0.5, "targets": int(count), "per_target": -math.log(0.5),
                              "summed": float(summed), "mean": float(summed / count), "round_four_decimals": f"{summed / count:.4f}"}

    path = "scripts/course_experiments/architecture.py"
    revision = "a5133f444e0dca2347635f74f8ac83a27bb203d4"
    old = subprocess.run(["git", "show", f"{revision}:{path}"], capture_output=True, check=True).stdout
    require(sha(old) == record["code_sha256"][path], "歷史實作必須符合實驗記錄的 SHA")
    current = (ROOT / path).read_text()
    functions = ("_copy_matching", "_clone_config", "_train", "_heldout", "_benchmark", "_fixed_decode", "_cache_probe", "_sdpa_probe", "run_efficiency")
    same = {name: function_text(current, name) == function_text(old.decode(), name) for name in functions}
    require(all(same.values()), "核對實驗的相關函式應與受指紋綁定的歷史實作相同")
    result["historical_implementation"] = {"path": path, "git_revision": revision, "sha256": sha(old), "relevant_functions_identical": same}
    result["record_environment"] = {k: record[k] for k in ("python_version", "torch_version", "device", "gpu", "seed", "step_scale")}
    result["original_sources"] = originals()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
