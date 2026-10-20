"""重算有效目標、遮罩與固定抽樣紀錄；不建立模型或執行訓練。"""

import hashlib
import inspect
import json
import platform
import random
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.course_experiments.common import Context, records_sha256, split_records, text_examples
from scripts.course_experiments.run import experiment_spec
from scripts.course_experiments.text import arithmetic_records
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import loss_sum, masked_loss


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def mixture(short_count, long_count, long_length):
    counts = {"短答": short_count, "長答": long_count}
    lengths = {"短答": 1, "長答": long_length}
    tokens = {source: counts[source] * lengths[source] for source in counts}
    total = sum(tokens.values())
    return {"counts": counts, "lengths": lengths, "tokens": tokens, "total": total,
            "ratios": {source: value / total for source, value in tokens.items()}}


torch.set_num_threads(1)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-16/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "scope": "重算數量及確定性介面；不重跑 CUDA 訓練、品質、速度或記憶體量測。",
}
result["mixtures"] = [mixture(90, 10, 20), mixture(90, 10, 1), mixture(90, 5, 20), mixture(180, 1, 20)]
check(result["mixtures"][0]["tokens"] == {"短答": 90, "長答": 200}, "基本有效目標數不符")
check(result["mixtures"][1]["ratios"] == {"短答": 0.9, "長答": 0.1}, "等長答案比例不符")
check(result["mixtures"][2]["total"] == 190, "減少長答後的總數不符")
check(Fraction(180, 20) == Fraction(9, 1), "抽樣筆數比例不符")

messages = [
    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
    [{"role": "user", "content": "long question"}, {"role": "assistant", "content": "abc"}],
]
examples = [render_chat(row) for row in messages]
x, y, valid = pad_batch(examples)
tok = ByteTokenizer()
check(int((y != IGNORE).sum()) == 6, "assistant 內容與 EOS 的有效目標數不符")
check(bool((y[~valid] == IGNORE).all()), "padding 的 label 應忽略")
check(bool((x[~valid] == tok.pad_id).all()), "padding 的 input 應是 PAD")
for original, (_, labels) in zip(messages, examples, strict=True):
    expected = tok.encode(original[-1]["content"]) + [tok.eos_id]
    check(labels[labels != IGNORE].tolist() == expected, "問題不應成為有效目標")
logits = torch.zeros((*y.shape, tok.vocab_size), dtype=torch.float64)
total_loss, count = loss_sum(logits, y)
mean_loss = masked_loss(logits, y)
expected_loss = torch.log(torch.tensor(tok.vocab_size, dtype=torch.float64))
check(int(count) == 6, "loss 分母應等於有效目標數")
check(torch.allclose(mean_loss, expected_loss, atol=1e-12, rtol=0), "loss 平均分母不符")
cropped = pad_batch([examples[1]], max_length=len(examples[1][0]) - 1)
result["masking"] = {
    "input_ids": x.tolist(), "labels": y.tolist(), "valid": valid.tolist(),
    "shape": list(y.shape), "dtype": str(y.dtype), "effective_targets": int(count),
    "sum_loss": float(total_loss), "mean_loss": float(mean_loss), "expected_mean": float(expected_loss),
    "after_cropping_one_target": int((cropped[1] != IGNORE).sum()),
    "includes_eos": True,
}

labels = torch.zeros(290, dtype=torch.long)
probabilities = torch.cat((torch.full((90,), 0.1), torch.full((200,), 0.9))).to(torch.float64)
logits = torch.stack((probabilities.log(), (1 - probabilities).log()), dim=-1)
per_target = F.cross_entropy(logits, labels, reduction="none")
short_loss, long_loss = per_target[:90].sum(), per_target[90:].sum()
weights = torch.cat((torch.ones(90), torch.full((200,), 0.1))).to(torch.float64)
result["loss_contribution"] = {
    "target_counts": [90, 200], "target_share_long": 200 / 290,
    "short_loss_sum": float(short_loss), "long_loss_sum": float(long_loss),
    "long_loss_share": float(long_loss / per_target.sum()),
    "weighted_long_share": float((per_target[90:] * weights[90:]).sum() / (per_target * weights).sum()),
    "unchanged_target_count": len(per_target),
    "note": "位置占比不等於代價占比；來源權重不改 label 數量。",
}
result["aggregate_counterexample"] = {
    "source_counts": [90, 10], "before_source_accuracy": [0.5, 0.9],
    "after_source_accuracy": [0.7, 0.1], "before_overall": (90 * 0.5 + 10 * 0.9) / 100,
    "after_overall": (90 * 0.7 + 10 * 0.1) / 100,
    "scope": "透明假設算例，支持總平均可遮住少數來源退步；非訓練結果。",
}

record = json.loads((ROOT / "docs/course-experiments/results/sft_ablation.json").read_text())
attributes = split_records(generate_records("attributes-sft"), seed=42)
arithmetic = split_records(arithmetic_records(), seed=42)
split_checks = {}
for source, parts in (("attributes", attributes), ("arithmetic", arithmetic)):
    for split, rows in parts.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        sha = hashlib.sha256(raw).hexdigest()
        expected = record["results"]["data"][source][split]
        check(sha == expected["sha256"], "生成資料切分與實驗紀錄不符")
        split_checks[f"{source}/{split}"] = {"sha256": sha, "records": len(rows), "matches_record": True}
result["record_split_checks"] = split_checks
result["replayed_sampling"] = {}
for name, rows in (("b-only", arithmetic["train"]), ("replay", attributes["train"] + arithmetic["train"])):
    encoded = text_examples(rows, mode="sft", max_length=128)
    sampler = random.Random(42)
    sampled = [sampler.choices(encoded, k=16) for _ in range(500)]
    total = sum(int((y != IGNORE).sum()) for batch in sampled for _, y in batch)
    training = record["results"]["runs"][name]["training"]
    check(total == training["effective_tokens"], "抽樣累計有效目標數與紀錄不符")
    check(records_sha256(rows) == training["records_sha256"], "訓練資料順序與紀錄不符")
    result["replayed_sampling"][name] = {
        "steps": 500, "batch_size": 16, "sampled_examples": 8000, "seed": 42,
        "training_records": len(rows), "effective_tokens": total,
        "recorded_effective_tokens": training["effective_tokens"],
        "effective_length_distribution": dict(sorted(Counter(int((y != IGNORE).sum()) for _, y in encoded).items())),
        "records_sha256": records_sha256(rows), "matches_record": True,
    }
result["replay_ratio"] = 36384 / 18453

spec = experiment_spec("sft_ablation")
check(spec["dependencies"] == ["sft"], "回放配方依賴不符")
check(experiment_spec("sft")["assets"] == ["ultrachat-sft"], "固定 SFT 的資料包依賴不符")
dependency = Context("cpu", ROOT, ROOT, ROOT).dependency("course/chapters", "05.md")
check(dependency == ROOT / "course/chapters/05.md", "dependency 預期路徑不符")
result["reproduction_dependencies"] = {
    "sft_ablation": spec, "sft_assets": experiment_spec("sft")["assets"],
    "dependency_resolution": str(dependency.relative_to(ROOT)),
    "actual_default_checkpoint": "outputs/course-experiments/course-v1/sft/model.pt",
    "scope": "核對入口與依賴；未啟動完整訓練或下載資料包。",
}
paths = [
    "docs/course-experiments/results/sft_ablation.json", "docs/course-experiments/plan.json",
    "scripts/course_experiments/common.py", "scripts/course_experiments/text.py",
    "scripts/course_experiments/run.py", "scripts/prepare_data.py", "tiny_perceptron/data.py",
    "tiny_perceptron/model.py", "course/training.md",
]
result["repository_sha256"] = {path: digest(path) for path in paths}
result["historical_code_hash_comparison"] = {
    path: {"current": digest(path), "recorded": record["code_sha256"].get(path),
           "identical": digest(path) == record["code_sha256"].get(path)}
    for path in paths if path in record["code_sha256"]
}
result["result"] = "所有數量、遮罩與抽樣核對通過。"
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
