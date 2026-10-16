"""5.16 獨立 CPU 有界核對；不載入模型、不訓練、不重現歷史成績。"""

import contextlib
import hashlib
import io
import json
import math
import platform
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.course_experiments.text import arithmetic_records
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import loss_sum, masked_loss

torch.set_num_threads(1)
body = re.search(r"^## 5\.16 .*?(?=^## |\Z)", (ROOT / "course/chapters/05.md").read_text(), re.M | re.S)[0]
frozen = ROOT / "docs/course-repair-20261017/freeze-02/sources/5.16.md"
assert body.encode() == frozen.read_bytes()
namespace = {}
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(re.search(r"```python\n(.*?)```", body, re.S)[1], namespace)
assert namespace["tokens"] == {"短答": 90, "長答": 200}
assert namespace["total"] == 290

def ratio(short_count, long_count, short_length, long_length):
    a, b = short_count * short_length, long_count * long_length
    return {"short_tokens": a, "long_tokens": b, "total": a + b, "short_ratio": a / (a + b), "long_ratio": b / (a + b)}

cases = {
    "main": ratio(90, 10, 1, 20),
    "exercise_equal_lengths": ratio(90, 10, 1, 1),
    "exercise_half_long_records": ratio(90, 5, 1, 20),
    "target_90_10": ratio(180, 1, 1, 20),
}
assert cases["target_90_10"]["long_ratio"] == 0.1
assert cases["exercise_half_long_records"]["total"] == 190

# 明示手工 label 的 3 格問題前綴；長度 1/20 是有效答案位置數，未假裝是文字 tokenizer 的輸出。
examples = []
source_ids = []
for source, n, length in [("short", 90, 1), ("long", 10, 20)]:
    for _ in range(n):
        examples.append((torch.ones(3 + length, dtype=torch.long), torch.tensor([-100] * 3 + [0] * length)))
        source_ids.append(source)
x, y, valid = pad_batch(examples)
effective = y != -100
assert int(effective.sum()) == 290
counts = {source: sum(int(effective[i].sum()) for i, s in enumerate(source_ids) if s == source) for source in ("short", "long")}
assert counts == {"short": 90, "long": 200}
logits = torch.zeros((*y.shape, 2), dtype=torch.float64, requires_grad=True)
total_loss, denominator = loss_sum(logits, y)
mean_loss = masked_loss(logits, y)
assert math.isclose(float(mean_loss.detach()), math.log(2), abs_tol=1e-12)
mean_loss.backward()
assert torch.count_nonzero(logits.grad[~effective]).item() == 0
changed = logits.detach().clone()
changed[~effective] = torch.tensor([999.0, -999.0], dtype=torch.float64)
assert torch.equal(loss_sum(changed, y)[0], total_loss.detach())
raw_losses = torch.nn.functional.cross_entropy(logits.detach().reshape(-1, 2), y.reshape(-1), ignore_index=-100, reduction="none").reshape(y.shape)
short_loss = float(raw_losses[:90].sum())
long_loss = float(raw_losses[90:].sum())
weighted_loss = (short_loss + 0.1 * long_loss) / 290
assert int((y != -100).sum()) == 290
unequal_logits = logits.detach().clone()
unequal_logits[90:, :, 0] = 5.0
unequal_losses = torch.nn.functional.cross_entropy(unequal_logits.reshape(-1, 2), y.reshape(-1), ignore_index=-100, reduction="none").reshape(y.shape)
unequal_long_share = float(unequal_losses[90:].sum() / unequal_losses.sum())
_, cropped_y, _ = pad_batch(examples, max_length=8)
assert int((cropped_y != -100).sum()) == 140

tok = ByteTokenizer()
chat_examples = [render_chat([{"role": "user", "content": "why?"}, {"role": "assistant", "content": answer}], tok) for answer in ["a", "abc"]]
cx, cy, cv = pad_batch(chat_examples)
chat_counts = (cy != -100).sum(dim=1).tolist()
assert chat_counts == [2, 4]  # ASCII 答案 bytes 加 EOS；問題與 PAD 不計。
assert cy[~cv].eq(-100).all()

# 僅重算原始抽樣的 label 數。沒有建立、載入、forward 或更新任何模型。
historical_path = ROOT / "docs/course-experiments/results/sft_ablation.json"
historical = json.loads(historical_path.read_text())
attributes = split_records(generate_records("attributes-sft"), seed=42)
arithmetic = split_records(arithmetic_records(), seed=42)
historical_checks = {}
for name, records in [("b-only", arithmetic["train"]), ("replay", attributes["train"] + arithmetic["train"])]:
    raw = historical["results"]["runs"][name]
    train = raw["training"]
    assert records_sha256(records) == train["records_sha256"]
    encoded = text_examples(records, mode="sft", max_length=128)
    lengths = [int((labels != -100).sum()) for _, labels in encoded]
    sampler = random.Random(42)
    total = sum(sum(sampler.choices(lengths, k=16)) for _ in range(500))
    assert total == train["effective_tokens"]
    evaluations = {task: {split: {key: values[key] for key in ["matches", "records", "effective_tokens", "eos_rate"]} for split, values in raw[task].items()} for task in ["A_attributes", "B_arithmetic"]}
    historical_checks[name] = {"steps_recorded": train["steps"], "batch_size_from_implementation": 16, "sampled_records": 8000, "unique_training_records": len(records), "records_sha256": train["records_sha256"], "effective_tokens_recorded": train["effective_tokens"], "effective_tokens_recounted_without_training": total, "mean_tokens_per_sampled_record": total / 8000, "evaluations_from_original_only": evaluations}

source_hashes = {}
for filename in ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "scripts/prepare_data.py"]:
    digest = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    original = historical["code_sha256"].get(filename)
    if original is not None:
        assert original == digest
    source_hashes[filename] = {"current_sha256": digest, "historical_recorded_sha256": original, "matches_recorded_hash": digest == original if original else "not recorded; generated records hash matched"}

print(json.dumps({
    "scope": "CPU counts, masking, loss/gradient arithmetic, and historical result inspection; no model training or historical quality rerun",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform(), "threads": str(torch.get_num_threads())},
    "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
    "frozen_source_bytes_match": True,
    "lesson_code_stdout": output.getvalue(),
    "ratios": cases,
    "masked_loss_check": {"shape": list(x.shape), "padded_cells": x.numel(), "effective_tokens": int(denominator), "by_source": counts, "ignored_labels": int((~effective).sum()), "nll_sum": float(total_loss.detach()), "mean": float(mean_loss.detach()), "expected_log2": math.log(2), "tolerance": 1e-12, "ignored_gradient_nonzeros": int(torch.count_nonzero(logits.grad[~effective])), "changed_ignored_logits_same_loss": True, "uniform_long_loss_share": long_loss / (short_loss + long_loss), "unequal_error_long_loss_share": unequal_long_share, "custom_source_weight_0_1_long_mean_with_same_token_count": weighted_loss, "truncated_max_length8_effective_tokens": int((cropped_y != -100).sum())},
    "render_chat_check": {"effective_counts": chat_counts, "answers": ["a", "abc"], "convention": "ByteTokenizer ASCII bytes + supervised EOS; user and padded labels -100", "padded_labels_ignored": True},
    "historical_metadata": {key: historical[key] for key in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status"]},
    "historical_result_sha256": hashlib.sha256(historical_path.read_bytes()).hexdigest(),
    "historical_counts_only": historical_checks,
    "historical_token_ratio": 36384 / 18453,
    "historical_limit": historical["results"]["matched_budget"],
    "source_hashes": source_hashes,
    "aggregate_caution_derivation": {"before_source_losses": [1.0, 1.0], "after_source_losses": [0.8, 2.0], "source_position_weights": [0.9, 0.1], "before_overall": 1.0, "after_overall": 0.9 * 0.8 + 0.1 * 2.0, "meaning": "hypothetical arithmetic, not empirical: aggregate may fall while minority loss rises"},
}, ensure_ascii=False, indent=2))
