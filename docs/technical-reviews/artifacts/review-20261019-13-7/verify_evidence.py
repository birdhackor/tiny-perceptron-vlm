"""核對文字計數、既有實驗紀錄與分數公式；不重新訓練模型。"""

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.behavior import _pair_examples, _preference_parts
from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.alignment import dpo_loss, sequence_log_probability
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def select(rows, prompt):
    return next(row for row in rows if row.get("prompt", row.get("messages", [{}])[0].get("content")) == prompt)


report_path = "docs/course-experiments/results/dpo.json"
record = json.loads((ROOT / report_path).read_bytes())
results = record["results"]
branch = results["format_only"]
tok = ByteTokenizer()
counts = []
for text in ("4", "4。" * 8, "4。" * 2, "4; answer complete"):
    counts.append({"text": text, "utf8_hex": text.encode().hex(), "body_tokens": len(tok.encode(text)), "with_eos": len(tok.encode(text)) + 1})
assert [row["body_tokens"] for row in counts] == [1, 32, 8, 18]
assert [row["with_eos"] for row in counts] == [2, 33, 9, 19]

parts = _preference_parts(split_records(arithmetic_records(), seed=record["seed"]))
split_checks = {}
for split, rows in parts.items():
    pairs = [{**row, "rejected": row["chosen"] + "; answer complete"} for row in rows]
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs).encode()
    actual_hash = hashlib.sha256(encoded).hexdigest()
    assert actual_hash == branch["data"][split]["sha256"]
    split_checks[split] = {"records": len(rows), "families": len({row["family"] for row in rows}), "regenerated_sha256": actual_hash}
    if split not in branch["preference"]:
        continue
    stored = branch["preference"][split]
    for expected, sample, pair in zip(pairs, stored["samples"], _pair_examples(pairs, 128), strict=True):
        for field in ("prompt", "chosen", "rejected", "family"):
            assert expected[field] == sample[field]
        lengths = [int((labels != IGNORE).sum()) for _, labels in pair]
        assert lengths == [sample["chosen_answer_tokens"], sample["rejected_answer_tokens"]]
        for _, labels in pair:
            assert labels[-1].item() == tok.eos_id
        margin = sample["policy_chosen_logp"] - sample["policy_rejected_logp"]
        assert math.isclose(margin, sample["policy_margin"], abs_tol=1e-10)
        assert math.isclose(margin - sample["reference_margin"], sample["relative_margin"], abs_tol=1e-10)
    assert stored["records"] == len(stored["samples"])
    assert stored["chosen_higher_absolute_probability"] == sum(s["policy_margin"] > 0 for s in stored["samples"])
    assert stored["relative_preference_improved"] == sum(s["relative_margin"] > 0 for s in stored["samples"])

test = branch["preference"]["test"]
generation = branch["arithmetic"]["test"]
assert len(test["samples"]) == len(generation["samples"]) == 7
assert sum(s["reference_margin"] > 0 for s in test["samples"]) == 7
assert test["relative_preference_improved"] == test["chosen_higher_absolute_probability"] == 7
assert branch["training"]["steps"] == 200
assert generation["matches"] == sum(s["exact"] for s in generation["samples"]) == 0
for row in generation["samples"]:
    ids = row["generated_ids"]
    raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
    assert tok.decode(raw) == row["generated"]
    assert (raw == tok.encode(row["expected"])) == row["exact"]
assert select(generation["samples"], "2+2=?")["generated"] == "3"

before = select(results["before"]["validation"]["samples"], "0+3=?")
after = select(branch["preference"]["validation"]["samples"], "0+3=?")
assert before["chosen"] == after["chosen"] == "3"
assert abs(before["policy_chosen_logp"] - (-13.30943)) < 0.000005
assert abs(after["policy_chosen_logp"] - (-13.37556)) < 0.000005
assert abs(after["relative_margin"] - 115.77823) < 0.000005
reference_rejected = before["policy_chosen_logp"] - after["reference_margin"]
chosen_change = after["policy_chosen_logp"] - before["policy_chosen_logp"]
rejected_change = after["policy_rejected_logp"] - reference_rejected
assert chosen_change < 0 and rejected_change < chosen_change
assert math.isclose(chosen_change - rejected_change, after["relative_margin"], abs_tol=1e-10)

style_path = "docs/course-experiments/results/style.json"
style = json.loads((ROOT / style_path).read_bytes())
assert style["results"]["content_training"]["checkpoint"] == "content.pt"
assert style["results"]["content_evaluation"]["test"]["matches"] == 0

examples = _pair_examples([{"prompt": "2+2=?", "chosen": "4", "rejected": "4; answer complete"}], 128)[0]
x, labels, valid = pad_batch(list(examples))
logits = torch.zeros((*labels.shape, tok.vocab_size), dtype=torch.float64)
scores = sequence_log_probability(logits, labels)
target_counts = (labels != IGNORE).sum(-1)
expected_scores = -target_counts.to(torch.float64) * math.log(tok.vocab_size)
assert torch.allclose(scores, expected_scores, atol=1e-12, rtol=0)
assert target_counts.tolist() == [2, 19]
pc, pr, rc, rr = [torch.tensor([value], dtype=torch.float64) for value in (-2.0, -6.0, -3.0, -5.0)]
sum_loss = dpo_loss(pc, pr, rc, rr).item()
mean_loss = dpo_loss(pc / 2, pr / 3, rc / 2, rr / 3).item()
assert not math.isclose(sum_loss, mean_loss)

inputs = [report_path, style_path, "tiny_perceptron/data.py", "tiny_perceptron/alignment.py", "scripts/course_experiments/behavior.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "docs/course-review-20261019/freeze-01/sources/13.7.md", "docs/course-review-20261019/freeze-04.json"]
output = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "input_sha256": {path: sha(path) for path in inputs},
    "token_counts": counts,
    "split_regeneration": split_checks,
    "recorded_training_environment": {key: record[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status")},
    "recorded_training": branch["training"],
    "test_metrics_recomputed": {"records": 7, "reference_short_higher": 7, "relative_margin_positive": 7, "policy_short_higher": 7, "free_generation_matches": 0},
    "test_candidates": test["samples"],
    "test_generation": generation["samples"],
    "validation_counterexample": {"before_chosen_logp": before["policy_chosen_logp"], "after": after, "reference_rejected_logp_derived": reference_rejected, "chosen_logp_change": chosen_change, "rejected_logp_change": rejected_change, "relative_margin_recomputed": chosen_change - rejected_change},
    "base_generation": {"checkpoint": "content.pt", "records": style["results"]["content_evaluation"]["test"]["records"], "matches": 0},
    "uniform_logits_check": {"dtype": str(logits.dtype), "shape": list(logits.shape), "effective_counts": target_counts.tolist(), "sequence_scores": scores.tolist(), "sum_loss_example": sum_loss, "mean_loss_example": mean_loss},
    "scope": "重新計數、重建資料、核對既有紀錄與確定性公式；沒有重跑 CUDA 訓練或由權重重新生成回答。",
    "result": "全部斷言通過",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
