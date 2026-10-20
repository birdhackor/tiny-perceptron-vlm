"""核對偏好資料的對齊、UTF-8 前綴與固定資料切分，不執行訓練。"""

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import _digest, _utf8_prefix
from tiny_perceptron.alignment import sequence_log_probability
from tiny_perceptron.data import IGNORE, SPECIALS, ByteTokenizer, render_chat
from tiny_perceptron.model import masked_loss


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def render(answer):
    return render_chat(
        [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": answer}]
    )


tok = ByteTokenizer()
assert tok.encode("1+1=?") == [57, 51, 57, 69, 71]
assert [tok.encode(value)[0] for value in ["2", "3", "4"]] == [58, 59, 60]
assert len(SPECIALS) == 8
assert [tok.bos_id, tok.eos_id, tok.user_id, tok.assistant_id] == [1, 2, 3, 4]
expected_prefix = [1, 3, 57, 51, 57, 69, 71, 2, 4]
examples = {}
for answer in ["2", "3", "4", "34"]:
    x, y = render(answer)
    expected_ids = [byte + 8 for byte in answer.encode("utf-8")]
    assert x.tolist() == expected_prefix + expected_ids
    assert y.tolist() == [IGNORE] * 8 + expected_ids + [tok.eos_id]
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    assert len(x) == len(y)
    assert int((y != IGNORE).sum()) == len(expected_ids) + 1
    examples[answer] = {"x": x.tolist(), "y": y.tolist(), "dtype": str(x.dtype)}

# 建立完全指定的條件機率，讓 EOS 的機率隨答案字元改變。
def controlled_logits(x):
    probs = torch.full((1, len(x), tok.vocab_size), 1 / tok.vocab_size, dtype=torch.float64)
    probs[0, 8] = (1 - 0.4) / (tok.vocab_size - 1)
    probs[0, 8, 59] = 0.4
    eos_probability = 0.2 if int(x[-1]) == 59 else 0.9
    probs[0, -1] = (1 - eos_probability) / (tok.vocab_size - 1)
    probs[0, -1, tok.eos_id] = eos_probability
    return probs.log()


chosen_x, _ = render("2")
rejected_x, rejected_y = render("3")
right_logits = controlled_logits(rejected_x)
wrong_logits = controlled_logits(chosen_x)
right_logp = float(sequence_log_probability(right_logits, rejected_y[None]))
wrong_logp = float(sequence_log_probability(wrong_logits, rejected_y[None]))
assert math.isclose(right_logp, math.log(0.4 * 0.2), abs_tol=1e-12)
assert math.isclose(wrong_logp, math.log(0.4 * 0.9), abs_tol=1e-12)
base_loss = float(masked_loss(right_logits, rejected_y[None]))
changed_logits = right_logits.clone()
changed_logits[0, :8, 0] = 1000
assert float(masked_loss(changed_logits, rejected_y[None])) == base_loss
assert math.isclose(base_loss, -right_logp / 2, abs_tol=1e-12)

report = json.loads((ROOT / "docs/course-experiments/results/dpo.json").read_bytes())
metadata = json.loads((ROOT / "assets/training/sources/ultrafeedback-dpo.json").read_bytes())
url = metadata["raw_training_file"]["source_url"]
curl_command = ["curl", "-fL", "--max-time", "45", "-sS", "-D", "-", url]
fetched = subprocess.run(curl_command, capture_output=True, check=True)
headers, separator, body = fetched.stdout.rpartition(b"\r\n\r\n")
assert separator
response = json.loads(body)
expected_response = next(
    item["sha256"]
    for item in report["assets"][0]["files"]
    if item["path"].endswith("/source-viewer-response.json")
)
assert sha(body) == expected_response
rows = [item["row"] for item in response["rows"]]
assert len(rows) == 100
assert not any(item["truncated_cells"] for item in response["rows"])
records = []
lengths = {field: [] for field in ["prompt", "chosen", "rejected"]}
cropped = {field: 0 for field in lengths}
for index, row in enumerate(rows):
    answers = []
    for side in ["chosen", "rejected"]:
        original = next(message["content"] for message in reversed(row[side]) if message["role"] == "assistant")
        assert row[side][0]["role"] == "user"
        assert row[side][0]["content"] == row["prompt"]
        prefix = _utf8_prefix(original, 120)
        assert original.startswith(prefix) and len(prefix.encode("utf-8")) <= 120
        answers.append(prefix)
        lengths[side].append(len(prefix.encode("utf-8")))
        cropped[side] += int(prefix != original)
    prompt = _utf8_prefix(row["prompt"], 120)
    lengths["prompt"].append(len(prompt.encode("utf-8")))
    cropped["prompt"] += int(prompt != row["prompt"])
    assert row["prompt"].startswith(prompt) and len(prompt.encode("utf-8")) <= 120
    records.append(
        {
            "family": row.get("prompt_id", _digest(row["prompt"])),
            "prompt": prompt,
            "chosen": answers[0],
            "rejected": answers[1],
            "source_row": index,
            "source_record_sha256": _digest(row),
        }
    )
parts = split_records(records, seed=report["seed"])
split_checks = {}
family_sets = []
for name, values in parts.items():
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values).encode("utf-8")
    actual = {"records": len(values), "families": len({row["family"] for row in values}), "sha256": sha(serialized)}
    expected = report["results"]["ultrafeedback_pilot"]["data"][name]
    assert all(actual[key] == expected[key] for key in actual)
    split_checks[name] = actual
    family_sets.append({row["family"] for row in values})
assert all(not left & right for i, left in enumerate(family_sets) for right in family_sets[i + 1 :])
sequence_lengths = [
    len(
        render_chat(
            [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row[side]}]
        )[0]
    )
    for row in records
    for side in ["chosen", "rejected"]
]
assert max(sequence_lengths) <= 244 < 256

text = "甲" * 39 + "乙丙"
assert _utf8_prefix(text, 118) == "甲" * 39
assert _utf8_prefix(text, 119) == "甲" * 39
assert _utf8_prefix(text, 120) == "甲" * 39 + "乙"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python "
    "docs/technical-reviews/artifacts/review-20261019-13-2/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "repository_inputs_sha256": {
        path: sha((ROOT / path).read_bytes())
        for path in [
            "tiny_perceptron/data.py",
            "tiny_perceptron/alignment.py",
            "tiny_perceptron/model.py",
            "scripts/course_experiments/behavior.py",
            "scripts/course_experiments/text.py",
            "scripts/course_experiments/common.py",
            "assets/training/sources/ultrafeedback-dpo.json",
            "docs/course-experiments/results/dpo.json",
        ]
    },
    "special_tokens": dict(enumerate(SPECIALS)),
    "examples": examples,
    "context_probability_check": {
        "right_context_probability": math.exp(right_logp),
        "crossed_context_probability": math.exp(wrong_logp),
        "right_log_probability": right_logp,
        "crossed_log_probability": wrong_logp,
        "valid_targets": 2,
        "loss": base_loss,
        "masked_positions_do_not_change_loss": True,
        "scope": "完全指定的機率算例，支持上下文不可交叉使用；沒有測量訓練品質。",
    },
    "dataset": {
        "url": url,
        "revision": metadata["revision"],
        "curl_command": curl_command,
        "response_sha256": sha(body),
        "http_headers": headers.decode("utf-8"),
        "rows": len(rows),
        "truncated_cells": 0,
        "official_source_split": "train_prefs",
        "seed": report["seed"],
        "splits": split_checks,
        "families_disjoint": True,
        "maximum_utf8_bytes": {field: max(values) for field, values in lengths.items()},
        "cropped_fields": cropped,
        "multibyte_boundary_checks": True,
        "maximum_rendered_input_tokens": max(sequence_lengths),
        "rendered_sequences": len(sequence_lengths),
        "raw_third_party_records_saved": False,
    },
    "result": "全部斷言通過；未執行訓練。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
