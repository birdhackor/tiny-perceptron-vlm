"""離線核對保存的驗證紀錄、整數配分與比較條件。"""

import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review"
seen = {}


def read(path):
    path = ROOT / path
    raw = path.read_bytes()
    seen[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def require(value, message):
    if not value:
        raise AssertionError(message)


spec = importlib.util.spec_from_file_location("scorer", ROOT / "scripts/score_natural_v4_validation.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
manifest = read("docs/natural-assistant/v4/manifest.json")
protocol = read("docs/natural-assistant/v4/validation-protocol-lower-lr.json")
selection = read("docs/natural-assistant/v4/selection.json")
saved = read(BASE / "scored/scores.json")
result = read(BASE / "raw/result.json")
private = read(BASE / "private-map.json")
grades = read(BASE / "combined/grades.json")
read(BASE / "blind-packet.json")
transcripts = read(BASE / "raw/transcripts.json")
train = read("docs/natural-assistant/evidence/v4-runtime/train-37217452291/training-audit-summary.json")
train_run = read("docs/natural-assistant/evidence/v4-runtime/train-37217452291/github-run.json")
index = read("docs/natural-assistant/evidence/v4-runtime/train-37217452291/actual-artifact-index.json")
cases, audio = scorer.expected_cases(manifest, protocol)
records = {}
for variant in protocol["candidate_order"]:
    items = read(BASE / f"raw/generations-{variant}.json")
    records[variant] = {scorer.case_id(item["id"], item["task"]): item for item in items}
    require(len(items) == len(records[variant]) == 132, "生成題數與唯一性不符")
    require(set(records[variant]) == set(cases), "生成題目與固定題單不符")

binding = private["artifact_binding"]
require(binding == saved["artifact_binding"], "得分的來源綁定不符")
for field, path in [
    ("manifest_sha256", "docs/natural-assistant/v4/manifest.json"),
    ("protocol_sha256", "docs/natural-assistant/v4/validation-protocol-lower-lr.json"),
    ("validation_result_sha256", (BASE / "raw/result.json").relative_to(ROOT).as_posix()),
    ("transcripts_sha256", (BASE / "raw/transcripts.json").relative_to(ROOT).as_posix()),
]:
    require(binding[field] == seen[path], "來源綁定的 SHA 不符")
for item in binding["generation_files"]:
    require(item["sha256"] == seen[(BASE / "raw" / item["name"]).relative_to(ROOT).as_posix()], "生成 SHA 不符")
require(saved["manual_grades_sha256"] == seen[(BASE / "combined/grades.json").relative_to(ROOT).as_posix()], "人工評分 SHA 不符")
require(saved["blind_packet_sha256"] == seen[(BASE / "blind-packet.json").relative_to(ROOT).as_posix()], "盲測題包 SHA 不符")
replayed, replay_selection = scorer.score_blind(
    (manifest, protocol, cases, records, binding, ROOT), BASE, BASE / "combined/grades.json"
)
require(replayed == saved, "重算得分與保存版本不同")
require(replay_selection["selected_variant"] == selection["selected_variant"] == "base", "選版結果不同")

transcript_index = {item["id"]: item for item in transcripts}
shared_fields = ["user", "image", "input_tokens", "image_grid_thw", "reference_answer"]
for key, case in cases.items():
    row = case["row"]
    reference = records["base"][key]
    for variant, mapping in records.items():
        record = mapping[key]
        require(record["variant"] == variant and record["split"] == "validation", "生成身分不符")
        require(all(record.get(field) == reference.get(field) for field in shared_fields), "候選的輸入或處理結果不同")
        user = transcript_index[row["id"]]["transcript"] if case["group"] == "voice_actual_asr_chat" else row["user"]
        require(record["user"] == user and record.get("image") == row.get("image"), "生成輸入不符固定題目")
        if case["group"] == "voice_typed_reference_chat":
            require(record["input_mode"] == "typed reference transcript", "正確轉寫路徑不符")

group_names = list(scorer.WEIGHTS)
denominators = [protocol["validation_denominators"][name] for name in group_names]
group_factors = [Fraction(1, 10), Fraction(1, 10), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 15), Fraction(1, 15), Fraction(1, 15)]
derived_weights = [75600 * factor / count for factor, count in zip(group_factors, denominators)]
require(derived_weights == list(scorer.WEIGHTS.values()), "五組平均與整數配分不等價")
require(sum(denominators) == 132 and sum(denominators[:2]) == 84, "總分母不同")
summary = {}
for variant, mapping in records.items():
    counts = list(saved["variants"][variant]["correct_counts"].values())
    weighted = sum(Fraction(count) * factor / total for count, factor, total in zip(counts, group_factors, denominators))
    numerator = sum(count * weight for count, weight in zip(counts, scorer.WEIGHTS.values()))
    require(weighted == Fraction(numerator, 75600), "分數推導不符")
    incomplete = [record for record in mapping.values() if not scorer.eos_complete(record, 384)]
    require(all(record["generated_tokens"] == len(record["generated_token_ids"]) == 384 and record["generated_token_ids"][-1] not in record["eos_token_ids"] and record["truncated"] for record in incomplete), "不完整回答不是用滿上限且無 EOS")
    summary[variant] = {
        "counts": counts,
        "numerator": numerator,
        "denominator": 75600,
        "percent": round(100 * float(weighted), 2),
        "equal_question_score": f"{sum(counts)}/132",
        "incomplete_count": len(incomplete),
        "incomplete_token_lengths": [record["generated_tokens"] for record in incomplete],
        "gate_results": saved["variants"][variant].get("gates"),
        "group_scores": {
            "photo": str((Fraction(counts[0], 28) + Fraction(counts[1], 56)) / 2),
            "chat": str((Fraction(counts[5], 9) + Fraction(counts[6], 4) + Fraction(counts[7], 4)) / 3),
            "single_ocr": str(Fraction(counts[3], 10)),
        },
    }
require([x["incomplete_count"] for x in summary.values()] == [6, 3, 3], "不完整回答數不同")
audit = train["independent_audit"]
require(audit["train_run_id"] == result["execution"]["adapter_run_id"], "訓練批次不同")
for archive, adapter in zip(audit["archives"], result["execution"]["adapters"]):
    require(adapter["checkpoint"] == f"step-{archive['step']:06d}" and adapter["adapter_sha256"] == archive["adapter_sha256"], "候選存檔身分不同")
require(protocol["declared_utc"] < train_run["run_started_at"], "門檻宣告未早於本次訓練")
visual = [case["row"] for case in cases.values() if case["group"] in {"photo_summary", "photo_fact"}]
images = Counter(row["image"] for row in visual)
reference_probs = [0.4, 0.45]
example = {"reference_probabilities": reference_probs, "reference_losses": [-math.log(p) for p in reference_probs], "other_token_probabilities": [0.5, 0.46], "greedy_answer_still_wrong": True}
source_paths = ["scripts/score_natural_v4_validation.py", "tiny_perceptron/natural_assistant.py", "docs/natural-assistant/v4/TRAINING.md", "docs/review-tools/section_facts.py"]
for path in source_paths:
    seen[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-8/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(scorer.core.torch.__version__), "device": "cpu", "platform": platform.platform()},
    "checks": "全部斷言通過；以原始生成紀錄與已保存的逐題語義判讀重放正式評分，未重新訓練或推論。",
    "input_sha256": seen,
    "variants": summary,
    "weights": list(scorer.WEIGHTS.values()),
    "minimum_counts": [16, 30, 17, 5, 2, 2, 4, 3],
    "selected_variant": selection["selected_variant"],
    "same_inputs_checked": {"generations_per_variant": 132, "fields": shared_fields, "typed_voice_cases": 4, "actual_asr_voice_cases": 4},
    "config": {key: result[key] for key in ["model", "model_revision", "asr_model", "asr_revision", "min_pixels", "max_pixels", "seed", "device", "dtype", "versions"]},
    "protocol_before_train": {"declared_utc": protocol["declared_utc"], "train_started_at": train_run["run_started_at"], "followup_informed_by_prior_validation": True},
    "training_archives": audit["archives"],
    "asr_transcript_examples": [{"reference": row["reference_transcript"], "actual": row["transcript"]} for row in transcripts if row["task"] == "speech_chat"],
    "photo_dependency": {"questions": len(visual), "distinct_photos": len(images), "questions_per_photo": sorted(set(images.values()))},
    "loss_counterexample": example,
    "scope": "核對保存之實驗紀錄的一致性與算術；不重判照片語義分、不下載私有權重、不重跑 L4 生成或訓練。",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
