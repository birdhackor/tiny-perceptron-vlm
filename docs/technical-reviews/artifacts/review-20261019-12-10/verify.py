import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]
import sys

sys.path.insert(0, str(ROOT))
from scripts.course_experiments.modalities import _audio_records, _media, _sequence


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


record_path = "docs/course-experiments/results/audio.json"
record = json.loads((ROOT / record_path).read_text())
results = record["results"]
splits = _audio_records()
for split, rows in splits.items():
    assert rows == results["data"]["splits"][split]["records"], "資料生成器與已保存資料不一致"
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
    assert hashlib.sha256(canonical).hexdigest() == results["data"]["splits"][split]["sha256"], "資料指紋不一致"
test = splits["test"]
n = len(test)
assert n == 14, "測試題數與預期不同"
assert [row["answer"] for row in test] == ["low"] * 8 + ["high"] * 6, "類別順序與預期不同"
assert {row["question"] for row in test} == {"pitch?"}, "提問未固定"
for name in ("test", "blank", "shuffle"):
    evaluation = results[name]
    samples = evaluation["samples"]
    assert [s["row"] for s in samples] == list(range(n)), "逐題紀錄不完整"
    assert evaluation["examples"] == n and not evaluation["skipped"], "評估分母不一致"
    assert evaluation["correct"] == sum(s["generated"] == s["target"] for s in samples), "已保存正確題數不一致"
    for s, row in zip(samples, test):
        assert s["target"] == row["answer"], "原題標籤不一致"
        assert s["exact_match"] == (s["generated"] == s["target"]), "逐題正確性不一致"

ctx = SimpleNamespace(device="cpu")
prefixes = [_sequence(row, ctx, answer=False)[0].tolist() for row in test]
assert all(prefix == prefixes[0] for prefix in prefixes), "生成前綴不一致"
silent_lengths = []
for row in test:
    _, waveform = _media(row, ctx)
    silent = torch.zeros_like(waveform)
    assert torch.count_nonzero(silent).item() == 0, "靜音波形仍有非零樣本"
    assert torch.count_nonzero(torch.fft.rfft(silent)).item() == 0, "靜音仍有非零頻譜"
    silent_lengths.append(len(silent))

rows = []
for i, sample in enumerate(results["shuffle"]["samples"]):
    donor = (i + n // 2) % n
    assert sample["donor_row"] == donor, "換入題目不符合指定置換"
    original = results["test"]["samples"][donor]
    assert sample["generated_ids"] == original["generated_ids"], "相同聲音的生成結果不同"
    rows.append({
        "recipient": i + 1,
        "donor": donor + 1,
        "recipient_frequency": test[i]["frequency"],
        "donor_frequency": test[donor]["frequency"],
        "original_target": test[i]["answer"],
        "donor_target": test[donor]["answer"],
        "generated": sample["generated"],
        "original_correct": sample["generated"] == test[i]["answer"],
        "donor_correct": sample["generated"] == test[donor]["answer"],
    })
groups = [rows[:1], rows[1:7], rows[7:8], rows[8:]]
table = [[len(group), sum(r["original_correct"] for r in group), sum(r["donor_correct"] for r in group)] for group in groups]
assert table == [[1, 0, 0], [6, 0, 6], [1, 1, 1], [6, 2, 4]], "彙總表格與預期不同"
original_errors = [i for i, s in enumerate(results["test"]["samples"]) if not s["exact_match"]]
donor_errors = sorted(r["donor"] - 1 for r in rows if not r["donor_correct"])
assert donor_errors == original_errors == [4, 6, 7], "答錯聲音的身分不同"
assert all(s["generated"] == "high" for s in results["blank"]["samples"]), "靜音回答不是固定 high"
assert results["text_only_majority_accuracy"] == 8 / 14, "多數類基準不一致"
families = {name: {r["family"] for r in records} for name, records in splits.items()}
for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
    assert not families[left] & families[right], "資料家族跨越切分"

simple_frequencies = [200, 220, 440, 660]
simple_truth = ["high" if f > 300 else "low" for f in simple_frequencies]
simple_accuracy = sum(a == b for a, b in zip(simple_truth, ["high"] * 4)) / 4
exercise_frequencies = [200] * 9 + [440]
exercise_truth = ["high" if f > 300 else "low" for f in exercise_frequencies]
exercise_accuracy = sum(a == b for a, b in zip(exercise_truth, ["low"] * 10)) / 10
assert simple_truth == ["low", "low", "high", "high"] and simple_accuracy == 0.5, "平衡算例不一致"
assert exercise_accuracy == 0.9, "不平衡算例不一致"
assert ["high" if f > 300 else "low" for f in (200, 310)] == ["low", "high"], "替換算例標籤不一致"

inputs = [record_path, "scripts/course_experiments/modalities.py", "tiny_perceptron/multimodal.py", "docs/course-experiments/README.md", "docs/course-review-20261019/freeze-01/sources/12.10.md", "docs/course-review-20261019/freeze-04.json"]
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-10/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "input_sha256": {path: sha(path) for path in inputs},
    "recorded_code_matches": {path: sha(path) == record["code_sha256"][path] for path in inputs[1:3]},
    "original_run_environment": {k: record[k] for k in ("revision", "seed", "device", "gpu", "torch_version", "python_version", "evidence_status")},
    "training_steps": results["training"]["steps"],
    "history_steps": len(results["training"]["history"]),
    "split_sizes": {name: len(records) for name, records in splits.items()},
    "test_class_counts": dict(Counter(row["answer"] for row in test)),
    "simple_example": {"truth": simple_truth, "accuracy": simple_accuracy},
    "replacement_example": {"frequencies": [200, 310], "truth": ["low", "high"]},
    "imbalanced_exercise": {"examples": 10, "low": 9, "high": 1, "accuracy": exercise_accuracy},
    "majority_baseline": {"correct": 8, "examples": n, "accuracy": 8 / n},
    "original_accuracy": {"correct": results["test"]["correct"], "examples": n},
    "blank_accuracy_against_original": {"correct": results["blank"]["correct"], "examples": n, "generated_counts": dict(Counter(s["generated"] for s in results["blank"]["samples"]))},
    "silent_waveform_lengths": sorted(set(silent_lengths)),
    "silent_spectra_all_zero": True,
    "shuffle_rows": rows,
    "table_count_original_correct_donor_correct": table,
    "shuffle_totals": {"examples": n, "original_correct": sum(r["original_correct"] for r in rows), "donor_correct": sum(r["donor_correct"] for r in rows)},
    "same_target_after_swap": sum(r["original_target"] == r["donor_target"] for r in rows),
    "changed_output_after_swap": sum(r["generated"] != results["test"]["samples"][i]["generated"] for i, r in enumerate(rows)),
    "same_donor_output_all_rows": True,
    "original_error_rows_one_based": [i + 1 for i in original_errors],
    "donor_error_rows_one_based": [i + 1 for i in donor_errors],
    "error_sounds": [test[i] for i in original_errors],
    "question": "pitch?",
    "identical_generation_prefixes": True,
    "disjoint_frequency_families": {k: sorted(v) for k, v in families.items()},
    "all_checks_passed": True,
    "scope": "重算已保存完整實驗的逐題結果，並執行資料與前綴生成器及零波形檢查；未重新訓練或載入實驗權重。",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
