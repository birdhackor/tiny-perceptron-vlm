"""19.18 獨立審閱的透明算例；不執行模型、不代表產品實測。"""

import hashlib
import itertools
import json
import platform
from pathlib import Path

import torch

from scripts.reading_time import lesson_slices


ROOT = Path(__file__).resolve().parents[4]
body = lesson_slices((ROOT / "course/chapters/19.md").read_text())["19.18"]
freeze = json.loads((ROOT / "docs/course-revision-20261016/freeze-08.json").read_text())
entry = next(page for page in freeze["pages"] if page["page_id"] == "19.18")
body_hash = hashlib.sha256(body.encode()).hexdigest()
assert body_hash == entry["source_sha256"]
assert body.encode() == (ROOT / entry["snapshot"]).read_bytes()

# 人工固定紀錄：high 並非逐題支配 medium；最後一題全部檔都錯。
truth = [0, 1, 2, 3, 4]
predictions = {
    "low": [0, 0, 0, 0, 0],
    "medium": [0, 1, 2, 0, 0],
    "high": [0, 1, 0, 3, 0],
}
rounds = {"low": 1, "medium": 3, "high": 5}
levels = tuple(predictions)
correct = {
    level: [int(a == y) for a, y in zip(predictions[level], truth, strict=True)]
    for level in levels
}
oracle_levels = [
    min(
        (level for level in levels if correct[level][i]),
        key=rounds.get,
        default="low",  # 全錯仍算錯；此時只選最便宜的失敗。
    )
    for i in range(len(truth))
]
oracle_count = sum(correct[level][i] for i, level in enumerate(oracle_levels))
policies = []
for choices in itertools.product(levels, repeat=len(truth)):
    count = sum(correct[level][i] for i, level in enumerate(choices))
    total_rounds = sum(rounds[level] for level in choices)
    assert count <= oracle_count
    policies.append((count, total_rounds))
minimum_rounds_at_max_accuracy = min(cost for count, cost in policies if count == oracle_count)
assert minimum_rounds_at_max_accuracy == sum(rounds[level] for level in oracle_levels)
assert correct["medium"][2] == 1 and correct["high"][2] == 0
assert sum(correct[level][-1] for level in levels) == 0

# 假設串行成本：省一輪 5ms，額外 C 花 8ms，總時間反而增加 3ms。
time_per_round_ms = 5
fixed_rounds, auto_rounds, controller_ms = 3, 2, 8
fixed_ms = fixed_rounds * time_per_round_ms
auto_ms = auto_rounds * time_per_round_ms + controller_ms
assert auto_rounds < fixed_rounds and auto_ms > fixed_ms

# 假設獨立連續延遲，真 p99 為 q，P(所有 n 次皆低於 q) = 0.99**n。
# n=10 時仍有約 90.4% 機會連樣本最大值都低於真 p99。
n, p = 10, 0.99
max_below_true_p99_probability = p**n
assert 0.90 < max_below_true_p99_probability < 0.91

# 只核對 A 歷史報告的任務、版本與分母；不重跑歷史模型。
history_path = ROOT / "docs/selftrained/results/v2-final-public-results.json"
history = json.loads(history_path.read_text())
moe = history["architectures"]["moe"]
tasks = moe["final_test"]["per_task_final_reply"]
historical_subset = {
    task: {
        "question_count": tasks[task]["count"],
        "source_groups": tasks[task]["source_groups"],
        "image_assets": tasks[task]["image_assets"],
        "recording_assets": tasks[task]["recording_assets"],
        "semantic": tasks[task]["semantic"],
    }
    for task in ["vision_clothing", "voice_qa"]
}

result = {
    "purpose": "獨立審閱的人工固定紀錄與數學核對；沒有模型訓練或GPU量測",
    "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/revision-20261016/19.18-contract-check.py",
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "CPU；cuda available=" + str(torch.cuda.is_available()),
    },
    "source_freeze": {"sha256": body_hash, "snapshot": entry["snapshot"], "exact_match": True},
    "synthetic_record": {"truth": truth, "predictions": predictions, "rounds": rounds},
    "oracle": {
        "chosen_levels": oracle_levels,
        "correct": oracle_count,
        "question_denominator": len(truth),
        "enumerated_policies": len(policies),
        "minimum_total_rounds_at_max_accuracy": minimum_rounds_at_max_accuracy,
        "all_wrong_question_remains_wrong": True,
        "search_cost_included": False,
        "scope": "只對此固定紀錄成立；非同預算保證、非可部署策略",
    },
    "controller_cost_derivation": {
        "assumed_ms_per_B_round": time_per_round_ms,
        "fixed_rounds": fixed_rounds,
        "auto_rounds": auto_rounds,
        "assumed_C_ms": controller_ms,
        "fixed_total_ms": fixed_ms,
        "auto_total_ms": auto_ms,
        "scope": "代入假設值的成本算例，非實測延遲",
    },
    "small_sample_tail_derivation": {
        "assumption": "n 個 iid 連續樣本，F(q)=p",
        "formula": "P(max(X_1,...,X_n) < q) = F(q)^n = p^n",
        "n": n,
        "p": p,
        "probability": max_below_true_p99_probability,
        "scope": "說明少量樣本最大值不能可靠代表真p99，不是產品延遲估計",
    },
    "historical_A_inspection": {
        "path": history_path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "hf_revision": history["hf_revision"],
        "training_revision": moe["training_revision"],
        "evaluation_revision": moe["final_test"]["revision"],
        "final_test_count": moe["final_test"]["count"],
        "tasks": historical_subset,
        "data_scope": history["data_scope"],
        "comparison_limit": history["comparison_limit"],
        "scope": "讀原始凍結報告；沒有聲稱重跑。B/C仍是課文預告契約，未要求不存在的結果。",
    },
}
print(json.dumps(result, ensure_ascii=False, indent=2))
