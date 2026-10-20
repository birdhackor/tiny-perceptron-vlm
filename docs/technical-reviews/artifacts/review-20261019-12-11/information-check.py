"""核對相同轉寫的資訊限制；所有標籤都由手寫條件提供。"""

import json
import platform
import random
import sys
from fractions import Fraction


examples = [
    {"transcript": "你好", "pitch": "low"},
    {"transcript": "你好", "pitch": "high"},
]

# 固定映射的兩個可能答案都只能答對其中一筆。
deterministic_results = []
for answer in ("low", "high"):
    predictions = [answer for row in examples]
    deterministic_results.append(
        {
            "answer": answer,
            "predictions": predictions,
            "correct": sum(
                prediction == row["pitch"]
                for prediction, row in zip(predictions, examples, strict=True)
            ),
        }
    )

# 抽樣規則的任務輸入只有轉寫；亂數沒有取得音高標籤。
rng = random.Random(4)


def sampled_answer(transcript):
    assert isinstance(transcript, str)
    return rng.choice(("low", "high"))


sampled_predictions = [sampled_answer(row["transcript"]) for row in examples]

# 若兩個音高條件等機率，相同轉寫讓它們共用同一個輸出分布。
# 設回 low 的機率為 p，平均答對率為 (p + 1 - p) / 2。
expected_accuracy = []
for p in (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1)):
    accuracy = (p + (1 - p)) / 2
    expected_accuracy.append({"probability_low": str(p), "accuracy": str(accuracy)})

pitch_contours = [
    {"transcript": "你好", "first_half_hz": 200, "second_half_hz": 400},
    {"transcript": "你好", "first_half_hz": 400, "second_half_hz": 200},
]
contour_answers = [
    row["first_half_hz"] > row["second_half_hz"] for row in pitch_contours
]
assert sampled_predictions == ["low", "high"]
assert all(item["correct"] == 1 for item in deterministic_results)
assert all(item["accuracy"] == "1/2" for item in expected_accuracy)
assert contour_answers == [False, True]

print(
    json.dumps(
        {
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "device": "CPU，Python 標準函式庫",
            },
            "examples": examples,
            "deterministic_results": deterministic_results,
            "sampled_rule": {
                "seed": 4,
                "inputs": [row["transcript"] for row in examples],
                "predictions": sampled_predictions,
                "explanation": "相同任務輸入可抽到不同答案；亂數不帶音高資訊，這一次碰巧答對不能證明可靠辨識。",
            },
            "expected_accuracy": expected_accuracy,
            "pitch_contours": pitch_contours,
            "first_half_higher": contour_answers,
            "lexical_answers": [row["transcript"] for row in examples],
            "scope": "手寫條件與精確分數推導；沒有波形、ASR、語氣分類或模型訓練。二分之一的推導限於兩條件等機率、亂數獨立於音高。",
        },
        ensure_ascii=False,
        indent=2,
    )
)
