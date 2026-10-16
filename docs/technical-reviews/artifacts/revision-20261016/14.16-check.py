"""14.16 的原始程式與透明數值核對；沒有模型訓練或實測效果推論。"""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices

body = lesson_slices((ROOT / "course/chapters/14.md").read_text(encoding="utf-8"))["14.16"]
code = re.search(r"```python\n(.*?)```", body, re.S)[1]
capture = io.StringIO()
namespace = {}
with contextlib.redirect_stdout(capture):
    exec(compile(code, "course/chapters/14.md#14.16", "exec"), namespace)
snippet_output = capture.getvalue()
assert snippet_output == "早輪 0.2 後輪 0.4 懲罰 0.2\n早輪 0.5 後輪 0.4 懲罰 0.0\n"

losses = [1.2, 0.8, 0.6, 0.7, 0.9]
best_round = min(range(len(losses)), key=losses.__getitem__) + 1
assert best_round == 3

# 有限權重的總和目標不等於硬限制。以下是明示的兩個可行候選。
penalty = namespace["penalty"]
soft_candidates = []
for before, after in [(0.2, 0.4), (1.0, 0.9)]:
    base = before + after
    reg = penalty(before, after)
    soft_candidates.append({"before": before, "after": after, "CE_sum": base,
                            "lambda": 1.0, "penalty": reg, "objective": base + reg})
assert soft_candidates[0]["objective"] < soft_candidates[1]["objective"]
assert soft_candidates[0]["penalty"] > 0

# 單一正解類別為 0：CE 下降但 argmax 仍錯。
probabilities = [(0.2, 0.8), (0.4, 0.6)]
ce_examples = [{"probabilities": list(p), "gold": 0, "CE": -math.log(p[0]),
                "argmax": max(range(len(p)), key=p.__getitem__)} for p in probabilities]
assert ce_examples[1]["CE"] < ce_examples[0]["CE"]
assert all(row["argmax"] == 1 for row in ce_examples)

# 兩題、正解皆為 0；平均 CE 與答案正確率可排出不同輪數。
metrics = []
for round_number, gold_probs in [(3, [0.99, 0.49]), (4, [0.51, 0.51])]:
    metrics.append({"round": round_number, "gold_probabilities": gold_probs,
                    "mean_CE": sum(-math.log(p) for p in gold_probs) / len(gold_probs),
                    "accuracy": sum(p > 0.5 for p in gold_probs) / len(gold_probs),
                    "denominator": len(gold_probs)})
assert metrics[0]["mean_CE"] < metrics[1]["mean_CE"]
assert metrics[0]["accuracy"] < metrics[1]["accuracy"]

# h -> h/2 為收縮，固定點 0；常數 decoder (0.25, 0.75) 仍對 gold=0 答錯。
states = [2.0 ** (-r) for r in range(6)]
fixed_decoder = [0.25, 0.75]
assert all(math.isclose(states[i + 1], states[i] / 2) for i in range(5))
assert max(range(2), key=fixed_decoder.__getitem__) != 0

result = {
    "lesson_id": "14.16",
    "source_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    "environment": {"python": platform.python_version(), "platform": platform.platform(),
                    "device": "CPU; builtins/math only; no torch, GPU, model or training"},
    "original_snippet": code,
    "original_snippet_stdout": snippet_output,
    "heldout_curve": {"losses": losses, "best_round": best_round, "minimum_loss": min(losses)},
    "soft_objective_candidates": soft_candidates,
    "CE_and_argmax": ce_examples,
    "loss_and_accuracy_rankings": metrics,
    "fixed_point_counterexample": {"update": "h_next=h/2", "states": states,
                                    "fixed_point": 0.0, "decoder_probabilities_at_every_state": fixed_decoder,
                                    "gold": 0, "predicted_class": 1},
    "result": "All assertions passed; constructed examples only; no empirical model claim."
}
print(json.dumps(result, ensure_ascii=False, indent=2))
