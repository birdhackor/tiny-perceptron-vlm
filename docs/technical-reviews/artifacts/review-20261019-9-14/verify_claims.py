"""保存透明算例、固定候選的溫度縮放與教材連結核對結果。"""

import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent


def section(path, identifier):
    raw = path.read_bytes()
    match = re.search(
        rb"^## " + re.escape(identifier.encode()) + rb" .*?(?=^## |\Z)",
        raw,
        re.M | re.S,
    )
    assert match is not None, "找不到指定標題"
    return match.group()


def close(actual, expected, tolerance):
    assert abs(actual - expected) <= tolerance, "計算結果超出指定容忍差"


torch.set_num_threads(1)
torch.set_default_device("cpu")
y = torch.tensor([1.0, 0.0, 0.0])
probabilities = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [1 / 3, 1 / 3, 1 / 3]]
expected_ce = [-math.log(0.8), -math.log(0.2), math.log(3)]
expected_brier = [0.06, 1.14, 2 / 3]
examples = []
for values, ce_expected, brier_expected in zip(probabilities, expected_ce, expected_brier):
    p = torch.tensor(values)
    ce = float(-p[0].log())
    brier = float(((p - y) ** 2).sum())
    close(ce, ce_expected, 2e-7)
    close(brier, brier_expected, 2e-7)
    examples.append(
        {
            "p": values,
            "target": y.tolist(),
            "dtype": str(p.dtype),
            "shape": list(p.shape),
            "ce": ce,
            "brier_sum": brier,
            "independent_ce_float64": ce_expected,
            "independent_brier_float64": brier_expected,
            "tolerance": 2e-7,
        }
    )

p = torch.tensor([0.8, 0.1, 0.1], dtype=torch.float64)
one_hot = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
q = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
hard_ce = float(F.cross_entropy(p.log(), torch.tensor(0)))
one_hot_ce = float(F.cross_entropy(p.log(), one_hot))
soft_ce = float(F.cross_entropy(p.log(), q))
weighted_ce = float(-(q * p.log()).sum())
close(hard_ce, -math.log(0.8), 1e-12)
close(one_hot_ce, hard_ce, 1e-12)
close(soft_ce, weighted_ce, 1e-12)
bce = []
for truth in [0.0, 1.0]:
    answer_probability = torch.tensor(0.8, dtype=torch.float64)
    loss = float(F.binary_cross_entropy(answer_probability, torch.tensor(truth, dtype=torch.float64)))
    expected = -math.log(0.8 if truth else 0.2)
    close(loss, expected, 1e-12)
    bce.append({"p_yes": 0.8, "y_yes": truth, "bce": loss, "expected": expected})

true_distribution = [0.7, 0.3]
expected_scores = []
for forecast in [[0.7, 0.3], [0.9, 0.1], [0.5, 0.5]]:
    ce = -sum(actual * math.log(reported) for actual, reported in zip(true_distribution, forecast))
    brier = sum(
        true_distribution[outcome]
        * sum((forecast[index] - int(index == outcome)) ** 2 for index in range(2))
        for outcome in range(2)
    )
    expected_scores.append({"forecast": forecast, "expected_ce": ce, "expected_brier_sum": brier})
assert all(row["expected_ce"] > expected_scores[0]["expected_ce"] for row in expected_scores[1:])
assert all(row["expected_brier_sum"] > expected_scores[0]["expected_brier_sum"] for row in expected_scores[1:])

logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], dtype=torch.float64)
labels = torch.tensor([0, 1, 1, 2])
temperature_grid = [0.5, 1.0, 2.0, 4.0]
temperature_rows = []
for temperature in temperature_grid:
    scaled = (logits / temperature).softmax(-1)
    assert torch.equal(scaled.argmax(-1), logits.argmax(-1)), "正溫度改變了最高候選"
    temperature_rows.append(
        {
            "temperature": temperature,
            "mean_ce": float(F.cross_entropy(logits / temperature, labels)),
            "predicted_class": scaled.argmax(-1).tolist(),
            "mean_max_probability": float(scaled.max(-1).values.mean()),
        }
    )
chosen = min(temperature_rows, key=lambda row: row["mean_ce"])
tie_logits = torch.tensor([2.0, 2.0, -3.0], dtype=torch.float64)
assert all(int((tie_logits / value).softmax(-1).argmax()) == 0 for value in temperature_grid)

frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/9.14.md"
current = section(ROOT / "course/chapters/09.md", "9.14")
assert current == frozen.read_bytes(), "目前小節與指定凍結原文不同"
links = []
for reference in re.findall(r"\]\(([^)]+)\)", current.decode()):
    path, identifier = reference.split("#")
    target = ROOT / "course/chapters" / path
    heading = section(target, identifier).decode().splitlines()[0]
    links.append({"reference": reference, "target": str(target.relative_to(ROOT)), "heading": heading})
policy_text = (ROOT / "course/chapters/13.md").read_text()
policy_definition = next(line for line in policy_text.splitlines() if "正在更新、之後實際作答的模型叫" in line)
assert "策略模型（policy）" in policy_definition

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-14/verify_claims.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
    "outcome": "全部斷言通過；沒有訓練模型或測量泛化效能。",
    "reported_confidence_example": {"count": 100, "correct": 40, "observed_accuracy": 40 / 100, "reported_probability": 0.8, "gap": 0.8 - 40 / 100},
    "examples": examples,
    "cross_entropy_equivalence": {"hard_ce": hard_ce, "one_hot_ce": one_hot_ce, "soft_target": q.tolist(), "soft_ce": soft_ce, "weighted_ce": weighted_ce, "bce": bce},
    "expected_scoring_rules": {"true_distribution": true_distribution, "rows": expected_scores},
    "temperature_grid": {"logits": logits.tolist(), "labels": labels.tolist(), "rows": temperature_rows, "chosen_temperature": chosen["temperature"], "fixed_argmax": logits.argmax(-1).tolist(), "tie_case": tie_logits.tolist(), "sample_count": len(labels)},
    "subgroup_counterexample": {"confidence": 0.8, "group_A": {"correct": 10, "count": 10, "accuracy": 1.0}, "group_B": {"correct": 6, "count": 10, "accuracy": 0.6}, "combined_accuracy": 16 / 20, "conclusion": "整體校準不能推出各子群校準；這是人工構造，並非模型實測。"},
    "uniform_predictor": {"class_counts": [1, 1, 1], "p_everywhere": [1 / 3] * 3, "argmax_with_fixed_tie_rule": 0, "accuracy": 1 / 3, "top_label_calibration_gap": 0.0, "conclusion": "平衡三類上可同時沒有校準差與沒有辨識能力；不推廣為任意資料分布都已校準。"},
    "derivations": {
        "cross_entropy": "q 為合法分布且 p 在 q 的支撐上為正時，CE(q,p)=H(q)+KL(q||p)；KL>=0，等號僅在 p=q。one-hot 的唯一非零項是 -log(p_correct)；q=(y,1-y) 時得到 BCE。以教師或含錯標籤的 q 作目標，只能保證理想期望代價偏好該 q，不能保證 q 等於事實分布。",
        "brier": "Y~q，B(p,Y)=sum_j(p_j-1[Y=j])^2；取期望得到 sum_j(p_j-q_j)^2 + 1-sum_j(q_j^2)，所以唯一最小點是 p=q。這裡未除候選數。",
        "positive_temperature": "有限 logits 與 0<tau<infinity 下，z_i>z_j 等價於 z_i/tau>z_j/tau；exp 嚴格遞增且 softmax 分母相同，故次序與最大值集合不變。浮點上溢、下溢與非有限值不在此代數論證中。",
        "finite_data": "期望代價在真 q 最小，不等於有限樣本估出的 q 恰好是真 q；模型類別未涵蓋真分布或標籤被改變時，最佳可達解也不必是真分布。",
        "uniform_numbers": "-ln(1/3)=ln(3)=1.0986122886681098；(1/3-1)^2+(1/3)^2+(1/3)^2=2/3=0.6666666666666666。",
    },
    "source_identity": {"current_sha256": hashlib.sha256(current).hexdigest(), "freeze_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(), "svg_references": re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", current.decode())},
    "cross_references": links,
    "policy_model_definition": policy_definition,
}
(ARTIFACT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
