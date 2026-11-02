from math import exp

# 家族、版本、raw logits、正解index；以下都是人工假設值
rows = [
    ("p1", "早", [3.0, 0.0, -1.0], 0),
    ("p1", "晚", [0.0, 3.0, -1.0], 1),
    ("p2", "早", [1.2, 1.0, -1.0], 0),
    ("p2", "晚", [3.0, 0.0, -1.0], 1),
    ("p3", "早", [-1.0, 3.0, 0.0], 0),
    ("p3", "晚", [0.0, 3.0, -1.0], 1),
]
T, tau = 1.25, 0.85


def probabilities(raw_logits):
    scaled = [z / T for z in raw_logits]  # raw只縮放一次
    weights = [exp(z - max(scaled)) for z in scaled]
    return [w / sum(weights) for w in weights]


pairs = {}
accepted = []
correct = 0
for family, variant, logits, gold in rows:
    p = probabilities(logits)
    pred = max(range(len(p)), key=lambda i: p[i])
    ok = pred == gold
    correct += int(ok)
    if family not in pairs:
        pairs[family] = []
    pairs[family].append(ok)
    # 本例假設資訊足夠檢查皆為yes；none不自動接受
    if max(p) >= tau and pred != 2:
        accepted.append(ok)

paired = sum(len(v) == 2 and all(v) for v in pairs.values())
errors = sum(not ok for ok in accepted)
risk = errors / len(accepted) if accepted else None
print("逐題", correct, "/", len(rows))
print("兩題都對", paired, "/", len(pairs))
print("接受", len(accepted), "/", len(rows), "錯誤", errors)
print("coverage", round(len(accepted) / len(rows), 3))
print("accepted risk", None if risk is None else round(risk, 3))
