import math


def softmax(z, T=1.0):
    scaled = [x / T for x in z]
    exp = [math.exp(x - max(scaled)) for x in scaled]
    return [x / sum(exp) for x in exp]


for T in [1.0, 0.1]:
    x = max(softmax([1.0, 0.0, 0.0], T))
    y = max(softmax([0.5, 0.0, -10.0], T))
    print("跨題反例", T, "X", round(x, 5), "Y", round(y, 5))

# 五個假設的校準家族。原始分數相同，gold分別是0、0、0、1、2。
calibration = [([math.log(3), 0.0, 0.0], y) for y in [0, 0, 0, 1, 2]]
temperatures = [0.25 * 2 ** (k / 8) for k in range(33)]


def nll(T):
    return sum(-math.log(softmax(z, T)[y]) for z, y in calibration) / len(calibration)


T = min(temperatures, key=lambda t: (nll(t), abs(t - 1), t))
print("校準選T", T, "平均NLL", round(nll(T), 4))

ids = ["card", "address", "none"]  # 每筆人工分數固定按此順序


def row(z, gold, info_pred=True, info_gold=True, supported_gold=True):
    return dict(z=z, gold=gold, info_pred=info_pred, info_gold=info_gold, supported_gold=supported_gold)


policy = [
    row([4.0, 0.0, -2.0], 0),
    row([0.0, 4.0, -2.0], 1),
    row([3.0, 0.0, -2.0], 0),
    row([2.0, 0.0, -2.0], 1),
    row([0.0, 0.0, 4.0], 2),
    row([4.0, 0.0, -2.0], 0, info_pred=False, info_gold=False),
]
final = {
    "原措辭": [row([4.0, 0.0, -2.0], 0), row([0.0, 3.0, -2.0], 1), row([0.0, 0.0, 4.0], 2)],
    "新措辭": [
        row([4.0, 0.0, -2.0], 1),
        row([2.0, 0.0, -2.0], 0),
        row([4.0, 0.0, -2.0], 0, info_pred=True, info_gold=False),
    ],
}


def evaluate(rows, tau):
    accepted = errors = 0
    for r in rows:
        p = softmax(r["z"], T)
        best = min(range(len(ids)), key=lambda i: (-p[i], ids[i]))
        take = p[best] >= tau and ids[best] != "none" and r["info_pred"]
        if take:
            accepted += 1
            errors += best != r["gold"] or not r["supported_gold"] or not r["info_gold"]
    risk = errors / accepted if accepted else None
    return accepted, errors, accepted / len(rows), risk


taus = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
feasible = []
for tau in taus:
    n, e, coverage, risk = evaluate(policy, tau)
    if risk is not None and risk <= 0.05:
        feasible.append((coverage, tau))
if not feasible:
    raise RuntimeError("決策資料上沒有可行門檻；不能強行部署")
# 覆蓋率最大者優先；同覆蓋率選較高門檻。之後固定，不讀final來調。
_, tau = max(feasible)
print("固定tau", tau)
for name, rows in [("policy", policy), *final.items(), ("全部final", sum(final.values(), []))]:
    n, e, coverage, risk = evaluate(rows, tau)
    print(
        name,
        "接受",
        n,
        "/",
        len(rows),
        "錯誤",
        e,
        "coverage",
        round(coverage, 4),
        "risk",
        None if risk is None else round(risk, 4),
    )
print("零接受", evaluate(policy, 1.0))
