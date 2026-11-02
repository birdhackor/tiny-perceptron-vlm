"""固定 T／tau 網格；calibration、policy、final 不互相調參。"""

import math

import torch
import torch.nn.functional as F

TEMPERATURES = tuple(0.25 * 2 ** (k / 8) for k in range(33))
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)


def calibrated(logits, temperature):
    if not math.isfinite(temperature) or temperature <= 0 or not 2 <= len(logits) <= 8:
        raise ValueError("T／候選數不合法")
    values = torch.tensor(logits, dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("公開 logits 必須有限")
    return F.softmax(values / temperature, dim=-1).tolist()


def select_temperature(families):
    if len(families) != 80 or len({row["family"] for row in families}) != 80:
        raise ValueError("calibration 必須是 80 個不同家族")
    losses = []
    for temperature in TEMPERATURES:
        terms = []
        for family in families:
            if set(family["tasks"]) != {"choice", "info", "rubric"}:
                raise ValueError("每家族必須完整提供三型預定代表題")
            task_terms = []
            for record in family["tasks"].values():
                logits = torch.tensor(record["logits"], dtype=torch.float64)
                calibrated(record["logits"], temperature)  # 保留公開輸入型別／範圍驗證。
                gold = record["ids"].index(record["gold"])
                task_terms.append(float(-F.log_softmax(logits / temperature, dim=-1)[gold]))
            terms.append(sum(task_terms) / 3)
        losses.append({"temperature": temperature, "family_equal_type_equal_nll": sum(terms) / 80})
    best = min(
        losses, key=lambda row: (row["family_equal_type_equal_nll"], abs(row["temperature"] - 1), row["temperature"])
    )
    return {"temperature": best["temperature"], "grid": losses, "calibration_families": 80}


def accepted_rows(families, temperature, threshold):
    rows = []
    for family in families:
        choice, info = family["tasks"]["choice"], family["choice_info"]
        probabilities = calibrated(choice["logits"], temperature)
        # p_max 是校準後最大值；答案仍由 raw logits + stable ID 決選。
        chosen = min(range(len(choice["ids"])), key=lambda i: (-choice["logits"][i], choice["ids"][i]))
        enough = min(range(len(info["ids"])), key=lambda i: (-info["logits"][i], info["ids"][i]))
        answer = choice["ids"][chosen]
        accepted = max(probabilities) >= threshold and answer != "none" and info["ids"][enough] == "yes"
        if type(family["world_supported"]) is not bool or type(family["choice_information_sufficient"]) is not bool:
            raise ValueError("gold support／資訊契約需明示 bool")
        error = accepted and (
            answer != choice["gold"] or not family["world_supported"] or not family["choice_information_sufficient"]
        )
        rows.append(
            {
                "family": family["family"],
                "accepted": accepted,
                "error": bool(error),
                "answer": answer,
                "p_max": max(probabilities),
            }
        )
    accepted = sum(row["accepted"] for row in rows)
    errors = sum(row["error"] for row in rows)
    return {
        "families": len(rows),
        "accepted": accepted,
        "errors": errors,
        "coverage": accepted / len(rows) if rows else 0,
        "empirical_risk": errors / accepted if accepted else None,
        "rows": rows,
    }


def select_policy(families, temperature):
    if len(families) != 60 or len({row["family"] for row in families}) != 60:
        raise ValueError("policy 必須是獨立 60 家族")
    grid = [{"threshold": threshold, **accepted_rows(families, temperature, threshold)} for threshold in THRESHOLDS]
    feasible = [row for row in grid if row["accepted"] > 0 and row["errors"] / row["accepted"] <= 0.05]
    if not feasible:
        return {"feasible": False, "threshold": None, "grid": grid, "behavior": "零自動接受；全部轉交"}
    best = max(feasible, key=lambda row: (row["coverage"], row["threshold"]))
    return {"feasible": True, "threshold": best["threshold"], "grid": grid}


def wilson(correct, total):
    if not 0 <= correct <= total or total < 1:
        return None
    p, z = correct / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - radius), min(1, center + radius)]
