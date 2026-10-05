def rule_best_action(mode):
    """獨立驗收規則，沒有讀取 RM、policy 或候選機率。"""
    return {"number": 0, "explain": 1, "missing": 3}[mode]

def build_records():
    """人工指定偏好規則；題目與候選由固定 Python 模板生成。"""
    records = []
    # 單價限制為正，避免單價 0 時即使不知道數量也能確定總價的例外。
    for a in range(1, 11):
        for b in range(a, 11):
            for mode in MODES:
                if mode == "missing":
                    prompt = f"單價 {a} 元，運費 {b} 元，總價多少？"
                    candidates = [
                        str(a + b),
                        f"單價加運費是 {a + b} 元，所以總價是 {a + b} 元。",
                        str(a + b + 1),
                        "請補充購買數量。",
                    ]
                    # 數量未給：只能確定澄清比任何擅自報總價更好，不強迫錯答互相排名。
                    preferences = [[3, other] for other in range(3)]
                    correctness = [False, False, False, True]
                else:
                    prompt = f"{'只回數字' if mode == 'number' else '用一句話解釋'}：{a} 加 {b} 等於多少？"
                    candidates = [str(a + b), f"{a} 加 {b} 是 {a + b}。", str(a + b + 1), "請再提供更多資訊。"]
                    ranking = [0, 1, 3, 2] if mode == "number" else [1, 0, 3, 2]
                    preferences = [
                        [winner, loser] for index, winner in enumerate(ranking) for loser in ranking[index + 1 :]
                    ]
                    correctness = [True, True, False, False]
                best = rule_best_action(mode)
                records.append(
                    {
                        "family": f"pair:{a}:{b}",
                        "operands": [a, b],
                        "mode": mode,
                        "prompt": prompt,
                        "candidates": candidates,
                        "features": [a / 10, b / 10, float(mode == "explain"), float(mode == "missing")],
                        "preference_pairs": preferences,
                        "expected_action": best,
                        "candidate_content_or_clarification_correct": correctness,
                        "candidate_meets_full_request": [action == best for action in range(4)],
                        "label_source": "human-designed fixed ranking rules, applied by Python; no recruited human raters",
                        "candidate_source": "Python templates compute answers in advance; policy does not calculate or generate text",
                    }
                )
    return records
