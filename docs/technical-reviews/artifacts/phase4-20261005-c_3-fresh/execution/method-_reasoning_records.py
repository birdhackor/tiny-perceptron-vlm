def _reasoning_records():
    rows = []
    for a in range(6):
        for b in range(6):
            for c in range(6):
                rows.append(
                    {
                        # 相同數字的所有排列共用 family，測試沒有訓練題的重新排列。
                        "family": ":".join(map(str, sorted((a, b, c)))),
                        "a": a,
                        "b": b,
                        "c": c,
                        "question": f"({a}+{b})+{c}=?",
                        "truth": a + b + c,
                    }
                )
    return rows
