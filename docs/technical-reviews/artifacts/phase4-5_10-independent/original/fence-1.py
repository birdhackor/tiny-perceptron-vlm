candidates = [
    {"lr": 0.001, "validation": 0.7},
    {"lr": 0.01, "validation": 0.8},
]
selected = candidates[0]
for candidate in candidates[1:]:
    if candidate["validation"] > selected["validation"]:
        selected = candidate
print("選擇的設定", selected)
print("test仍封存，尚未計算最終成績")
