examples = [
    {"question": "1加2等於多少？", "calculator": False, "action": "ASK"},
    {"question": "請解釋加法的意思。", "calculator": True, "action": "DIRECT"},
    {"question": "請原樣回答數字3。", "calculator": True, "action": "DIRECT"},
    {"question": "請算總價。", "calculator": True, "action": "ASK"},
    {"question": "1加2等於多少？", "calculator": False, "action": "ASK"},
]
for row in examples:
    print(row["question"], "計算器可用", row["calculator"], "→", row["action"])
