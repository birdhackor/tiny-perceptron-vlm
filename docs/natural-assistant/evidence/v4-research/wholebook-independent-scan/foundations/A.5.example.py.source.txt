examples = [
    {"question": "甲店地址", "hit": True, "correct": True},
    {"question": "乙店地址", "hit": True, "correct": False},
    {"question": "丙店地址", "hit": False, "correct": False},
]
count = len(examples)
recall = sum(e["hit"] for e in examples) / count
accuracy = sum(e["correct"] for e in examples) / count
print("有正確來源", recall)
print("答案正確", accuracy)
for e in examples:
    if not e["correct"]:
        print(e["question"], "已找到但答錯" if e["hit"] else "尚未找到所需來源")
