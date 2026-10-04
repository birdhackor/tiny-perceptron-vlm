from collections import Counter

from tiny_perceptron.capstone import build_dataset

splits, _ = build_dataset()
counts = Counter(row["task"] for row in splits["test"])
for task, count in sorted(counts.items()):
    print("最後檢查", task, "分母", count)
print("合計", sum(counts.values()), "這只是考題數量，不是答對數量")
