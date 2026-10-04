from tiny_perceptron.capstone import build_dataset

splits, _ = build_dataset()
for task in ("rag",):
    row = next(row for row in splits["train"] if row["task"] == task)
    print("任務", task, "問題", row["user"])
    print("工作設定", row["system"], "示範回答", row["answer"])
