records = [
    {"family": "貓照片A", "split": "train", "question": "有幾隻貓？"},
    {"family": "貓照片A", "split": "test", "question": "哪隻貓抬起前腳？"},
    {"family": "招牌照片B", "split": "test", "question": "招牌寫什麼？"},
    {"family": "招牌照片B", "split": "test", "question": "有沒有中文字？"},
]
families = {name: {row["family"] for row in records if row["split"] == name} for name in ["train", "test"]}
print("訓練家族", sorted(families["train"]))
print("測試家族", sorted(families["test"]))
print("跨份重疊", sorted(families["train"] & families["test"]))
