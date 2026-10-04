records = [
    {"family": "公園照片A", "split": "train", "question": "人正在做什麼？"},
    {"family": "公園照片A", "split": "test", "question": "圖裡有腳踏車嗎？"},
    {"family": "招牌照片B", "split": "test", "question": "請讀出文字。"},
    {"family": "招牌照片B", "split": "test", "question": "圖片裡有中文字嗎？"},
]
families = {name: {row["family"] for row in records if row["split"] == name} for name in ["train", "test"]}
print("訓練家族", sorted(families["train"]))
print("測試家族", sorted(families["test"]))
print("跨份重疊", sorted(families["train"] & families["test"]))
