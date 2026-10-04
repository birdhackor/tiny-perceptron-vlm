from collections import Counter

from tiny_perceptron.capstone import build_dataset

splits, manifest = build_dataset(seed=43)
families = {name: {row["family"] for row in rows} for name, rows in splits.items()}
for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
    print(first, second, "重疊家族", len(families[first] & families[second]))
for name, rows in splits.items():
    print(name, "題數", len(rows), "任務", dict(Counter(row["task"] for row in rows)))
print("第一筆訓練問題", splits["train"][0]["user"])
audio_test = [row for row in splits["test"] if row["task"] == "audio"]
labels = Counter(row["audio"]["pitch"] for row in audio_test)
print("最後音高標籤", dict(labels))
print("完全不聽、固定答high的基準", labels["high"], "/", len(audio_test))
for task in ("image_color", "image_shape"):
    baseline = manifest["task_majority_baselines"]["test"][task]
    print(task, "固定答案基準", baseline["majority_label"], baseline["correct"], "/", baseline["count"])
