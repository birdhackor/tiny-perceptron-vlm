import torch
from tiny_perceptron.alignment import reliability_bins

confidence = torch.tensor([0.55, 0.65, 0.85, 0.95])
correct = torch.tensor([1, 0, 1, 0])
bins = reliability_bins(confidence, correct, bins=2)
for row in bins:
    print(
        "區間",
        row["left"],
        row["right"],
        "筆數",
        row["count"],
        "信心",
        round(row["confidence"], 2),
        "正確率",
        row["accuracy"],
    )
ece = sum(row["count"] / len(confidence) * abs(row["accuracy"] - row["confidence"]) for row in bins)
print("平均校準差", round(ece, 2))
