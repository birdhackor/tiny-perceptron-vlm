import torch

scores = torch.tensor([[2.0, 1.0, 0.0]])
correct_index = 1
for scale in [1.0, 10.0]:
    probability = (scores * scale).softmax(-1)
    confidence, prediction = probability.max(-1)
    print(
        "倍率",
        scale,
        "信心",
        round(confidence.item(), 4),
        "選擇",
        prediction.item(),
        "正確",
        prediction.item() == correct_index,
    )
