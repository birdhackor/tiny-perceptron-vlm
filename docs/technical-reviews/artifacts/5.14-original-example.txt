import torch

w = torch.tensor(1.0)
gradient = 2 * (w - 3)
print("起點代價", (w - 3).square().item())
for lr in [0.01, 0.1, 2.0]:
    new = w - lr * gradient
    print("步幅", lr, "新位置", new.item(), "新代價", (new - 3).square().item())
