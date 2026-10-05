import torch

up = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
down = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def ffn(x):
    return torch.relu(x @ up) @ down


x = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
for batch in (x, torch.cat([x, x], dim=0)):
    print("輸出", ffn(batch).tolist())
    print("token數", len(batch), "權重數", up.numel() + down.numel())
