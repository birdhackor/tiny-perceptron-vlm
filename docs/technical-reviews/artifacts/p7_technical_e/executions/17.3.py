import torch
w = torch.tensor([[0.7, 0.2], [-0.7, 1.2]])
scale = 0.5
restored = (w / scale).round() * scale
x = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
original_output = x @ w.T
compressed_output = x @ restored.T
print("權重平均絕對差", round((w - restored).abs().mean().item(), 4))
print("原輸出", original_output.round(decimals=4).tolist())
print("還原權重輸出", compressed_output.round(decimals=4).tolist())
print("逐項輸出差", (original_output - compressed_output).round(decimals=4).tolist())
