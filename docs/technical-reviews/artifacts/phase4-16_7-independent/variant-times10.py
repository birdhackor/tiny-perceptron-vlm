import torch

a = torch.tensor([[1.001, 2.002], [3.003, 4.004]]) * 10
b = torch.tensor([[0.5, 1.0], [1.5, -1.0]])
full = a @ b
with torch.autocast("cpu", dtype=torch.bfloat16):
    low = a @ b
print("原輸入型別", a.dtype, "低精度輸出", low.dtype)
print("FP32", full.round(decimals=4).tolist())
print("BF16", low.float().tolist())
print("最大差異", round((full - low.float()).abs().max().item(), 4))
