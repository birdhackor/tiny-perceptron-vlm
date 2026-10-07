import torch
print('torch',torch.__version__)
for d in [torch.float32,torch.float16,torch.bfloat16]:print(d,'max',torch.finfo(d).max,'eps',torch.finfo(d).eps,'bytes',torch.tensor(0,dtype=d).element_size())
a=torch.tensor([[1.001,2.002],[3.003,4.004]]);b=torch.tensor([[.5,1.],[1.5,-1.]])
full=a@b
with torch.autocast('cpu',dtype=torch.bfloat16):low=a@b
print("原輸入型別",a.dtype,"低精度輸出",low.dtype);print("FP32",full.round(decimals=4).tolist());print("BF16",low.float().tolist());print("最大差異",round((full-low.float()).abs().max().item(),4))
