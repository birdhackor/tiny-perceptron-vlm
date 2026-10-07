import torch
from torch import nn
layer = nn.Linear(4, 3).eval()
x = torch.ones(2, 4)
compiled = torch.compile(layer, backend="eager")
with torch.no_grad():
    expected, actual = layer(x), compiled(x)
print("torch",torch.__version__)
print("device",str(x.device))
print("輸出形狀", tuple(actual.shape))
print("介面最大差", (expected - actual).abs().max().item())
setup, plain_per_call, compiled_per_call = 2.0, 0.01, 0.006
for calls in (100, 1000):
    print("假設呼叫數", calls, "原版秒數", plain_per_call * calls, "編譯版秒數", setup + compiled_per_call * calls)
