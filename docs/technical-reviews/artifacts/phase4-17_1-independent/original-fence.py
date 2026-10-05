import torch

weight = torch.zeros(64, 32)
for dtype in (torch.float32, torch.float16, torch.int8):
    converted = weight.to(dtype)
    count = converted.numel()
    per_value = converted.element_size()
    print(dtype, "元素", count, "每格bytes", per_value, "總bytes", count * per_value)
