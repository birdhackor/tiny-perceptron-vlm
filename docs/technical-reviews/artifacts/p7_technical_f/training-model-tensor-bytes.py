from pathlib import Path
import torch

paths = [
    Path("outputs/p7_technical_f/checkpoints/attributes.pt"),
    Path("outputs/p7_technical_f/checkpoints/attributes-int4.pt"),
    Path("outputs/p7_technical_f/checkpoints/attributes-int8.pt"),
]
for path in paths:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"]
    tensor_bytes = sum(t.numel() * t.element_size() for t in state.values())
    print(path.name, "模型張量bytes", tensor_bytes, "整個檔案bytes", path.stat().st_size)
