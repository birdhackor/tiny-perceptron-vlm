from pathlib import Path
import torch

paths = [
    Path("checkpoints/attributes.pt"),
    Path("checkpoints/attributes-int4.pt"),
    Path("checkpoints/attributes-int8.pt"),
]
for path in paths:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"]
    tensor_bytes = sum(t.numel() * t.element_size() for t in state.values())
    print(path.name, "模型張量bytes", tensor_bytes, "整個檔案bytes", path.stat().st_size)
