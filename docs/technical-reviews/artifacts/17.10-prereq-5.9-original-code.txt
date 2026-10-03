import time
import torch
from tiny_perceptron.model import TinyLM, ModelConfig

model = TinyLM(ModelConfig(width=8))
x = torch.tensor([[1, 2, 3]])
model.eval()
with torch.no_grad():
    model(x)
    start = time.perf_counter()
    for _ in range(10):
        model(x)
    average_seconds = (time.perf_counter() - start) / 10
parameters = model.description()["parameters"]
print("參數格數", parameters, "FP32參數bytes", parameters * 4)
print("CPU每次前向秒數", average_seconds)
