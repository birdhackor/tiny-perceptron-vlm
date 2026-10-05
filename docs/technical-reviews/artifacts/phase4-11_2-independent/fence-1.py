import torch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM

model = MultiModalLM(TinyLM(ModelConfig(width=8)))
model.requires_grad_(False)
model.image_projector.requires_grad_(True)
trainable = []
parameters_to_update = []
for name, p in model.named_parameters():
    if p.requires_grad:
        trainable.append((name, p.numel()))
        parameters_to_update.append(p)
optimizer = torch.optim.AdamW(parameters_to_update, lr=0.001)
print("可訓練", trainable)
print("更新參數總數", sum(p.numel() for group in optimizer.param_groups for p in group["params"]))
