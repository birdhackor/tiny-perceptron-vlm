from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM

model = MultiModalLM(TinyLM(ModelConfig(width=8)))
for mode in ["projector", "partial", "all"]:
    model.requires_grad_(mode == "all")
    model.image_projector.requires_grad_(True)
    if mode == "partial":
        model.language.blocks[-1].requires_grad_(True)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(mode, count)
