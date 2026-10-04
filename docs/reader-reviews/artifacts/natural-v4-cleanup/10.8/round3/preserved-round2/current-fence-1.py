import torch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM

torch.manual_seed(0)
lm = TinyLM(ModelConfig(width=8)).eval()
multimodal = MultiModalLM(lm).eval()
ids = torch.tensor([1, 20, 30])
a = lm(ids[None])["logits"]
b = multimodal(ids)["logits"]
print("分數形狀", tuple(a.shape))
print("純文字入口相同", torch.equal(a, b))
