import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene

torch.manual_seed(0)
tok = ByteTokenizer()
model = MultiModalLM(TinyLM(ModelConfig(width=8)))
model.requires_grad_(False)
model.image_projector.requires_grad_(True)
answer = tok.encode("red square") + [tok.eos_id]
prefix = [tok.bos_id, tok.user_id, tok.image_id, tok.eos_id, tok.assistant_id]
ids = torch.tensor(prefix + answer)
labels = torch.tensor([-100] * len(prefix) + answer)
out = model(ids, labels, image=scene())
model.image_projector.requires_grad_(False)
loss = masked_loss(out["logits"], out["labels"])
print("可訓練總數", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("loss.requires_grad", loss.requires_grad)
loss.backward()
print("backward已完成", True)
print("接頭未累積梯度", model.image_projector.weight.grad is None)
print("語言權重未累積梯度", model.language.embedding.weight.grad is None)
