import copy
import torch
from tiny_perceptron.model import TinyLM, ModelConfig
policy = TinyLM(ModelConfig(width=8))
reference = copy.deepcopy(policy).eval().requires_grad_(False)
before = reference.embedding.weight.clone()
with torch.no_grad(): policy.embedding.weight.add_(0.1)
scores = reference(torch.tensor([[1, 2]]))["logits"]
print("參考未跟著改", torch.equal(before, reference.embedding.weight))
print("參考不記梯度", scores.requires_grad)
print('torch',torch.__version__)
print('eval doc',torch.nn.Module.eval.__doc__)
print('requires_grad_ doc',torch.nn.Module.requires_grad_.__doc__)
