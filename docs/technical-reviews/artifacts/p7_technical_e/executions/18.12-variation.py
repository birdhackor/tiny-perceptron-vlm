import torch
from tiny_perceptron.model import TinyLM,ModelConfig
from tiny_perceptron.alignment import distillation_kl
torch.manual_seed(0);teacher=TinyLM(ModelConfig(width=16,experts=3,top_k=2)).eval().requires_grad_(False);student=TinyLM(ModelConfig(width=8,vocab_size=265));ids=torch.tensor([[1,2]])
with torch.no_grad():t=teacher(ids)['logits']
s=student(ids)['logits'];print('shapes',t.shape,s.shape)
try:distillation_kl(s,t,torch.tensor([[1,2]]),temperature=1)
except ValueError as e:print('expected ValueError',str(e))
else:raise AssertionError('shape mismatch not rejected')
