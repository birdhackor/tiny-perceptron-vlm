import torch
from tiny_perceptron.quantization import fake_quantize

x = torch.tensor([0.1, 0.4, 0.9], requires_grad=True)
y = fake_quantize(x, bits=4)
y.sum().backward()
print("向前", y.detach().round(decimals=4).tolist())
print("STE梯度", x.grad.tolist())
hard = x.detach().clone().requires_grad_()
scale = 0.9 / 7
plain = (hard / scale).round() * scale
plain.sum().backward()
print("直接round梯度", hard.grad.tolist())
