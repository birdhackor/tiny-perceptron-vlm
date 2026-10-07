import torch
from tiny_perceptron.quantization import fake_quantize
x=torch.tensor([.1,.4,.9],requires_grad=True)
y=fake_quantize(x,bits=4)
y.square().sum().backward()
print('y',y.tolist(),'grad',x.grad.tolist(),'2y',(2*y).tolist())
assert torch.allclose(x.grad,2*y)
