import torch
value = torch.tensor([0.4], requires_grad=True)
target = torch.tensor([1.0])
value_loss = (value - target).square().mean()
value_loss.backward()
print("估計員代價與梯度", round(value_loss.item(), 4), round(value.grad.item(), 4))
print('torch',torch.__version__)
