import torch
from tiny_perceptron.posttraining import exact_kl
value = torch.tensor([0.4], requires_grad=True)
target = torch.tensor([1.0])
value_loss = (value - target).square().mean()
value_loss.backward()
print("估計員代價與梯度", round(value_loss.item(), 4), round(value.grad.item(), 4))
policy_logits = torch.tensor([[0.8, 0.2]]).log()
reference_logits = torch.tensor([[0.5, 0.5]]).log()
print("相對固定參考的KL", round(exact_kl(policy_logits, reference_logits).item(), 4))
v=torch.tensor([1.4],requires_grad=True);l=(v-target).square().mean();l.backward()
print('variation_value',l.item(),v.grad.item());print('identical_kl',exact_kl(reference_logits,reference_logits).item());print('torch',torch.__version__)
