import torch
from tiny_perceptron.posttraining import ppo_clipped_objective
ratio = torch.tensor([0.7, 1.0, 1.3, 0.7, 1.0, 1.3], requires_grad=True)
advantage = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
old_log_probability = torch.full((6,), 0.5).log()
new_log_probability = old_log_probability + ratio.log()
result = ppo_clipped_objective(new_log_probability, old_log_probability, advantage, clip_range=0.2)
loss = -result["surrogate"].sum()
loss.backward()
print("待提高的值", [round(value, 2) for value in result["surrogate"].tolist()])
print("下降代價時對ratio的梯度", [round(value, 2) for value in ratio.grad.tolist()])
print('torch',torch.__version__)
