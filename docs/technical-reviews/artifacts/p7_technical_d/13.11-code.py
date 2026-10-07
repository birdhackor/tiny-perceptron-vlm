import torch
from tiny_perceptron.posttraining import bandit_advantage
rewards = torch.tensor([1.0, 0.0])
old_values = torch.tensor([0.4, 0.4], requires_grad=True)
advantage = bandit_advantage(rewards, old_values)
print("優勢估計", [round(value, 2) for value in advantage.tolist()])
print("更新策略時還追蹤基準梯度嗎", advantage.requires_grad)
print('torch',torch.__version__)
