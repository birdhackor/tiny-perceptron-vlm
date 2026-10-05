import torch
from tiny_perceptron.model import loss_sum

logits = torch.zeros(2, 4, 5, requires_grad=True)
labels = torch.tensor([[1, 2, 3, 4], [1, 2, -100, -100]])
total, count = loss_sum(logits, labels)
mean_loss = total / count
print("有效答案", count.item())
print("總代價", total.item(), "平均代價", mean_loss.item())
mean_loss.backward()
