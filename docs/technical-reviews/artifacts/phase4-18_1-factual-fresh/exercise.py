import torch

candidates = ["4", "3", "5"]
probabilities = [torch.tensor([0.8, 0.15, 0.05]), torch.tensor([0.4, 0.35, 0.25])]
for prob in probabilities:
    answer = candidates[prob.argmax().item()]
    print("回答", answer)
