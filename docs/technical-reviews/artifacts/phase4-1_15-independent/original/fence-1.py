import torch

z = torch.tensor([3.0, 1.0, 0.0])
for temperature in [0.5, 1.0, 2.0]:
    probability = (z / temperature).softmax(dim=0)
    print(temperature, probability)
print("greedy的候選ID", z.argmax().item())
