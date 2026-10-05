import torch

text = "貓看狗。狗吃魚。貓看鳥。貓"
chars = sorted(set(text))
ids = [chars.index(char) for char in text]
counts = torch.ones(len(chars), len(chars))
for current, answer in zip(ids[:-1], ids[1:]):
    counts[current, answer] += 1
probability = counts / counts.sum(dim=1, keepdim=True)
row = chars.index("貓")
print(chars)
print(counts[row])
print(probability[row])
assert torch.allclose(probability.sum(dim=1), torch.ones(len(chars)))
