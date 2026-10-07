import torch

chars = ["貓", "看", "狗", "。"]
p = torch.tensor(
    [
        [0.05, 0.85, 0.05, 0.05],
        [0.40, 0.05, 0.50, 0.05],
        [0.05, 0.70, 0.05, 0.20],
        [0.50, 0.05, 0.40, 0.05],
    ]
)
g = torch.Generator().manual_seed(42)
current = 0
output = [chars[current]]
for _ in range(12):
    current = torch.multinomial(p[current], 1, generator=g).item()
    output.append(chars[current])
print("".join(output))
assert len(output) == 13
