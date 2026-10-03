import torch

cases = {
    "較均衡": torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
    "集中": torch.tensor([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]),
}
for name, prob in cases.items():
    chosen = prob.argmax(-1)
    fraction = torch.bincount(chosen, minlength=3).float() / len(chosen)
    mean_prob = prob.mean(0)
    auxiliary = 3 * (fraction.detach() * mean_prob).sum()
    total = 0.5 + 0.01 * auxiliary
    print(name, "負載", fraction.tolist(), "輔助項", round(auxiliary.item(), 4), "總誤差", round(total.item(), 4))
