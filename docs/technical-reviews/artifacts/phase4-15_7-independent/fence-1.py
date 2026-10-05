import math
import torch

for renormalize in (False, True):
    scores = torch.tensor([math.log(2), 0.0], requires_grad=True)
    prob = scores.softmax(-1)
    p = prob.topk(1).values[0]
    gate = p / p if renormalize else p
    output = gate * 2
    loss = output.square()
    loss.backward()
    print("重新正規化", renormalize, "輸出", round(output.item(), 4), "梯度", scores.grad.round(decimals=4).tolist())
