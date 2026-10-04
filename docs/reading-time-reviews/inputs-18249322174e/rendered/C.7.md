# C.7：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
import torch

logits = torch.tensor([0.0, 0.0], requires_grad=True)
action = 1
reward, baseline = 1.0, 0.5
before = logits.detach().softmax(0)
loss = -(reward - baseline) * logits.log_softmax(0)[action]
loss.backward()
with torch.no_grad():
    updated = logits - 0.1 * logits.grad
    after = updated.softmax(0)
print("梯度", logits.grad.tolist())
print("新分數", updated.tolist())
print("前機率", before.tolist())
print("後機率", [round(p, 6) for p in after.tolist()])
```

```text
梯度 [0.25, -0.25]
新分數 [-0.02500000037252903, 0.02500000037252903]
前機率 [0.5, 0.5]
後機率 [0.487503, 0.512497]

```

