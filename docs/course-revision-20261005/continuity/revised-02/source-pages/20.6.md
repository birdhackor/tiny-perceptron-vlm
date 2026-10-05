## 20.6 保留底座，怎麼只學一小份修正？

輸入四項特徵，要輸出三項分數。原矩陣 W 可保持固定，再加一條 A→B 的小修正：A把四項縮成r項，B把r項轉回三項，最後與原輸出相加。r叫rank，此例的修正尺度是 alpha/r；固定底座仍參與計算。

![固定W與可更新A、B接受同一輸入，輸出相加。](../figures/natural_base_adapter.svg)

```python
import torch
from torch import nn
from tiny_perceptron.alignment import LoRALinear

torch.manual_seed(42)
layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=2)
base_before = layer.base.weight.detach().clone()
branch_before = layer.b.detach().clone()
optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
loss = layer(torch.ones(1, 4)).square().mean()
loss.backward()
optimizer.step()
print("可更新數字", sum(p.numel() for p in layer.parameters() if p.requires_grad))
print("原矩陣完全未變", torch.equal(base_before, layer.base.weight))
print("修正B真的改變", not torch.equal(branch_before, layer.b))
```

這個局部例子以輸出平方平均為代價，讓輸出接近0，不是讀照片。`backward()`算梯度，`step()`才真的更新。A有2×4個數，B有3×2個數，共14個可更新數字；原W不變，B改變。這個小例只示範更新機制：14反而比原W的3×4=12多，不能用它示範省參數。分支要更少，須讓 `r×(輸入數+輸出數)` 小於原W的 `輸入數×輸出數`。獨立保存舊值後再比，才能知道哪條路收到更新。

練習把 rank 改成1，先算A的1×4與B的3×1，共7個可更新數字，才比原W的12少；固定底座仍需載入（見[20.3](20.md#20.3)）。

完整候選在語言注意力的q、v投影加rank8、alpha16的LoRA，共1,605,632個可更新數字。視覺底座與Whisper不因這一步重訓。小例學習率0.1用來看一次更新，完整配方為0.00003，不能把兩個目標混成同一次實驗。

修正檔約6.44MB，不含底座或完整續訓狀態。[訓練核對](../../docs/natural-assistant/evidence/v4-runtime/train-37217452291/training-audit-summary.json)記錄112個修正張量的前後指紋改變，底座只有若干位置抽查，沒有逐位比較全部二十億數值。更新發生與用途改善，仍要另分兩種證據。

