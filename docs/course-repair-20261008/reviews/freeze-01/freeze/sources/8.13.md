## 8.13 同名設定，為什麼可能給不同修正幅度？

某格原權重W是0.7，LoRA的BA在這格是0.3。按本工具`ΔW=(alpha/rank)BA`，rank2、alpha2時，有效權重為`0.7+0.3=1.0`；alpha改4時變`0.7+0.6=1.3`。加倍的是修正，不是整個1.0。

因此比較縮放時，應固定A、B與rank，只改alpha。下面先讓B非零，再從合併權重扣掉原W，只看ΔW。

```python
import torch
from torch import nn
from tiny_perceptron.alignment import LoRALinear

torch.manual_seed(0)
layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=2)
with torch.no_grad():
    layer.b.fill_(1.0)
delta1 = layer.merged_weight() - layer.base.weight
layer.alpha = 4
delta2 = layer.merged_weight() - layer.base.weight
print("加倍誤差", (delta2 - 2 * delta1).abs().max().item())
print("符合兩倍", torch.allclose(delta2, 2 * delta1, atol=1e-6, rtol=0))
```

第二份修正應是第一份的兩倍。`abs().max()`找逐格最大差，`allclose`檢查所有項是否在容許誤差內；此處絕對容許值`1e-6`，沒有額外相對容許量。應看到接近0的誤差與`True`，細微差來自有限浮點精度。

其他工具可能用`alpha/√rank`。設定名稱相同，不表示公式相同。rank改變也會改A、B形狀，不能只看倍率由1變0.5，就說最後學到的ΔW減半。保存與載入adapter時，需要核對公式、設定、層名和基模。

練習把第二次alpha改成6，兩處比較的`2 * delta1`同步改為`3 * delta1`，印出的名稱也改成三倍。這是既定數字的縮放檢查，沒有訓練新風格。接下來回到更一般的要求：除了怎樣說，還要說對本次指定的部分。

<details>
<summary>補充：既有實驗的條件與完整紀錄</summary>

[8.8的實際訓練](#8.8)固定採用alpha/rank，rank與alpha都是4，因此倍率為1。檔案也明列這個公式，載入工具會先核對它與原模型的數字指紋。這裡把模型參數按固定順序計算成一串識別碼，叫數字指紋；載入時重算，再與保存時的識別碼比對，用來核對預期的基模數字，而不只相信檔名相同。這一組沒有訓練不同rank或另一種縮放公式，不能用其成績選出最佳rank，更不能把同名參數直接移到另一套公式後稱為相同實驗。

</details>

