## 17.5 不同大小的權重，應共用一把刻度尺嗎？

兩列權重分別是`[0.01,0.03,0.05,0.08]`與`[10,30,50,80]`。四位元對稱碼−7至7，全表共用刻度要容納80，所以scale約11.4286。小列除以它全靠近0，四項會一起消失。

若每列自己選刻度，小列用`0.08/7`，大列仍用`80/7`，就能在不改位元數下保留小列差別。這次比較的是**哪些數共用刻度**。

```python
import torch
from tiny_perceptron.quantization import quantize_symmetric

w = torch.tensor([[0.01, 0.03, 0.05, 0.08], [10.0, 30.0, 50.0, 80.0]])
for per_row in (False, True):
    q, scale = quantize_symmetric(w, bits=4, per_channel=per_row)
    restored = q.float() * scale
    error = (w - restored).abs().mean(-1)
    print("每列獨立", per_row, "整數", q.tolist())
    print("每列MAE", error.round(decimals=4).tolist(), "scale格數", scale.numel())
```

`per_channel=True`在這張線性權重表指每個輸出列各一個scale，不是每個輸入欄。`abs()`先取每格差的絕對值，`mean(-1)`沿最後一維取平均；這張2列、4欄表的最後一維是每列四個權重，因此得到兩個「每列平均絕對誤差」（MAE）。全表共用時，小列碼全0，MAE0.0425；逐列時碼為`[1,3,4,7]`，MAE約0.0025。大列兩種都約2.5。逐列看誤差，才不讓大列平均掩蓋小列。

代價是多存刻度：一個FP32 scale需4byte，兩列獨立變8byte。也可以一列內每固定幾個權重共用刻度，叫**分組量化（per-group）**；組小通常較少受極值牽連，卻需更多scale與分組資訊。若每個權重都藏一份浮點scale，壓縮收益可能消失。

練習把第一列乘1000，兩列範圍便相同。共用與逐列誤差會接近，逐列仍多存scale。分組是精度與附帶成本的取捨，不是免費改善。

