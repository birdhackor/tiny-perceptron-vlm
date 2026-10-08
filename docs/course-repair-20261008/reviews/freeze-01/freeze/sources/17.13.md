## 17.13 保住大值與保住小值，如何看取捨？

四個特徵`[0.1,0.2,0.3,100]`共用4-bit刻度。為容納100，scale100/7約14.2857，前三項全落0。這種少數遠大於其他值的數叫**極端值（outlier）**。

先把來源截到−1至1，scale改1/7，可以保住更多小值細節；但100變1，損失99。要看這項取捨，誤差必須對原x計算，不能只比截斷後來源。

```python
import torch
from tiny_perceptron.quantization import quantize_symmetric

x = torch.tensor([0.1, 0.2, 0.3, 100.0])
for clip in (False, True):
    source = x.clamp(-1, 1) if clip else x
    q, scale = quantize_symmetric(source, bits=4)
    restored = q.float() * scale
    error = (x - restored).abs()
    print("截斷", clip, "還原", restored.round(decimals=4).tolist())
    print(
        "小值MAE",
        round(error[:3].mean().item(), 4),
        "極值差",
        round(error[-1].item(), 4),
        "全部MAE",
        round(error.mean().item(), 4),
    )
```

關鍵是`error=(x−restored).abs()`。不截斷時還原`[0,0,0,100]`，小值MAE0.2、極值差0、全部MAE0.15；截斷後小值MAE約0.0381，極值差99，全部MAE約24.7786。

平均與分組結果給了不同視角：小值明顯改善，總平均卻變糟。極值如果是噪音，或代表任務的重要訊號，會影響合理選擇。本例沒有任務模型，不能替我們決定。

也可以縮小共用scale的分組，或讓敏感位置維持較高精度，代價是附帶儲存與運算。門檻先用校準、驗證資料比較，再用不參與挑選的最後題檢查完整用途。

練習將100改2，截斷極值差變1，小值還原不變；不截斷的刻度則變2/7，小值不再全部歸零。分清這次改的是極值，不是bit或截斷界線。

