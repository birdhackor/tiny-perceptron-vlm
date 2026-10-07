## 17.12 中間特徵的範圍，在何時決定？

校準材料的特徵是`[−0.5,0.2,0.8,1]`，最大絕對值1；新輸入是`[−0.2,0.4,3]`，出現了3。固定先前範圍，3會被截到1；按當次輸入重選範圍，又會把刻度變粗。

用代表工作情境的輸入收集範圍，叫**校準（calibration）**，本身不必更新模型權重。本節用**固定／static**與**動態／dynamic**區分activation範圍是在事前還是當次決定，沒有替所有工具的同名API定義配方。

```python
import torch

calibration = torch.tensor([-0.5, 0.2, 0.8, 1.0])
evaluation = torch.tensor([-0.2, 0.4, 3.0])
fixed_max = calibration.abs().max()
dynamic_max = evaluation.abs().max()


def simulate(x, maximum):
    scale = maximum / 7
    return (x / scale).round().clamp(-7, 7) * scale


for name, maximum in (("固定", fixed_max), ("動態", dynamic_max)):
    restored = simulate(evaluation, maximum)
    overflow = (evaluation.abs() > maximum).sum().item()
    print(name, "範圍", maximum.item(), "超出項數", overflow, "還原", restored.round(decimals=4).tolist())
```

這個算例使用−7到7的整數格子。`simulate`以`maximum/7`選scale，讓整數7對應正的maximum；除刻度並取整後，`clamp(-7,7)`把超出範圍的碼壓到兩端，乘回scale時便最多還原到正負maximum。固定範圍1，超出1項，還原約`[−0.1429,0.4286,1]`；動態範圍3，沒有超出，還原約`[0,0.4286,3]`。動態保留3，卻把−0.2歸零；「沒有超出」不是唯一品質標準。

固定方法省去當次估範圍的工作，但校準資料若低估了實際常見的範圍，就可能頻繁截斷。動態需額外統計當前特徵，仍有有限格子精度。校準範圍也可以按百分位數選：例如能涵蓋99%已收集絕對值的界線，允許極少數值截斷來換較細刻度。

資料應涵蓋實際會遇到的內容、長度或音量，最後測試另留。若看測試結果再反覆選範圍，那組分數已受到選擇影響。

練習在calibration加3，evaluation不變，固定最大也變3，兩方法得到相同還原。這是受控算例，實際校準不能替每個最後測試錯誤臨時補資料。

