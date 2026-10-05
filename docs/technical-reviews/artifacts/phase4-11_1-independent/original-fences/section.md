## 11.1 接頭輸出不同，就算理解圖片了嗎？

文字描述「紅色圓形」時能回答紅，換成紅圓圖片時也應回答紅，再換藍圓圖片時應改答藍。這三個對照分別檢查原文字能力、圖片接入與圖片內容的作用。只知道兩個輸出分數不同，還不夠。

先看一個更小的算例：寬 4 的全 0、全 1 特徵，經同一接頭變成寬 8 的向量。

```python
import torch
from torch import nn

torch.manual_seed(0)
projector = nn.Linear(4, 8)
a, b = torch.zeros(1, 4), torch.ones(1, 4)
difference = (projector(a) - projector(b)).norm()
print("輸出形狀", tuple(projector(a).shape))
print("對輸入差異敏感", difference.item() > 0)
```

輸出 `(1,8)`，兩個輸出差向量的長度大於 0，印 True。`Linear` 用權重乘輸入並加偏置；例如權重 `[1,2,0,0]`、偏置 0.5，兩輸入會得到 0.5、3.5。隨機配方同樣可能傳遞差異，但沒有正確答案或文字模型，所以不證明懂紅藍。

「維度相容」只說每位置的數字個數符合介面；「語義對齊」還要確認這些線索能被核心用來生成正確內容。原文字題先失敗，就不能把後面所有失敗歸給圖片接頭；文字題成功而圖片題失敗，才進一步查圖片特徵與接頭。

將以下兩行放在算 difference 之前，再重新執行：

```python
with torch.no_grad():
    projector.weight.zero_()
```

所有輸入乘 0，輸出只剩相同偏置，所以差長度 0、比較 False。這是手動安排對照權重，不是訓練；在差值已算完後清零，舊 difference 也不會自行重算。它找出通路不傳差異，仍不能單獨判斷是否懂圖片。

<details>
<summary>回顧與查證</summary>

可回顧：[10.6的projector接頭](10.md#10.6)、[10.5的視覺特徵訓練](10.md#10.5)。

</details>

