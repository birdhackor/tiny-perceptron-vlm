## 10.6 8 維圖片特徵，如何接到 12 維文字介面？

圖片入口每位置給 8 個數字，文字核心每位置需要 12 個。先用投影接頭 projector，把原有特徵重新組合成 12 個數字；它改的是每位置的寬度，不是把 16 塊圖變成 12 塊。

```python
import torch
from torch import nn

torch.manual_seed(0)
vision = torch.randn(1, 16, 8)
projector = nn.Linear(8, 12)
language_features = projector(vision)
print("視覺", tuple(vision.shape))
print("文字介面", tuple(language_features.shape))
print("接頭矩陣", tuple(projector.weight.shape))
```

`vision` 是人工造出的 `(1,16,8)` 特徵。線性接頭 `Linear(8,12)` 的矩陣是 `(12,8)`，各位置共用它，輸出 `(1,16,12)`。像輸入 `[1,2]` 分別取第一值、第二值、兩值相加，得到 `[1,2,3]`：新增輸出欄是重新組合，不是多出一份獨立知識。

尺寸相容使後面的算式能執行，含義是否合用還需學習。視覺特徵可能區分紅與藍，文字核心卻把自己的特徵用來區分詞義；成對圖片與正確描述可以教接頭，讓視覺差異成為生成答案的線索，這稱為對齊。

若編碼器把紅藍壓成完全相同表示，接頭無法可靠從相同輸入還原不同顏色。若輸出寬度設為 10，文字模型仍要求 12，則先遇到尺寸不符。這兩種失敗分別是資訊不足與介面不合，不能都靠改圖片大小處理。

練習把 12 改成 10，先算 `(1,16,10)` 和 `(10,8)`，再說明為何不能直接接 12 維核心。下一章再用完整回答確認是否對齊。

<details>
<summary>回顧與查證</summary>

可回顧：[10.3的特徵軸](10.md#10.3)、[W.5矩陣變換](../first-steps.md#W.5)。

</details>

