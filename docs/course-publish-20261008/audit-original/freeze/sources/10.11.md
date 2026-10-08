## 10.11 圖找文和文找圖，為什麼要分開算？

同一張配對分數表，找文字時在每橫列選候選；找圖片時則在每直欄選候選。先看手設表 `[[2.0,0.1],[0.4,1.8]]`：橫列代表圖，直欄代表文字，正確配對都是索引 0、1。

```python
import torch
from torch.nn import functional as F

scores = torch.tensor([[2.0, 0.1], [0.4, 1.8]], requires_grad=True)
labels = torch.arange(2)
image_to_text = F.cross_entropy(scores, labels)
text_to_image = F.cross_entropy(scores.T, labels)
loss = (image_to_text + text_to_image) / 2
loss.backward()
print("圖找文", round(image_to_text.item(), 4))
print("文找圖", round(text_to_image.item(), 4))
print("平均", round(loss.item(), 4))
print("梯度符號", torch.sign(scores.grad).tolist())
```

原表算「圖找文」，轉置 `scores.T` 讓文字變橫列、圖變直欄，算「文找圖」。兩個交叉熵約 0.1799、0.1758，平均約 0.1779；梯度符號是 `[[-1,1],[1,-1]]`。同一分數因此收到兩方向候選競爭的影響。

兩項不必相等：圖找文比較 2.0 對 0.1、0.4 對 1.8；文找圖比較 2.0 對 0.4、0.1 對 1.8，差距不同。若只用對稱表，很容易把雙向誤讀為同一算式重複兩次。

雙向平均表達兩種查詢都重要，但不保證每次效果勝過單向。原六類小實驗中，兩種版本的最後題、兩個方向都為 6/6，沒有觀察到答對數優勢；驗證也非全部正確。這只能支持該有限候選配對，不能當大型檢索比較。

檢索是在既有卡片中選一張，生成描述要逐步寫出文字。兩者可共享特徵，能力證據仍需分開。相同描述對多張圖時，也要沿用上一節的多正確配對規則。

練習改成 `(2*image_to_text + text_to_image)/3`，預期梯度表 `scores.grad` 仍為2×2，代價更靠近圖找文。它改的是重視程度，沒有增加新圖片資訊。

<details>
<summary>回顧與查證</summary>

可回顧：[10.10把配對當分類](10.md#10.10)、[W.5矩陣轉置](../first-steps.md#W.5)。

原實驗的資料切分、更新設定與逐題結果見[既有實報](../../docs/course-experiments/results/contrastive.json)；重做入口見[實驗說明](../../docs/course-experiments/README.md)。這是上述有限任務的歷史紀錄。

</details>
