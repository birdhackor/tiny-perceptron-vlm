## 1.6 怎樣把接字偏好存成可調的數字？

為了逐格看清可調表，我們將字表縮成三個候選：「貓、看、狗」，本節重新約定 ID 依序是 0、1、2。看到貓時，我們先給下一字分數 `[0,2,0]`：看得分最高，貓與狗相同。這樣存偏好的好處是，以後可依答案調整分數，不必每次重數整段文字。

這些可調數字叫**參數**。表的列仍是目前字，欄是候選下一字；分數能正、能負，不要求總和為 1。ID 對照固定，訓練時要改的是表格裡的分數。

PyTorch 的 `nn.Embedding` 是可調的查表工具。`nn.Embedding(3,3)` 建立三列、每列三數；這裡直接用三數當候選分數。先手動填表，再查貓與狗兩列。

```python
import torch
from torch import nn

scores = nn.Embedding(3, 3)
with torch.no_grad():
    scores.weight.zero_()
    scores.weight[0, 1] = 2.0
inputs = torch.tensor([0, 2])
output = scores(inputs)
print(scores.weight)
print(output)
```

整張 `scores.weight` 是 `[[0,2,0],[0,0,0],[0,0,0]]`。`inputs=[0,2]` 要求取第 0 與第 2 列，所以 `output` 是 `[[0,2,0],[0,0,0]]`，形狀 `[2,3]`：兩道目前字問題，每題三個候選。

`zero_()` 清零，接著只把貓列、看欄設成 2；尾端底線表示直接改原物件。`torch.no_grad()` 包住這次手動填表，免得把設定初值記成需要求導的運算。查表之後並沒有做參數更新，這裡所有偏好都由我們指定。

分數表需要 V×V 格：V 種目前字，每種都有 V 個下一字候選。它能給分也能被調整，但仍只看最近一字；看到「顏色=」與「形狀=」時，最後一字同為等號，仍會取同一列。

練習在填表區加入 `scores.weight[2,0]=3.0`。預測貓那題保持 `[0,2,0]`，狗那題變成 `[3,0,0]`，再核對。你改了哪一列，就先影響使用那列的輸入。

<details>
<summary>補充與重做</summary>

輸出可能附 `Parameter containing`、`requires_grad=True` 與 `grad_fn=<EmbeddingBackward0>`。前兩者表示這張表是日後可求導的參數，後者表示查表留下運算記錄；它們都不表示已訓練。tensor 見 [W.3](../first-steps.md#W.3)。第 2 章會用同一查表工具取得中間特徵，再計算候選分數。

</details>

