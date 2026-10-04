## 20.6 保留底座，怎麼只學一小份修正？

你想修改一本書裡某些用法，不一定要把每一頁重印。LoRA讓原有矩陣照常運算，在旁邊加上一條可以學的小修正。它省下需要更新與保存的數字，但原底座仍是計算的一部分。

先回看[8.8的兩張小矩陣](08.md#8.8)與[8.13的縮放約定](08.md#8.13)。原矩陣W將輸入轉成結果；修正路線先經A，縮到較少的中間格，再經B回到所需尺寸，最後加回原結果。中間格數叫rank，常寫r；alpha控制修正的縮放，本例使用alpha/r。

![同一輸入經固定原矩陣與可更新的A、B兩路，在輸出端相加](../figures/natural_base_adapter.svg)

先真的更新一個四項輸入、三項輸出的層：

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

一筆輸入的四個值都是1。輸出的平方平均作為代價，讓這次練習試著把輸出靠近0；它不是圖片或文字任務。`detach().clone()`留下獨立的舊值，才能在更新後比較。A有2×4＝8個數，B有3×2＝6個數，所以可更新數字是14；後兩項是True。

更新器只接收`requires_grad`為True的參數。固定原矩陣表示它的數字不更新，不是把那條路關掉；回答的代價仍要經過底座的計算，才能告訴修正應該怎麼改。本章在圖文底座的語言注意力q與v投影加上rank 8、alpha 16的修正，視覺底座與ASR不因這一步重新訓練。完整微調的學習率固定為0.00003；上方的小層例子用0.1追蹤一次更新，兩者是不同規模、不同目標的練習。

放到完整底座時，這次可更新的修正共有1,605,632個數字，約佔含修正後模型的0.0754%。保存成單份修正權重檔約6.44 MB；它不包含完整底座，也不是包含更新器狀態的續訓包。實際更新範圍只有LoRA，112個修正張量的前後指紋都改變，底座的披露樣本前後未變；這是抽查，沒有宣稱把二十億個底座數字全數逐位比較。小例子幫我們看懂兩條路，[完整訓練核對](../../docs/natural-assistant/evidence/v4-runtime/train-37217452291/training-audit-summary.json)則確認程式真的按這個範圍更新，回答品質仍需另驗。

練習把rank改成1，也把alpha改成1，保留相同的alpha/r縮放。A與B共4＋3＝7個數；原矩陣仍應不變。這只說明可調數字變少，還不能判斷自然任務的品質。下一節會找出真正指導這些數字學習的答案位置。

