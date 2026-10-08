## 10.3 每塊 48 個像素值，如何變成 8 個特徵？

上一節每塊有 48 個像素值，接下來希望每個位置用 8 個數字表示。可以用同一張加權配方表，對每塊的 48 個值乘權重、加總，再加偏置，產生 8 個輸出。這一排數字叫向量；配方表則是可以訓練的線性層。

```python
import torch
from torch import nn
from tiny_perceptron.multimodal import scene, patchify

torch.manual_seed(0)
patches = patchify(scene()[None], 4)
embedding = nn.Linear(48, 8)
features = embedding(patches)
print("輸入", tuple(patches.shape))
print("輸出", tuple(features.shape))
print("矩陣", tuple(embedding.weight.shape))
```

`nn.Linear(48,8)` 建立 48 進、8 出的變換；權重矩陣 `(8,48)` 的每一行負責一個輸出。固定種子 0 讓初始化可重做。程式印出 `(1,16,48)`、`(1,16,8)`、`(8,48)`：16 個位置保留，只改每位置的寬度。將像素變成這種特徵的入口稱 patch embedding。

共享配方表示每個位置遇到相似像素時使用相同計算，不表示 8 個數字已經懂顏色。它們的用途要由訓練目標建立。文字 embedding 按字詞編號查一排數字；這裡直接計算像素亮度的加權組合，兩者的入口不同。

例如兩個亮度 `[1,0]` 和 `[0,1]`，都經權重 `[0.5,0.5]`、偏置 0 得到 0.5。這個特定配方抹掉兩者差異。48 轉 8 同樣可能壓掉資訊，訓練要尋找保留任務線索的配方；寬度變短本身不是理解的證據。

練習只把輸出 8 改成 16，預期特徵 `(1,16,16)`、矩陣 `(16,48)`。兩個 16 分別是位置數與每位置特徵數，不能因數值相同就合成一個概念。

<details>
<summary>回顧與查證</summary>

可回顧：[10.2的patch形狀](10.md#10.2)、[W.5的矩陣配方](../first-steps.md#W.5)。

</details>

