## 10.2 圖片切成小塊後，如何排成序列？

沿用上一節的 16×16 黑底紅色方塊圖，要排成一串可依序處理的材料。先切成邊長 4 的小方塊，英文叫 patch：每列 4 塊，共 16 塊；依列從左到右、再到下一列排列。每塊仍保存 RGB 的 3×4×4＝48 個數值。

![十六塊按列編號並展開，每塊仍有四十八個 RGB 數值](../figures/rewrite-10-patch-order.svg)

圖中的 0–15 是小塊所在的序列位置，不是圖片類別。文字模型的一個輸入單位稱 token；在這裡，一條圖片特徵的位置來自一整塊圖，而不是塊內每個像素都變成一個字。

```python
import torch
from tiny_perceptron.multimodal import scene, patchify, unpatchify

image = scene("red", "square")[None]
patches = patchify(image, 4)
restored = unpatchify(patches, 3, 16, 16, 4)
print("小塊序列", tuple(patches.shape))
print("拼回相同", torch.equal(image, restored))
```

`scene("red", "square")[None]` 生成同一張紅色方塊圖，再增加一張圖的批次軸。`patchify` 切塊並攤平每塊，得到 `(1,16,48)`。`unpatchify` 按通道 3、高寬 16、塊邊長 4 放回格子；`torch.equal` 印出 True，說明這次往返的每個數值相同。

往返成功核對的是兩個工具是否相容，還不能單獨證明排序符合約定。設四張卡排成2×2，上排a、b，下排c、d。依列應為 `[a,b,c,d]`，切和拼都改成依欄的 `[a,c,b,d]`，仍能還原原圖。因此還要直接核對第二位置是否來自右上卡 b，真圖則核對已知位置與 RGB 通道。

把塊邊長改成 8 時，切和拼的參數都要一起改：每列 2 塊，共 4 塊，每塊 3×8×8＝192 個值，預期 `(1,4,192)` 且仍能拼回。這個切法沒有丟像素；下一節把每塊壓成短特徵時，才涉及哪些差異被保留。

<details>
<summary>回顧與查證</summary>

可回顧：[10.1的批次與RGB軸](10.md#10.1)。

</details>

