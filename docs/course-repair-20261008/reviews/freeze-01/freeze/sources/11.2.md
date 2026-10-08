## 11.2 怎麼確認真正被更新的是哪一批參數？

只想教圖片接頭，就先把它的名字與數量列出，再確認優化器收錄同一批參數。接頭得到梯度、被列入更新名單、數值真的改變，是三個不同的檢查。

```python
import torch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM

model = MultiModalLM(TinyLM(ModelConfig(width=8)))
model.requires_grad_(False)
model.image_projector.requires_grad_(True)
trainable = []
parameters_to_update = []
for name, p in model.named_parameters():
    if p.requires_grad:
        trainable.append((name, p.numel()))
        parameters_to_update.append(p)
optimizer = torch.optim.AdamW(parameters_to_update, lr=0.001)
print("可訓練", trainable)
print("更新參數總數", sum(p.numel() for group in optimizer.param_groups for p in group["params"]))
```

程式先凍結整個模型，再開放 `image_projector`。`requires_grad` 控制是否為該參數累積新梯度；`named_parameters()` 提供名稱與參數，`numel()` 數數字。文字寬 8、圖片寬 16，因此權重有 8×16＝128 個值，偏置 8 個，共 136。

兩份清單收同一批參數：一份印名稱和數量，一份交給 AdamW。`optimizer.param_groups` 裡的參數數量也應為 136。這裡沒有算代價或執行 step，只確認名單，不宣稱已更新。

凍結文字權重不等於禁止它參與計算。只要計算仍被記錄，答案代價可以穿過固定文字運算，傳回接頭；若把整段文字計算放在 `no_grad()` 中，則可能切斷所需路徑。`eval()` 改變訓練模式行為，也不等於凍結。

階段切換時先決定範圍，再建優化器。若途中凍結，舊 grad 要清成 None，並排除更新名單；只改開關不會清掉歷史梯度。反過來，解凍新層也不會讓原優化器自動收它。None表示沒有梯度，本段AdamW會跳過該參數；0則是一份數值為零的梯度，仍可能受保留的歷史方向或權重衰減影響。歷史方向如何延續見[5.3](05.md#5.3)，零梯度仍縮小權重的例子見[5.5](05.md#5.5)。

練習在 `model.image_projector.requires_grad_(True)` 後、`trainable = []` 前，加入一行 `model.language.blocks[-1].requires_grad_(True)`，再重新執行本節程式。`[-1]` 選最後文字區塊；這個預設模型只有一個區塊，所以預期名單新增 `language.blocks.0`，總數由136成976。這次只核對名稱與優化器收錄的數量；[11.3](#11.3)接著檢查梯度通路，[5.1](05.md#5.1)示範step前後的數值更新。真正訓練時仍要分別核對這三件事。

<details>
<summary>回顧與查證</summary>

可回顧：[W.6梯度與更新](../first-steps.md#W.6)、[10.6接頭](10.md#10.6)、[5.17](05.md#5.17)。

</details>

