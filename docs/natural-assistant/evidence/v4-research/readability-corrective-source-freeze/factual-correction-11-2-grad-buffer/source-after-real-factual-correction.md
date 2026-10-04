## 11.2 哪些參數正在學習？

你打算只訓練圖片接頭，卻發現整個語言模型也在改變；或者參數有梯度，數值卻一直不動。訓練的範圍不能靠一句「已經凍結」確認：先決定哪些參數要接收新的梯度，再核對優化器收了哪些參數，最後檢查實際更新。本節先把訓練名單印出來，避免在錯誤範圍上花時間。

前置是[W.6梯度與更新](../first-steps.md#W.6)及[10.6接頭](10.md#10.6)。`requires_grad=False`表示不為該參數累積新的梯度，稱為凍結；optimizer是執行參數更新的優化器，只能更新交給它的參數。改這個開關不會清掉已經存在的`.grad`；若參數仍在優化器裡，AdamW仍可能使用留下的梯度更新它。`eval()`調整模型中的訓練行為，不等於凍結，詳見[5.17](05.md#5.17)。

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

輸入是文字寬度8的多模態模型，視覺寬度預設16。先凍結整個模型，再只開放`image_projector`。`named_parameters()`逐項提供一對內容：參數名稱與那份參數本身。`for name, p`把這一對分別收進`name`與`p`；名稱是字串，p是裝著可調數字的tensor。`if p.requires_grad`只讓開關為True的項目進入縮排內的兩行。`append`在清單尾端增加一筆：`trainable`保存名稱和數量，供我們閱讀；`parameters_to_update`保存p本身，交給優化器更新。`numel()`數出p裡有幾個數字。

因此輸出名單為`image_projector.weight`有128個數、`image_projector.bias`有8個數，總計136。權重128來自8×16，偏置是每個輸出多加一個數。兩份清單收的是同一批參數，沒有因為印出名稱就另造一份待更新權重；它們分別負責讓人核對與讓優化器實際取用。

創建AdamW時，清單再次只收`requires_grad`為True的參數；學習率0.001是本例設置的更新步長。`optimizer.param_groups`是優化器保存的分組清單，每組是一個字典，`group["params"]`取出該組的參數清單。最後的兩個`for`先走訪各組，再走訪組內每個參數p；`p.numel()`數出它有幾個數字，`sum`把這些數量加總成136，驗證聲明與真正收錄的參數相符。清單、字典和迴圈的寫法可回看[W.2](../first-steps.md#W.2)。程式還沒有計算答案代價或執行`step()`，所以不能說136個數字已經改變。

凍結編碼器或語言權重，也不一定切斷對接頭的訓練信號。只要計算仍被記錄，代價的梯度可以穿過固定運算回到接頭，類似固定齒輪仍能傳遞動作；凍結的是齒輪尺寸的調整。若對整段語言計算使用`no_grad()`，則可能截斷這條路徑，下一節會檢查接頭是否收到梯度。

正式模型也要核對可訓練名單、收到的梯度與更新前後差值；[11.3](11.md#11.3)把這些執行證據和描述結果放在一起，避免把「真的更新」等同於「已學會」。

還有一個容易漏掉的順序：先決定凍結範圍，再建立優化器。如果在訓練途中改成凍結，還要把該參數的舊梯度設成`None`，並將它排除在這一階段的優化器名單之外；只改開關，原清單與舊梯度都可能留下。如果建立時只收接頭，後來解凍語言層，清單又不會自動補上。最好在每一階段開始印出兩份名單並核對，參數有梯度後再看一次更新前後差值。梯度為0與未收到梯度也不同：前者可以是當前資料剛好不敏感，後者None可能表示它根本沒參與鏈路。對AdamW來說，0也不等於跳過更新，因為過去的更新紀錄與權重衰減仍可能使數值改變。這些資訊比單看「訓練中」的字樣具體。

練習在列名單前增加`model.language.blocks[-1].requires_grad_(True)`，只開放最後一個語言區塊。`model.language`是包裝內的文字模型，`blocks`保存它的一組組可調層，每組稱為區塊；索引`[-1]`取最後一組。本例預設只有一個區塊，編號為0，所以新增名稱以`language.blocks.0`開頭。先預測名單會新增這組參數，優化器總數大於136，再重跑核對本例總數976。不要僅在創建優化器之後解凍：那會讓新參數能算梯度，但原優化器尚未收錄它們。

