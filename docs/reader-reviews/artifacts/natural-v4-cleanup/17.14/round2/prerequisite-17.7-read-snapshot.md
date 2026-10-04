## 17.7 先只量化權重可以嗎？

權重在模型載入後固定，中間特徵卻隨每個輸入變動。要先體驗壓縮，可以只把權重存成低位元表示，而保留輸入與運算使用浮點數，這叫weight-only quantization（只量化權重）。前置是[逐列刻度](17.md#17.5)和[權重誤差傳遞](17.md#17.3)。這個選擇省掉目前對中間特徵範圍的判斷，但仍需要核對輸出。

用兩列權重 `[0.7,0.2]`與 `[-0.7,1.2]`，各列使用自己的四位元scale。第一列最大0.7，刻度0.1，兩項剛好落在整數7與2；第二列最大1.2，刻度1.2/7，-0.7會還原成約-0.6857。因此對輸入 `[1,1]`，原輸出 `[0.9,0.5]`，壓縮版本約 `[0.9,0.5143]`。我們保持輸入不變，才能把差異歸到同一份權重的轉換。

```python
import torch
from torch import nn
from tiny_perceptron.quantization import QuantizedLinear

layer = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[0.7, 0.2], [-0.7, 1.2]]))
compressed = QuantizedLinear(layer, bits=4)
x = torch.tensor([[1.0, 1.0]])
print("儲存碼型別", compressed.values.dtype, "輸出型別", compressed(x).dtype)
print("原輸出", layer(x).detach().round(decimals=4).tolist())
print("壓縮輸出", compressed(x).round(decimals=4).tolist())
print("壓縮buffer bytes", compressed.storage_bytes())
```

`QuantizedLinear`接收已經有權重的線性層，依每列刻度轉換，保存整數碼與scale。本例bits=4時，`values`是打包後的uint8容器，四個四位元數放成兩bytes；打包與拆回的細節在[下一節](17.md#17.8)。bits=8時則把每個帶正負號的整數碼各存成一個int8，不把兩碼合進一byte。scale兩個FP32數占八bytes，因此本例最後一行應為10，不是只看整數碼的2。這裡沒有bias，避免額外偏移影響計數。

第一行應是torch.uint8與torch.float32。這個對照說明「存成整數」不代表「矩陣乘法用整數」。教材的參考forward先拆碼、乘scale還原浮點權重，再用普通浮點線性運算；中間特徵x與輸出仍用FP32。原輸出和壓縮輸出應對上手算，第二項差約0.0143；型別本身並不能說明品質好壞。

buffer指模組保存但不交給一般梯度更新的資料，和普通Parameter不同。這份轉換以推理檢查為主，不是直接接原優化器繼續訓練的配方。載入時也要保留bits、shape與scale等還原資訊，不能只保存 `values`就期待任何層都知道如何拆解。

本次整模型轉換保留文字與位置嵌入、各正規化層的FP32參數，合計102,912 bytes；bias也維持FP32，但隨被替換的Linear存進buffer。這些選擇在[實報的`retained_float_modules`](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/quantization.json)與逐層buffer清單中明示，所以只說「4-bit模型」並不代表每個數字都只有四位元。

此參考實作可能在每次forward建立臨時浮點權重，所以存儲較小不保證更快，也不保證最高執行記憶體按相同比例下降。真正利用低位元資料的專用運算要看硬體支援；這裡先得到一個可讀、可核對的輸出基準。

練習只把bits改成8，先預測第一行的儲存碼型別改為torch.int8，輸出仍是torch.float32；四個整數碼占四bytes，加兩個scale共12bytes。再執行核對：本固定例的壓縮輸出約為 `[0.8984,0.5008]`，比四位元更接近原輸出。原layer與x保持不變，讓你只比較位元數的影響。

