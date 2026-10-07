## 17.7 只量化權重時，輸入與運算用什麼？

先固定兩列權重`[0.7,0.2]`、`[−0.7,1.2]`，輸入`[1,1]`，原輸出是`[0.9,0.5]`。將權重按每列4-bit刻度保存，輸入仍保留浮點數。這叫**只量化權重（weight-only quantization）**，暫時不用估每種輸入的中間特徵範圍。

第一列刻度0.1，兩項精確；第二列刻度1.2/7，−0.7還原約−0.6857，所以壓縮輸出約`[0.9,0.5143]`。下面固定輸入，核對這個改變。

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

`QuantizedLinear`保存低位元碼與逐列scale。4個四位元值打包成2byte，兩個FP32 scale另佔8byte，合計10。沒有bias，讓計數只包含這兩部分。

第一行存碼是`torch.uint8`，輸出是`torch.float32`。因為教材的forward先拆碼、乘scale還原浮點權重，再做普通浮點線性運算。**存成整數與用整數乘法是不同的事**，輸入和輸出在這裏都還是FP32。

這些保存但不交給一般梯度更新的資料叫buffer。轉換版用於推理核對，不能直接當作沿用原更新器續訓的配方。每次還原也可能建立暫存浮點矩陣，所以存得小仍需另量執行記憶體與速度。

練習只把bits改8。四碼各1byte，兩scale仍8byte，合計12；存碼變int8，輸出仍FP32。固定權重與輸入，才能比較位元數。

<details>
<summary>補充：既有實驗、來源與完整測量範圍</summary>

本次整模型轉換保留文字與位置嵌入、各正規化層的FP32參數，合計102,912 bytes；bias也維持FP32，但隨被替換的Linear存進buffer。這些選擇在[實報的`retained_float_modules`](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/quantization.json)與逐層buffer清單中明示，所以只說「4-bit模型」並不代表每個數字都只有四位元。

</details>

