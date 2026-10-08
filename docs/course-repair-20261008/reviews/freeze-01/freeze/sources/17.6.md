## 17.6 少四個位元，精度與容器各改了什麼？

取33個從−1到1均勻排列的權重，用同一份材料分別做8-bit與4-bit對稱量化。本工具的正碼上限是127或7，所以刻度分別約0.007874與0.142857。範圍相同，4-bit格子較粗。

```python
import torch
from tiny_perceptron.quantization import quantize_symmetric

w = torch.linspace(-1, 1, 33)
for bits in (8, 4):
    q, scale = quantize_symmetric(w, bits)
    restored = q.float() * scale
    error = (w - restored).abs()
    packed_bytes = (w.numel() * bits + 7) // 8
    print("bits", bits, "MAE", round(error.mean().item(), 6), "最大差", round(error.max().item(), 6))
    print("理想整數bytes", packed_bytes, "目前q容器bytes", q.numel() * q.element_size())
```

`linspace`含兩端共33項。第一行同時看MAE與最大差：8-bit約0.001909、0.003937，4-bit約0.034632、0.071429。最大差讓少數較糟位置不會被平均藏住。

第二行分開「理想碼的byte」與「目前容器byte」。8-bit33碼需33byte；4-bit共132bit，要17byte，最後半byte空著。`(總bit+7)//8`用整除向上補足。可是函式回傳q都用int8，兩次目前仍佔33byte；限制數值範圍與真正打包是兩步，scale也尚未加進理想碼大小。

較粗不代表每個值都更差：零與端點可能都精確，恰落4-bit格子的數也可能比8-bit近。更不能把單表誤差直接當文字答對率。

練習把w改成`torch.arange(-7,8).float()/7`。15個值就在4-bit格上，預測它的誤差幾乎零。這個例外提醒我們要看數值分佈，不用位元名稱代替檢查。

