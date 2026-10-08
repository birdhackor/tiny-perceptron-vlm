## 17.4 碼0一定代表浮點零嗎？

四位元有16種碼。若想覆蓋−1到2，可以將整數碼0至15均勻放在這段範圍：15個間隔，每格0.2。碼0代表−1，碼15代表2；浮點零在碼5，而不是碼0。

「代表浮點零的整數碼」叫**zero-point（零點）**。scale決定格距，zero-point決定零的位置。保存碼之後，還原公式是`(碼−zero)×scale`。

```python
import torch

x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
low = min(x.min().item(), 0.0)
high = max(x.max().item(), 0.0)
scale = (high - low) / 15
zero = round(-low / scale)
q = (x / scale + zero).round().clamp(0, 15).to(torch.uint8)
restored = (q.float() - zero) * scale
print("scale/zero", scale, zero)
print("整數碼", q.tolist())
print("還原", restored.round(decimals=4).tolist())
```

`min(...,0)`和`max(...,0)`讓選定範圍包含浮點零。此次low−1、high2，scale0.2、zero5；得到碼`[0,5,10,15]`，還原`[−1,0,1,2]`。這四項剛好在格上，所以沒有捨入損失；一般數字仍會有。

這種帶刻度與偏移的方法叫**仿射量化（affine quantization）**。程式用uint8，即一byte不帶負號的容器，只使用其中0至15；尚未把兩碼打包成一byte。全零範圍還需要特別處理scale，本例只示範非零寬度。

後面的打包器使用−8至7的有號值，另映成儲存碼。碼、零點與實際浮點值必須按各自規則對齊，不能只看「4-bit」就混用。

練習把x改成`[−2,0,1]`。範圍寬仍3，scale仍0.2，zero改10，碼變`[0,10,15]`。這正好分開格距與零位置兩份工作。

