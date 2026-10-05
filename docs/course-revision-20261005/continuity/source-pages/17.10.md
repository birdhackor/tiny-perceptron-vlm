## 17.10 文件變小，為什麼運算不一定更快？

一層64輸入、32輸出有2048個權重和32個bias。FP32數值共`2048×4+32×4=8320byte`。4-bit權重打包1024byte，32個scale128byte，bias128byte，合計1280byte。

數值儲存減少了，但本工具forward會先重建8192byte的浮點權重，再做浮點乘法。拆碼與還原帶來額外工作，低位元來源不會讓普通乘法自動變成低位元計算。

```python
import torch
from torch import nn
from tiny_perceptron.quantization import QuantizedLinear

layer = nn.Linear(64, 32)
compressed = QuantizedLinear(layer, bits=4)
x = torch.ones(2, 64)
float_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
print("原層數字bytes", float_bytes)
print("壓縮buffer bytes", compressed.storage_bytes())
print("整數碼bytes", compressed.values.numel() * compressed.values.element_size())
print("運算結果", tuple(compressed(x).shape), compressed(x).dtype)
```

前三行是8320、1280、1024；第一份含原權重與bias，第二份含碼、scale、bias，第三份只有碼。輸出`(2,32)`、FP32，表示兩筆輸入各有32特徵。這是儲存與運算介面核對，沒有在計時。

**kernel**是執行底層乘加和讀取的運算實作。專用低位元kernel可以邊讀小塊權重邊還原，或使用硬體支持的整數／混合精度路徑。是否更快取決於硬體、矩陣形狀、格式和實作；不能保證所有硬體受益。

比較成本要分別量文件大小、載入最高記憶體、執行最高記憶體，以及讀提示的prefill和逐步生成的decode時間。臨時還原矩陣也在執行成本內。計時先暖機，用同樣輸入試跑幾次不列入正式平均，再固定生成步數重複測量。

練習改bits8：整數碼2048byte、scale和bias各128，總2304byte；輸出仍FP32。這個數字不能代算速度。本教材保存的同硬體小模型實測中，逐層反量化的4-bit路徑較慢；完整量測範圍留在補充，不能外推其他後端。

<details>
<summary>補充：既有實驗、來源與完整測量範圍</summary>

[GPTQ原論文的實用加速段落](https://arxiv.org/abs/2210.17323v2)也把收益歸到專用的量化矩陣／浮點向量kernel與較少記憶體讀取，沒有把所有乘法改成整數。本工具採更直白的逐層完整反量化，也沒有實作GPTQ的權重校正。在本次L4測量，46-token提示、暖機後重複三次，FP32／4-bit／8-bit的prefill平均約2.405／4.958／2.768毫秒；固定八步、不因EOS早停的decode平均每步約2.371／4.882／2.814毫秒。這份小模型的參考4-bit路徑較慢，符合拆碼需要額外工作的可能性，不能外推成所有低位元後端都較慢。

先分清GPU記憶體的幾種計數。PyTorch的分配工具（allocator）負責拿取和重用GPU記憶體區塊；「已配置」計算它已交給張量等資料、尚未釋放的區塊。「保留」還包括暫時空出、但留著供下一次重用的區塊，所以沒有被目前張量使用，也可能仍占GPU空間。CUDA驅動程式（driver）是讓程式與GPU溝通、執行工作的底層軟體，它本身需要的空間也沒有全算在PyTorch的這組數值裡。

三版探針的CUDA已配置記憶體峰值比探針起點多335,872／376,320／359,936 bytes。這些是分配工具在本次前向期間的額外峰值；測量時其他版本仍同時留在GPU上，起點已有68,655,616 bytes。它們沒有量隔離程序的模型載入峰值、CUDA driver或所有保留記憶體，也不能用來宣稱4-bit整個執行只需376,320 bytes，或會隨檔案比例省RAM。各次探針會重設峰值計數，所以最外層報告的peak也有自己的測量範圍。

</details>

