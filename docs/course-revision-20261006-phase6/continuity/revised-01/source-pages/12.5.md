## 12.5 怎麼找出一段單音的主要頻率？

仍用 440 Hz 單音，這次長 0.2 秒。希望每個短時間框都能指出「哪些頻率較強」，並核對最強的格子是否對應 440 Hz。把時間框分解成各頻率的計算叫短時傅立葉轉換，簡寫 STFT。

![頻率沿縱軸、時間框沿橫軸；亮度代表各格能量的示意](../figures/multimodal_audio_axes.svg)

```python
import torch
from tiny_perceptron.multimodal import tone

wave = tone(440.0, seconds=0.2)
spectrum = torch.stft(wave, 400, 160, window=torch.hann_window(400), return_complex=True)
power = spectrum.abs().square()
peak = power.mean(-1).argmax().item()
print("頻率格、時間框", tuple(power.shape))
print("最高格編號", peak, "對應Hz", peak * 16000 / 400)
```

400 點一框、每次移 160 點，輸出功率 `(201,21)`：201 個非負頻率格、21 個時間框。函式預設在邊界補資料，使框中心落在相應時間點；所以它與直接 `unfold` 的完整框數不同。這裡 3200 點得到 21 框，不是沒有補齊的 18 框。

STFT 結果是複數，含頻率成分的大小與相位。`abs()` 取大小，再平方得到功率；例如 `3+4j` 的大小是 5，功率是 25。圖用功率亮度表達，不直接顯示相位。

頻率格間隔為 `16000／400＝40 Hz`，第 11 格對應 440 Hz。把各框功率平均後找最大，程式印出 11 與 440.0，符合這個單音的來源。真人語音含多種變化，最強一格只能描述當下能量，不能當成某個字的編號。

頻譜也不是音高列表：圖上不同頻率可能同時亮起。練習改成 480 Hz 單音，預期最大格約 12；若用 450 Hz，則不能要求它剛好落在整格，附近格會一起分到能量。

<details>
<summary>回顧與查證</summary>

可回顧：[12.2取樣率](12.md#12.2)、[12.3時間框](12.md#12.3)、[12.4加窗](12.md#12.4)。

</details>

