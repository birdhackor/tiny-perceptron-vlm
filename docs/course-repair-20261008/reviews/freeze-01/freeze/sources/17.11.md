## 17.11 中間特徵也量化，多了哪一種誤差？

用兩筆輸入x與權重w，都取兩列`[0.7,0.2]`、`[−0.7,1.2]`。先保留x、只改w，再把兩邊都映到刻度0.5的格子。希望看清：第二次比較新增的，是輸入特徵的量化誤差。

這裡**activation**指層的輸入或中間特徵，不只指GELU等啟動函數。權重固定，這些特徵卻會跟輸入內容改變。

```python
import torch

x = torch.tensor([[0.7, 0.2], [-0.7, 1.2]])
w = torch.tensor([[0.7, 0.2], [-0.7, 1.2]])
scale = 0.5


def simulate_quantization(t):
    return (t / scale).round() * scale


qx, qw = simulate_quantization(x), simulate_quantization(w)
reference = x @ w.T
weight_only = x @ qw.T
both = qx @ qw.T
print("原輸出", reference.round(decimals=4).tolist())
for name, output in (("只改權重", weight_only), ("權重加特徵", both)):
    print(name, output.round(decimals=4).tolist(), "MAE", round((reference - output).abs().mean().item(), 4))
```

`simulate_quantization`量化後立即還原，仍回傳浮點tensor，沒有打包或低位元kernel。刻度固定是爲了隔離誤差，真實權重與特徵未必用同一把scale。

原輸出約`[[0.53,−0.25],[−0.25,1.93]]`。只改權重的MAE0.19，兩邊都改則0.24。看看非對角兩格：只改權重時有誤差，雙邊量化卻剛好回到−0.25；最後一格1.93變1.25則更差。新增誤差不代表每項一定惡化，有時方向偶然抵消。

完整任務仍需檢查，而且中間範圍可能隨句子長度、圖像或音量改變。既有整模型PTQ與QAT實驗只改Linear權重，沒有量化activation，因此那些得分不能當作本節雙邊方法的成績。

練習只將x第一列改`[1,0]`。它已落0.5格子上，所以只改權重與雙邊量化的第一列相同，都是`[0.5,−0.5]`。

<details>
<summary>補充：既有實驗、來源與完整測量範圍</summary>

本節與接下來的範圍、極端值例子是CPU上可核對的數值機制。本章已完成的整模型量化與[QAT實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/qat.json)都只量化Linear權重，`activation_quantization`為false；它們沒有跑中間特徵校準、截斷或低位元activation訓練，不能把那些GPU分數當作本節方法的成績。

</details>

