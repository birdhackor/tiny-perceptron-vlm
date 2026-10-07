## 12.8 每個時間框，如何變成一條聲音特徵？

一段 0.1 秒單音經 log-mel 處理，得到 16 個頻帶、11 個時間框。現在希望每框用 8 個數字表示，依先後排成 11 條特徵，供後面的模型逐位置讀取；這一步不把整段平均成一個數字。

![每個時間框的16個頻帶值經共享入口變成8個特徵，先後仍保留](../figures/rewrite-12-frame-features.svg)

```python
from tiny_perceptron.multimodal import tone, log_mel, AudioEncoder

wave = tone()[None]
spectrogram = log_mel(wave)
encoder = AudioEncoder(bands=16, width=8)
features = encoder(wave)
print("頻譜", tuple(spectrogram.shape))
print("時間特徵", tuple(features.shape))
```

`tone()` 先產生一維波形；`[None]` 在最前面新增大小為 1 的軸，讓輸入表示一筆聲音。獨立算出的 `spectrogram` 用來列印並對照形狀；`encoder(wave)` 接收波形後，會在內部先算 log-mel，再調軸並套用逐框入口。程式印出頻譜 `(1,16,11)`、時間特徵 `(1,11,8)`。第一軸的 1 是一段聲音；頻譜表先按「頻帶、時間」保存，入口轉成「時間、頻帶」，讓同一組可訓練配方對每框的 16 個值產生 8 個輸出。本入口先做線性投影，再經正規化、另一線性層與非線性函數GELU，這些運算都保持時間位置。11 個位置仍代表時間框，不是 11 個字。

16 轉 8 是每框的摘要，與把 11 框全部平均掉不同。前者仍能表示哪種變化先來；後者可能只留整體多寡。配方共享表示每框使用同一入口，不表示隨機初始化的 8 個數字已經認得聲音。

入口要從任務材料學出有用特徵。單音任務可以教「頻率高於 300 Hz 回 high，否則 low」；真人短句的有限需求則需要真人錄音與需求標註。兩種答案不是同一件事：前者只驗證單音高低，不能據此宣稱聽懂中文問句。

下一節先用單音檢查接到文字核心後的梯度通路；[12.15](12.md#12.15) 再設計真人短句意圖入口。這樣頻譜轉特徵是共同機制，各任務仍有各自材料與判準。

練習只把入口 width 改成 12，時間特徵應為 `(1,11,12)`，原本 log-mel 頻帶仍為 16。若想用 32 帶，前處理與入口的 bands 都要一致，不能只改一邊。

<details>
<summary>回顧與查證</summary>

可回顧：[12.7的log-mel](12.md#12.7)、[10.3的共享線性變換](10.md#10.3)、[W.3軸順序](../first-steps.md#W.3)。

原實驗的資料切分、更新設定與逐題結果見[既有實報](../../docs/course-experiments/results/encoders.json)；重做入口見[實驗說明](../../docs/course-experiments/README.md)。這是上述有限任務的歷史紀錄。

</details>

