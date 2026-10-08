## 17.15 平均誤差很小，答案為什麼仍會改變？

只有兩候選EOS與「貓」。原分數`[1,1.005]`選貓；量化後變`[1.01,1.005]`，改選EOS，即停止回答。平均絕對差僅`(0.01+0)/2=0.005`，卻跨過了第一名邊界。

```python
import torch

tokens = ["<EOS>", "貓"]
original = torch.tensor([1.0, 1.005])
quantized = torch.tensor([1.01, 1.005])
mae = (original - quantized).abs().mean().item()
print("分數MAE", round(mae, 4))
print("原版選擇", tokens[original.argmax().item()])
print("量化版選擇", tokens[quantized.argmax().item()])
```

`argmax`取最大分數索引，再查候選名稱。輸出0.005、貓、EOS。這是手設分數的單步反例，沒有證明真模型必定提早停；它說明平均數不能保證離散選擇不變。若兩版選到不同的詞而仍繼續生成，下一步讀入的已生成回答也會不同，後續差異還可能累積。

真正比較讓原版與量化版讀同樣提示，用同樣輸入處理和生成設定。既看標準答案的預測代價，也實際生成，分開核對內容、範圍、格式和正常EOS。若用隨機抽樣，兩版配對設相同種子，另換幾個種子觀察；最高分生成則記爲greedy。

品質檢查要與模型檔案大小、記憶體和速度一起報告。17.14的QAT比較是另一份共同起點與更新預算，不能將不同更新步數的PTQ拿來當作只改變QAT的對照。

練習將兩版第二候選分數都改2。MAE仍0.005，卻都選貓。相同誤差在不同排名餘裕下，可以有不同後果。

<details>
<summary>補充：操作入口與既有結果</summary>

量化操作入口見[T.9](../training.md#T.9)，既有weight-only PTQ的配方與逐題記錄見[量化實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/quantization.json)，QAT的共同起點與對照方式見[17.14](17.md#17.14)。這些實驗只測合成屬性問答，不能替代JSON、日期澄清或安全拒絕的品質檢查。

</details>
