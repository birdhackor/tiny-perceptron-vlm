## 18.12 MoE 教師能教 Dense 學生嗎？

教師內部有三位expert，學生只有一套Dense FFN，是否要先把教師所有專家搬進學生？如果我們比較的是下一個token分布，不需要。教師的內部路由怎樣得到結果，是它自己的運算；學生只需在相同前文後，產生相同候選意義下的分布。我們沿用[相同候選身份](18.md#18.6)和[KL目標](18.md#18.8)，先確認不同內部結構仍輸出同一種答案分數。

例如教師寬度16、三位expert每次選兩位，學生寬度8、一套Dense。兩邊內部特徵分別16與8，不能直接逐項相減；但詞表都264項，讀相同兩個token時，輸出都是 `[1,2,264]`。這個共同輸出介面讓分布蒸餾可行，不要求同樣層數、相同隱藏寬度或同樣專家數量。

```python
import torch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.alignment import distillation_kl

torch.manual_seed(0)
teacher = TinyLM(ModelConfig(width=16, experts=3, top_k=2)).eval().requires_grad_(False)
student = TinyLM(ModelConfig(width=8))
ids = torch.tensor([[1, 2]])
with torch.no_grad():
    t = teacher(ids)["logits"]
s = student(ids)["logits"]
labels = torch.tensor([[1, 2]])
loss = distillation_kl(s, t, labels, temperature=1.0)
loss.backward()
print("教師/學生分數形狀", tuple(t.shape), tuple(s.shape))
print("總參數", teacher.description()["parameters"], student.description()["parameters"])
print("學生輸出權重有梯度", student.output.weight.grad.abs().sum().item() > 0)
```

輸入是同一段ID，兩模型詞表沿用相同預設配置。teacher內部用MoE，student用Dense；`distillation_kl`只接分數與有效位置標記，不讀內部expert索引。這個人工標籤把兩格都設有效，只檢驗資料介面與梯度，不模擬完整對話遮罩。應輸出相同 `(1,2,264)`，總參數18048與6104，學生輸出權重有梯度True。

兩模型都隨機初始化，此例證明能構成學習路徑，沒有證明教師有數學或語言能力，更沒有完成真正蒸餾。正式實驗需載入已驗證教師，保留同學生架構的未蒸餾基準，再比較任務品質與成本。教師的MoE總參數、每token啟動expert數與實際計算時間也應分別記錄，不能因每次只選兩位就把第三位權重儲存算成零。跨架構蒸餾是一條可選路線，不要求Dense成品一定經過MoE教師。

學生更小會限制能學到的細節。即使候選分布對齊，Dense學生仍可能無法在所有輸入近似教師，尤其是教師不同expert形成多種複雜規則時；這需要獨立評估，不能由shape相同推論能力等同。反過來，學生沒有router，也不代表它一定更快，實際耗時仍受硬體與輸入條件影響。

練習只將學生vocab_size設成265。先預測學生分數最後一軸變265、教師仍264，KL輔助函式`distillation_kl`會拒絕形狀不一致，再執行觀察錯誤。把它改回264之後路徑恢復；候選欄位先對上，內部結構才有自由。

<details>
<summary>選讀：既有對照與資料預算限制</summary>

正式支線載入[15.13預先指定的MoE教師](15.md#15.13)，比較兩個同架構、同初始化的Dense學生。普通CE學生以原故事文字為真值，用交叉熵提高真實下一個token的機率；CE+KL學生保留同一真值目標，再加入教師候選分布。這裡的CE目標是故事原文，與[18.8](18.md#18.8)用教師軟目標說明CE與KL關係的例子不同。

受資料預算限制，兩個學生都只讀前32篇訓練故事，教師原先讀完整409篇。因此教師與學生的差距還包含資料預算；能隔離教師訊號的是兩個讀相同資料的學生。這輪學生續寫仍重複片語，不能把平均預測代價改善當成已會寫故事。完整配方、逐篇原始ID與留出評估保存在[蒸餾實報的MoE任務](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/distillation.json)。

</details>

