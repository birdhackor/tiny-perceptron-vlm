## 4.6 每位置的八個特徵，怎樣變成候選字分數？

經過 block 後，每位置還是八個內部特徵；我們想要的卻是「下一字各候選得幾分」。因此最後再接一個八格到候選數的 Linear：每個候選各有一套權重，把八格加權成一個分數。

這個最後配方叫**輸出層**，也常叫語言模型 head；這裡的 head 是輸出功能，和注意力頭不是同一個零件。尚未變成最終預測的一排特徵叫**隱藏表示**，只是內部計算的名字。

先用二十個候選 ID 0 到 19，不賦予真實字義。輸入一筆 `[1,2,3]`，每位置都應產生二十個分數。

```python
import torch
from tiny_perceptron.model import TinyLM, ModelConfig

torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=20, width=8))
ids = torch.tensor([[1, 2, 3]])
logits = model(ids)["logits"]
print(logits.shape)
print("第一位置的前五個候選分數", logits[0, 0, :5])
print("各位置最高分ID", logits.argmax(dim=-1))
assert logits.shape == (1, 3, 20)
```

輸出 `[1,3,20]`：一筆、三個預測位置、每位置二十候選。`TinyLM` 依序接好字表、位置表、block、最後 LN 與輸出層，字典裡的 `"logits"` 是原始分數，還不是機率。

| 預測位置 | 這格可讀的前文 ID | 本例若正確續接，答案 ID |
| --- | --- | ---: |
| 0 | `[1]` | 2 |
| 1 | `[1,2]` | 3 |
| 2 | `[1,2,3]` | 4 |

三排是在答三道不同的下一字題。`argmax(dim=-1)` 沿候選軸取最高分，所以輸出 `[1,3]`，各格各有一個猜測。續寫只先用最後那排選新 ID，再把它接回輸入重算，而不是把三格猜測一次全接上。

若要用表中的三個答案算平均代價，每位置當一題，把 `[1,3,20]` 排成 `[3,20]`，答案 `[1,3]` 排成 `[3]`，再交給交叉熵：

```python
import torch.nn.functional as F

target = torch.tensor([[2, 3, 4]])
scores_for_questions = logits.reshape(-1, logits.shape[-1])
answers_for_questions = target.reshape(-1)
print("三題平均代價", F.cross_entropy(scores_for_questions, answers_for_questions).item())
```

`reshape` 只整理形狀，不改題目順序；`-1` 推算題數 3，最後候選軸仍 20。三排依次跟答案 2、3、4 比，本次初始化的平均代價約 3.336。這是隨機模型的計算示範，不能說已知道這些答案。

練習將字表大小 20 改成 25，輸入不變。輸出應 `[1,3,25]`，最高分 ID 可能因為重新初始化而變；確定增加的是每題五個候選分數。

<details>
<summary>補充與重做</summary>

原始分數與交叉熵見 [1.7](01.md#1.7)、[1.8](01.md#1.8)。本例輸入 embedding 與輸出 Linear 各有自己的可調表，沒有共享權重；後面另會討論共享。合法 ID 必須落在字表範圍，輸入 20 對二十字表會超界。這份無文字字義的 ID 例子只是介面材料，不是正文語料或訓練成果。

</details>

