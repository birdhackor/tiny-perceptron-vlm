## 7.11 預訓練模型怎麼學會對話？

讀文章時，模型練「原句接什麼」；當助理時，還要練「這位使用者要什麼回答」。監督式微調SFT把正確問答與角色邊界放進前文，只用回答目標繼續調參數。仍是下一token和交叉熵，不需另換一種模型才能回答。

先用0+0→0、0+1→1兩筆對話，把輸入、答案與有效位置接到模型。下面新建隨機小模型只求梯度，讓接口可見；真正微調則載入已有基模與匹配字表，再加入優化器更新。valid標出非PAD的有效輸入格；Y的非回答目標則填-100，masked_loss略過那些格的代價，只計回答bytes與EOS，沿用[7.1的答案標籤](#71-對話怎麼表示)。問題仍在輸入裡供模型讀取。

```python
import torch
from tiny_perceptron.data import toy_conversations, render_chat, pad_batch
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss

torch.manual_seed(42)
examples = [render_chat(messages) for messages in toy_conversations()[:2]]
x, y, valid = pad_batch(examples)
model = TinyLM(ModelConfig(width=8))
loss = masked_loss(model(x, valid=valid)["logits"], y)
loss.backward()
print("batch形狀", x.shape, "有效答案", (y != -100).sum().item())
print("問答接口代價", loss.item())
```

`toy_conversations` 是專案的合成加法小世界，前兩筆問題0+0與0+1、回答0與1，ASCII答案各一byte加EOS，所以有效目標共4。`pad_batch` 回傳X、Y與有效輸入表，模型處理兩筆每筆10位置，輸出每位置264個byte/結構候選。最後loss為正，backward有梯度，說明串接成功；這段沒有optimizer.step，尚未進行新訓練。

固定題目與生成規則，比較微調前後的內容、格式及舊能力。已有的小世界比較，先文字再問答的支線多了文字更新預算，不能把差異全歸因於階段形式。常見兩階段的資料成本和重用理由在本章後段展開，眼前只需記住材料從文章換成回答示範。

練習取第一筆對話，沿用[7.1的消息結構](#71-對話怎麼表示)：每項是含role與content的字典，第二項是assistant答案。以下逐項複製字典成新的消息清單，再把答案0改成3，原資料仍保留正確答案。

```python
messages = [message.copy() for message in toy_conversations()[0]]
messages[1]["content"] = "3"
exercise_x, exercise_y = render_chat(messages)
print("練習有效目標", exercise_y[exercise_y != -100].tolist())
```

`message.copy()`建立一份新字典；此處role與content都是字串，逐項複製足以讓新清單的修改保留在練習副本。`exercise_y[exercise_y != -100]`只選出要計代價的答案ID。執行前先預測原0的byte48加8=56變成3的51加8=59，輸出應為 `[59,2]`；EOS仍2，其他角色與問題不變。核對後把副本答案恢復0，避免把演示錯誤當正式訓練資料。

<details>
<summary>補充：實作約定與原始紀錄</summary>

我們已實跑兩條獨立路線。第一條從隨機模型開始，只用45筆屬性問答更新900次；第二條先把這45筆的問題與答案接成文字、練250次續寫，保存後再以對話角色與回答遮罩微調900次。第二條並沒有沿用前章的故事模型，也沒有先得到廣泛知識；它只在同一批小世界材料上多做了一個文字階段。兩條對話階段都用同樣寬度64、兩層、種子42、每批16筆與學習率0.003，最後用同一組留出題比較，成績見[7.17](#7.17)；下一節[7.12](#7.12)先用四格材料說明題目家族的切分規則。因為第二條多了250次文字更新，成績差異不能全當成「預訓練形式本身」的因果效果；要隔離它，還需另外匹配總訓練預算。

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

