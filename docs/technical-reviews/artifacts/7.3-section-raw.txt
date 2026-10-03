## 7.3 哪些位置應該學？

整理好的對話同時含問題與答案。若我們想訓練一個回答者，就常只要求它學assistant的內容，而不要求它自己寫出user會提什麼。[7.1](#7.1) 已把問題留在輸入X，現在要明確哪些Y位置可以參與代價。這份監督選擇與能讀哪些前文是不同的表。

本課label用-100表示忽略直接loss。有效監督條件是 `label!=-100`，不是查看輸入token是否屬於user。因為 [1.3](01.md#1.3) 的shift，第t位置輸入預測原序列t+1，標籤也要對到下一項；不能用當前字所屬角色直接當作該格答案角色。下一節會專門追首答案位置。

```python
from tiny_perceptron.data import render_chat

messages = [
    {"role": "user", "content": "問"},
    {"role": "assistant", "content": "答"},
]
x, y = render_chat(messages)
for position, (token, label) in enumerate(zip(x.tolist(), y.tolist(), strict=True)):
    status = "學下一項" if label != -100 else "忽略直接loss"
    print(position, "輸入ID", token, "目標", label, status)
print("有效目標數", (y != -100).sum().item())
assert (y != -100).sum().item() > 0
```

輸入一字中文問題與一字中文答案。ByteTokenizer以byte表示，問與答各三token，因此不能按一字就數一個有效目標，見 [6.7](06.md#6.7)。程序逐位置列出當前輸入ID、右移後label與是否計代價，最後有效數應4：答的三個byte與回答EOS。user內容、user段EOS與角色標記本身不作為要模仿的答案。

迴圈中的 `status = ... if ... else ...` 是條件表達式：label不是-100時選左側文字，否則選右側。`(y!=-100)` 得真/假tensor，sum把True當1數有效目標，item取成普通整數。

模型讀的是輸入X：它拿X中的合法ID，到一張數值表查出起始特徵。這個動作叫嵌入查表，英文embedding，見[4.1](04.md#4.1)。目標Y則交給交叉熵計算代價，其中的-100只表示「這個位置不計分」。-100不會拿去查輸入表，也不是合法文字ID；X中對應的問題文字仍會正常送進模型。

不算某位置loss，表示不直接讓該位置的logits承擔答案代價；這段輸入仍可供後面回答讀取。我們還要用 [3.6](03.md#3.6) 因果遮罩限制未來，不因為user不計loss就刪除問題。若刪除問題，模型反而沒有問答關聯的條件，只能學常見回答。

不同訓練目標可以監督全部對話或只部分角色，但必須說清楚目標與分母。這裡只學assistant，通常使訓練資源集中在回答內容與結束方式；它不能自動篩掉錯標答案，品質控制仍是資料工作。

練習只把assistant內容改成 `"答案"`，先預測兩中文共六byte，加EOS共七有效目標，再執行核對。輸入也會變長，而忽略的user問題仍存在。再試英文 `"OK"`，應兩byte加EOS，共三；計數單位隨編碼規則，不隨肉眼字數。

