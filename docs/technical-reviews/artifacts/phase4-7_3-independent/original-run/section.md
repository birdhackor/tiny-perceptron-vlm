## 7.3 哪些位置應該學？

「問→答」這筆對話，需要模型讀到「問」，但我們希望它練習寫的是「答」和回答結束。把問題留在輸入X，再分開標出哪些下一項答案要計代價，就能同時做到這兩件事。

Y中的-100表示忽略直接答案代價，不是輸入token。因X每格預測下一項，監督也要同樣右移；以下先印每格的輸入ID、目標與是否計分，下一節再定位首答案。

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

模型讀X的合法ID並查輸入特徵表，交叉熵讀Y的答案或-100；兩份表的用途不同。忽略user的直接代價不會刪問題，也不會禁止後面注意力讀它。只學assistant可以集中回答監督，仍需另外核對示範是否正確。

練習只把assistant內容改成 `"答案"`，先預測兩中文共六byte，加EOS共七有效目標，再執行核對。輸入也會變長，而忽略的user問題仍存在。再試英文 `"OK"`，應兩byte加EOS，共三；計數單位隨編碼規則，不隨肉眼字數。

<details>
<summary>補充：實作約定與原始紀錄</summary>

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

