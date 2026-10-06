## 7.4 回答第一個 token 在哪個位置被預測？

問題Q，回答A。完整序列是BOS、user、Q、EOS、assistant、A、EOS；要教模型寫第一個A，代價應落在assistant格，因為那格正在猜下一項。

兩字都是ASCII各一byte，Q的ID89、A的ID73。X去掉原序列末項，Y把原序列的下一項答案配到當前輸入格：原序列索引5的A，配到Y索引4，作為X索引4的assistant格所猜的答案；原索引6的EOS則配到Y索引5。也就是把答案配到前一格的預測位置。圖把位置索引、輸入ID和目標ID分欄，先看索引4的assistant→A，再看索引5的A→EOS。

![首答案由assistant格預測，索引和token ID各有用途](../figures/rewrite-07-04-answer-alignment.svg)

```python
from tiny_perceptron.data import ByteTokenizer, render_chat

tok = ByteTokenizer()
x, y = render_chat(
    [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
)
first = (y != -100).nonzero()[0].item()
print("X", x.tolist(), "Y", y.tolist())
print("首答案預測位置", first, "輸入", x[first].item(), "答案", y[first].item())
assert x[first].item() == tok.assistant_id
assert y[first].item() == tok.encode("A")[0]
```

輸入兩消息，Q的ASCII81加8為89，A的65加8為73。輸出X是 `[1,3,89,2,4,73]`，Y是四個-100後接73、2。`nonzero()` 找出條件為True的位置，取第一個得到first=4；該位置X=4，即assistant，而Y=73，即A。下一格X=73則負責預測EOS2。原句索引、當前輸入ID與目標ID，三種數字的用途要分開。

有效目標數對，不保證位置對。mask是哪些目標位置要計代價的標記，也要隨答案配到前一格的預測位置；若還留在原答案所在格，就可能讓首代價落在已讀到A的格。應同時查「X首有效格是assistant、Y是首回答byte」。`render_chat` 已完成這個配對，TinyLM和loss按同一位置比較，不能再移一次。

練習把問題Q改成 `"QQ"`，先預測首答案位置從4變5，但X[first]仍4、Y[first]仍73，再執行核對。接著只把答案A改成B，first仍5、答案ID變74。兩次分別改變前文長度與回答內容，可看清「首目標在哪」取決於哪些材料。

<details>
<summary>補充：實作約定與原始紀錄</summary>

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

