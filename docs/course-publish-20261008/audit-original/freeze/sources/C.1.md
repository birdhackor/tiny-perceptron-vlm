## C.1 寫出中間步驟能幫忙嗎？

2+3可以直接答5，也可以寫「從2開始數三次：3、4、5，所以是5」。第二份讓讀者核對方法，後續生成也能讀已寫的中間結果；但文字更長，步驟錯了也可能把結論帶錯。先分清回答形式與能力，不把長解釋自動當較可靠。

```python
from tiny_perceptron.data import ByteTokenizer

tokenizer = ByteTokenizer()
examples = [
    {"question": "2+3=?", "answer": "5"},
    {"question": "2+3=?", "answer": "從2開始數三次：3、4、5，所以是5。"},
]
for example in examples:
    ids = tokenizer.encode(example["answer"])
    print(example["question"], example["answer"], "回答token數", len(ids))
```

兩份人工答案由ByteTokenizer按UTF-8 byte編號，長1與47，沒有加角色或 EOS 結束標記（見[7.9](07.md#7.9)）。這是當前字元編碼下的長度，不是所有tokenizer通用。程式只看材料，沒有訓練或生成。

比較短答與步驟時，需要相同題目範圍、相同起點及新的數字家族，並說明配對的是題數還是token預算。舊 `(a+b)+c` 模型兩支都更新900次，但步驟讀過約9.6倍的有效回答 token，也就是訓練時計分的回答位置，含結束標記；單候選答對11/24，短答1/24；這個差別不能說成等計算量比較。

可見步驟是輸出的文字，不是模型內部計算的完整記錄。它可作為可查解法，也可能流暢卻錯。下一節用最後答對、前一步仍錯的短例，說明為何要另驗每個部分。

練習把第二份人工回答換成 `2+3=5`，先預測這五個ASCII字元在本例編成5個token，再核對長度。回答變短只改變這份材料，沒有證明模型更會算。

<details>
<summary>補充：舊兩種回答形式的比較條件</summary>

舊實驗比較 `(a+b)+c=?`，a、b、c都在0至5之間。同三個數字的所有排列共用一個家族，一起分到訓練、驗證或最後檢查，測未見組合而非新數字範圍。兩支從同一份隨機權重複製：短答只教最後整數，步驟教 `1+2=3;3+3=6;answer=6` 這類可驗算格式。

兩支都更新900次，抽到同樣數量的訓練題，但步驟回答長得多，因此有效回答token預算不同。主文的生成結果來自同一24題，每題各抽一份候選，使用相同[抽樣溫度](01.md#1.15)0.7。短答最多新增8個token，步驟最多48個，作答預算也較大。

這些條件只支持這次配方的有限比較，沒有完成相同訓練與生成token預算的比較，不能把差異全歸因於寫步驟。同起點核對、固定切分與原始生成見[正式報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/reasoning.json)；完整重做入口在[C.7](0C.md#C.7)。

</details>


