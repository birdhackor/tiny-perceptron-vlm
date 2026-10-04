# C.1：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

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

```text
2+3=? 5 回答token數 1
2+3=? 從2開始數三次：3、4、5，所以是5。 回答token數 47

```

