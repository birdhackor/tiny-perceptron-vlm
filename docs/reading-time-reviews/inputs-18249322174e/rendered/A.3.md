# A.3：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
from tiny_perceptron.retrieval import lexical_terms, retrieve

query = "書店地址"
docs = [
    {"id": "d1", "text": "貓喜歡曬太陽"},
    {"id": "d2", "text": "書店地址是青街8號"},
]
print("查詢片段", lexical_terms(query))
hits = retrieve(query, docs, k=1)
print("取回", hits)
print("改問法", retrieve("營業處在哪", docs, k=1))
```

```text
查詢片段 ['書', '店', '地', '址']
取回 [{'id': 'd2', 'text': '書店地址是青街8號'}]
改問法 []

```

