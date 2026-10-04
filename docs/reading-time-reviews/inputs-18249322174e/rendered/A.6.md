# A.6：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
from tiny_perceptron.retrieval import retrieve

query = "書店地址"
cases = [
    [{"id": "d1", "text": "貓在窗邊"}],
    [{"id": "d2", "text": "書店地址尚未公布"}],
]
for docs in cases:
    hits = retrieve(query, docs)
    if not hits:
        decision = "沒有取回相關來源，請補公告"
    elif "尚未公布" in hits[0]["text"]:
        decision = "來源未給地址，目前無法確定"
    else:
        decision = "需進一步核對來源與答案"
    print([d["id"] for d in hits], decision)
```

```text
[] 沒有取回相關來源，請補公告
['d2'] 來源未給地址，目前無法確定

```

