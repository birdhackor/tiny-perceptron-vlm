# C.4：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 4

```python
from collections import Counter

candidates = [3, 3, 4]
majority = Counter(candidates).most_common(1)[0][0]
verified = next((x for x in candidates if x == 2 + 2), None)
print("集合包含正解", 4 in candidates)
print("多數決", majority, "答對", majority == 4)
print("驗證器", verified, "答對", verified == 4)
```

```text
集合包含正解 True
多數決 3 答對 False
驗證器 4 答對 True

```

