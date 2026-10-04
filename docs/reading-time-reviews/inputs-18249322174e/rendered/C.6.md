# C.6：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
import re

answers = ["3", "4", "5", "答案是4"]
truth = 4
for text in answers:
    clean = text.strip()
    parsed = int(clean) if re.fullmatch(r"-?[0-9]+", clean) else None
    reward = float(parsed == truth)
    print(repr(text), "解析值", parsed, "reward", reward)
```

```text
'3' 解析值 3 reward 0.0
'4' 解析值 4 reward 1.0
'5' 解析值 5 reward 0.0
'答案是4' 解析值 None reward 0.0

```

