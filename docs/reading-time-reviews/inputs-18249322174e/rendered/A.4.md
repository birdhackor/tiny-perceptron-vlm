# A.4：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
source = {"id": "d2", "text": "書店地址是青街8號"}
answers = [
    {"address": "青街8號", "citation": "d2"},
    {"address": "紅街9號", "citation": "d2"},
]
for answer in answers:
    valid = answer["citation"] == source["id"]
    supports = valid and answer["address"] in source["text"]
    print(answer["address"], "引用存在", valid, "地址在原文", supports)
```

```text
青街8號 引用存在 True 地址在原文 True
紅街9號 引用存在 True 地址在原文 False

```

