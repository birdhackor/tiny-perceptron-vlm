# B.5：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 4

```python
examples = [
    {"question": "1加2等於多少？", "calculator": True, "action": "TOOL"},
    {"question": "請解釋加法的意思。", "calculator": True, "action": "DIRECT"},
    {"question": "請原樣回答數字3。", "calculator": True, "action": "DIRECT"},
    {"question": "請算總價。", "calculator": True, "action": "ASK"},
    {"question": "1加2等於多少？", "calculator": False, "action": "ASK"},
]
for row in examples:
    print(row["question"], "計算器可用", row["calculator"], "→", row["action"])
```

```text
1加2等於多少？ 計算器可用 True → TOOL
請解釋加法的意思。 計算器可用 True → DIRECT
請原樣回答數字3。 計算器可用 True → DIRECT
請算總價。 計算器可用 True → ASK
1加2等於多少？ 計算器可用 False → ASK

```

