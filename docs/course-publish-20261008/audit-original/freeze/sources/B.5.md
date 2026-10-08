## B.5 同樣有數字，何時需要計算器？

「1加2等於多少」要精確數值，「加法是什麼」要概念，「原樣回答3」只要照抄。同樣有數字或加字，不代表下一步都要呼叫工具。先訂一份可核對策略，再教模型按問題與工具狀態選。

| 材料與條件 | 這份策略的動作 |
| --- | --- |
| 精確計算，工具可用 | TOOL：請求計算器 |
| 概念解釋或照抄 | DIRECT：直接回答 |
| 缺單價／數量 | ASK：補必要資訊 |
| 精確計算但工具停用 | ASK：補核對方式 |

這是教材選擇的習慣，不是所有小加法永遠只能用工具。DIRECT、TOOL、ASK是模型可學的固定標記；生成方式仍是逐token，不因選動作就變成另一種文字模型。

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

五份人工標註依次TOOL、DIRECT、DIRECT、ASK、ASK。第一和最後問題相同，只換可用狀態，動作就變；問概念有加字卻仍直接答。

只把第一筆calculator改False，程式仍印人填的TOOL，不會自動判對錯。要同時核對標註是否符合策略，再改ASK。這份求助缺的是核對方式，不是數字。下一節才讓模型學這個選擇，工具的參數與執行仍是後面的另外工作。

