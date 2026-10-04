# A.7：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
budget = {"history": 20, "question_and_format": 15, "retrieval": 40, "answer": 25}
limit = 100
used = sum(budget.values())
print("輸入", used - budget["answer"], "答案預留", budget["answer"])
print("總預算", used, "放得下", used <= limit)
budget["history"] += 30
used = sum(budget.values())
print("歷史變長後", used, "超出", max(0, used - limit))
```

```text
輸入 75 答案預留 25
總預算 100 放得下 True
歷史變長後 130 超出 30

```

