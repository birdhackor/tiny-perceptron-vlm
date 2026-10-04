# A.1：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
question = "請依本次規則回答：3→?"
examples = ["1→2", "2→4"]
zero = question
few = "示例：" + "；".join(examples) + "。\n" + question
changed = "示例：1→3；2→6。\n" + question
print("無例子：", zero)
print("兩例子：", few)
print("換規則：", changed)
```

```text
無例子： 請依本次規則回答：3→?
兩例子： 示例：1→2；2→4。
請依本次規則回答：3→?
換規則： 示例：1→3；2→6。
請依本次規則回答：3→?

```

