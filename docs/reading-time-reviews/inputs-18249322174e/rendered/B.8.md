# B.8：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
direct_accuracy = 0.90
tool_accuracy = 0.99
wrong_cost = 10.0
tool_extra_cost = 0.02
direct_loss = (1 - direct_accuracy) * wrong_cost
tool_loss = (1 - tool_accuracy) * wrong_cost + tool_extra_cost
print("直接回答平均損失", round(direct_loss, 3))
print("工具流程平均損失", round(tool_loss, 3))
print("本例選擇", "TOOL" if tool_loss < direct_loss else "DIRECT")
```

```text
直接回答平均損失 1.0
工具流程平均損失 0.12
本例選擇 TOOL

```

