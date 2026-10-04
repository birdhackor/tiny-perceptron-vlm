# C.2：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
steps = [(2, 2, 5), (5, -1, 4)]
truth = 2 + 2
valid = [a + b == claimed for a, b, claimed in steps]
linked = [steps[i][0] == steps[i - 1][2] for i in range(1, len(steps))]
print("等式算對", valid)
print("接上前步", linked)
print("最後答案對", steps[-1][2] == truth)
```

```text
等式算對 [False, True]
接上前步 [True]
最後答案對 True

```

