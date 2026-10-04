# C.3：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
p = 0.3
for n in (1, 2, 4, 8):
    coverage = 1 - (1 - p) ** n
    print("候選數", n, "獨立假設下包含正解", round(coverage, 6))
candidates = [3, 4, 3, 5]
print("本組候選", candidates, "包含真值4", 4 in candidates)
```

```text
候選數 1 獨立假設下包含正解 0.3
候選數 2 獨立假設下包含正解 0.51
候選數 4 獨立假設下包含正解 0.7599
候選數 8 獨立假設下包含正解 0.942352
本組候選 [3, 4, 3, 5] 包含真值4 True

```

