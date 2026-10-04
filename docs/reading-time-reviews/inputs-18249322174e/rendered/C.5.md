# C.5：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
plans = [
    {"candidates": 1, "answer_tokens_each": 80},
    {"candidates": 4, "answer_tokens_each": 20},
]
verify_tokens_each = 5
for plan in plans:
    n = plan["candidates"]
    generated = n * plan["answer_tokens_each"]
    verified = n * verify_tokens_each
    print("候選", n, "生成", generated, "驗證", verified, "總token", generated + verified)
```

```text
候選 1 生成 80 驗證 5 總token 85
候選 4 生成 80 驗證 20 總token 100

```

