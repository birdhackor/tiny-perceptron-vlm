# 1.1：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
text = "貓看狗，狗看貓。"
chars = sorted(set(text))
to_id = {}
for i, char in enumerate(chars):
    to_id[char] = i
ids = [to_id[char] for char in text]
restored = "".join(chars[i] for i in ids)
print(chars)
print(ids)
print(restored)
assert restored == text
```

```text
['。', '狗', '看', '貓', '，']
[3, 2, 1, 4, 1, 2, 3, 0]
貓看狗，狗看貓。

```

