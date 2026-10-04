# B.2：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
from tiny_perceptron.retrieval import call_tool

requests = [
    '{"name":"add","arguments":{"a":2,"b":3}}',
    '{"name":"delete_all","arguments":{"a":2,"b":3}}',
    '{"name":"add","arguments":{"a":true,"b":3}}',
]
for text in requests:
    try:
        print("結果", call_tool(text))
    except ValueError as error:
        print("拒絕", error)
```

```text
結果 5
拒絕 只接受 add/multiply 與明確 arguments
拒絕 a、b 必須是數字，bool 或程式字串都不接受

```

