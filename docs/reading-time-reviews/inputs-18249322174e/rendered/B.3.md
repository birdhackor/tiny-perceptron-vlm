# B.3：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
from tiny_perceptron.retrieval import call_tool

request = {"name": "multiply", "arguments": {"a": 123, "b": 45}}
result = call_tool(request)
trace = [
    {"role": "assistant", "tool_request": request},
    {"role": "tool", "result": result},
]
print("返回值", result)
print("將送回對話的兩筆紀錄", trace)
assert result == 5535
```

```text
返回值 5535
將送回對話的兩筆紀錄 [{'role': 'assistant', 'tool_request': {'name': 'multiply', 'arguments': {'a': 123, 'b': 45}}}, {'role': 'tool', 'result': 5535}]

```

