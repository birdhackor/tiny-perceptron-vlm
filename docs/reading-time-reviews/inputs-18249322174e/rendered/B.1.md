# B.1：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 3

```python
import json

request = {"name": "multiply", "arguments": {"a": 123, "b": 45}}
training_example = {"user": "123乘45", "assistant_tool_request": request}
text = json.dumps(request, ensure_ascii=False)
print("訓練例子", training_example)
print("可傳遞的請求文字", text)
print("目前只有請求，尚未執行乘法")
```

```text
訓練例子 {'user': '123乘45', 'assistant_tool_request': {'name': 'multiply', 'arguments': {'a': 123, 'b': 45}}}
可傳遞的請求文字 {"name": "multiply", "arguments": {"a": 123, "b": 45}}
目前只有請求，尚未執行乘法

```

