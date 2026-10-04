import json

request = {"name": "multiply", "arguments": {"a": 123, "b": 45}}
training_example = {"user": "123乘45", "assistant_tool_request": request}
text = json.dumps(request, ensure_ascii=False)
print("訓練例子", training_example)
print("可傳遞的請求文字", text)
print("目前只有請求，尚未執行乘法")
