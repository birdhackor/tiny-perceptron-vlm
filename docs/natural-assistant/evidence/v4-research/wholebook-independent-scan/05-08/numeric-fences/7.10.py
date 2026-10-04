from tiny_perceptron.data import render_chat, pad_batch

example = render_chat(
    [
        {"role": "user", "content": "很長的問題"},
        {"role": "assistant", "content": "是"},
    ]
)
try:
    pad_batch([example], max_length=2)
except ValueError as error:
    print("預期資料錯誤", error)
else:
    raise AssertionError("應該偵測空監督")
x, y = example
first = (y != -100).nonzero()[0].item()
print("首有效索引", first, "最低保留長度", first + 1, "完整長度", len(x))
