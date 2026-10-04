from tiny_perceptron.data import ByteTokenizer, render_chat

tok = ByteTokenizer()
x, y = render_chat(
    [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
)
first = (y != -100).nonzero()[0].item()
print("X", x.tolist(), "Y", y.tolist())
print("首答案預測位置", first, "輸入", x[first].item(), "答案", y[first].item())
assert x[first].item() == tok.assistant_id
assert y[first].item() == tok.encode("A")[0]
