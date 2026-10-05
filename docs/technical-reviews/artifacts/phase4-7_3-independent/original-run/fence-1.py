from tiny_perceptron.data import render_chat

messages = [
    {"role": "user", "content": "問"},
    {"role": "assistant", "content": "答"},
]
x, y = render_chat(messages)
for position, (token, label) in enumerate(zip(x.tolist(), y.tolist(), strict=True)):
    status = "學下一項" if label != -100 else "忽略直接loss"
    print(position, "輸入ID", token, "目標", label, status)
print("有效目標數", (y != -100).sum().item())
assert (y != -100).sum().item() > 0
