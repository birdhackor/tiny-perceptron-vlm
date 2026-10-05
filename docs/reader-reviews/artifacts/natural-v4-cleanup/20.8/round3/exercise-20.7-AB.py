from tiny_perceptron.data import render_chat

x, y = render_chat(
    [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "AB"},
    ]
)
active = y != -100
first = active.nonzero()[0].item()
print("輸入位置數", len(x))
print("計分目標數", active.sum().item())
print("第一個計分目標的位置", first)
print("計分目標ID", y[active].tolist())
