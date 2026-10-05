from tiny_perceptron.data import ByteTokenizer, render_chat

tok = ByteTokenizer()
x, y = render_chat([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
print("最後有效目標", y[-1].item(), "EOS", tok.eos_id)
assert y[-1].item() == tok.eos_id
generated = [tok.encode("A")[0], tok.eos_id, tok.encode("B")[0]]
max_new_tokens = 2
visible = []
for token in generated[:max_new_tokens]:
    if token == tok.eos_id:
        break
    visible.append(token)
print("顯示的內容", tok.decode(visible))
