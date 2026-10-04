from tiny_perceptron.data import render_chat
for answer in ["A", "AB"]:
    x, y = render_chat([{"role": "user", "content": "Q"}, {"role": "assistant", "content": answer}])
    print("answer", answer, "X", x.tolist(), "Y", y.tolist())
