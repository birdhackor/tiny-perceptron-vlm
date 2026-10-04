from tiny_perceptron.data import render_chat

question = "2+2=?"
answers = {"簡短": "4", "生動": "4，就像兩雙筷子共有四根。"}
for style, answer in answers.items():
    inputs, labels = render_chat(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    print(style, "總位置", len(inputs), "學習位置", int((labels != -100).sum()))
