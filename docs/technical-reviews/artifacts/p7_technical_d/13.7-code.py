from tiny_perceptron.data import ByteTokenizer
tok = ByteTokenizer()
answers = ["4", "4。" * 8]
for answer in answers:
    body = tok.encode(answer)
    print("回答", answer, "正文位元組tokens", len(body), "含結束學習位置", len(body) + 1)
