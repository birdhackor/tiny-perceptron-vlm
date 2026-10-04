from tiny_perceptron.data import ByteTokenizer, render_chat

tokenizer = ByteTokenizer()
messages = [
    {"role": "system", "content": "精確計算用工具，解釋或照抄直接回答；缺資訊或工具先求助。計算器可用。"},
    {"role": "user", "content": "1加2等於多少？"},
    {"role": "assistant", "content": "TOOL"},
]
x, labels = render_chat(messages, tokenizer)
answer_ids = labels[labels != -100].tolist()
print("輸入位置數", len(x))
print("參與答案代價的文字", tokenizer.decode(answer_ids))
print("參與答案代價的位置數", len(answer_ids))
print("最後的標籤是EOS", answer_ids[-1] == tokenizer.eos_id)
