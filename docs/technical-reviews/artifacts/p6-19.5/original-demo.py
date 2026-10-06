from tiny_perceptron.selftrained.dataset import RecordEncoder
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer

messages = [
    {"role": "user", "content": "那個問題要怎麼處理？"},
    {"role": "assistant", "content": "請說明是地址、App還是卡片的問題。"},
]
tok = CharacterTokenizer.build([m["content"] for m in messages])
row = {"id": "teaching-demo", "task": "text", "messages": messages}
encoded = RecordEncoder(tok, ".").encode(row)
targets = [value for value in encoded["labels"] if value != -100]
print(tok.decode(targets, skip_special_tokens=False))
