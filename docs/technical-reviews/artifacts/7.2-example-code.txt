from tiny_perceptron.data import ByteTokenizer

tok = ByteTokenizer()
question = "1+1=?"
prompt = [tok.bos_id, tok.user_id] + tok.encode(question) + [tok.eos_id, tok.assistant_id]
print(prompt)
print("最後邊界", prompt[-1])
assert prompt[-1] == tok.assistant_id
