width = 32
for vocab in [300, 1000, 3000]:
    embedding = vocab * width
    input_and_output = 2 * embedding
    print("詞表", vocab, "輸入參數", embedding, "輸入FP32 bytes", embedding * 4, "不共享輸入輸出參數", input_and_output)
