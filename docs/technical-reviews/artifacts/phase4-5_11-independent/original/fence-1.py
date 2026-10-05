from tiny_perceptron.data import fingerprint

docs = ["貓  看狗", "貓 看狗", "狗 看貓"]
normalized = [" ".join(text.split()) for text in docs]
hashes = [fingerprint(text) for text in docs]
print(normalized)
print("原始筆數", len(docs), "正規化後不同內容", len(set(hashes)))
print("指紋字元數", len(hashes[0]))
assert hashes[0] == hashes[1]
