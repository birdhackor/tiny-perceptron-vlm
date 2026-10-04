from tiny_perceptron.data import split_documents

docs = ["貓看狗。", "狗看貓。", "鳥看魚。", "魚看鳥。", "貓看狗。", "貓看狗。"]
parts = split_documents(docs, seed=42)
for name, documents in parts.items():
    print(name, len(documents), documents)
assert not (set(parts["train"]) & set(parts["validation"]))
