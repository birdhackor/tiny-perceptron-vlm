from tiny_perceptron.data import split_documents, toy_documents
split = split_documents(toy_documents(), seed=42)
for name, documents in split.items():
    print(name, len(documents), documents)
