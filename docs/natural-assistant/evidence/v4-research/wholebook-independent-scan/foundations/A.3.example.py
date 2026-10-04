from tiny_perceptron.retrieval import lexical_terms, retrieve

query = "書店地址"
docs = [
    {"id": "d1", "text": "貓喜歡曬太陽"},
    {"id": "d2", "text": "書店地址是青街8號"},
]
print("查詢片段", lexical_terms(query))
hits = retrieve(query, docs, k=1)
print("取回", hits)
print("改問法", retrieve("營業處在哪", docs, k=1))
