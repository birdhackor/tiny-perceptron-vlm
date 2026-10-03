a = [
    {"source": "A", "text": "舊題一", "answer_tokens": 2},
    {"source": "A", "text": "舊題二", "answer_tokens": 2},
]
b = [
    {"source": "B", "text": "新題一", "answer_tokens": 2},
    {"source": "B", "text": "新題二", "answer_tokens": 2},
]
recipe = a + b
print("每批來源", [record["source"] for record in recipe])
print("有效答案預算", sum(record["answer_tokens"] for record in recipe))
print("待測B-only與replay各自的A/B成績")
