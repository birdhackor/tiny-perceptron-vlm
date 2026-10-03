plans = [
    {"candidates": 1, "answer_tokens_each": 80},
    {"candidates": 4, "answer_tokens_each": 20},
]
verify_tokens_each = 0
for plan in plans:
    n = plan["candidates"]
    generated = n * plan["answer_tokens_each"]
    verified = n * verify_tokens_each
    print("候選", n, "生成", generated, "驗證", verified, "總token", generated + verified)
