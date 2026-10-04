gold = ["TOOL", "TOOL", "DIRECT", "DIRECT", "ASK", "ASK"]
predicted = ["TOOL", "DIRECT", "TOOL", "DIRECT", "ASK", "DIRECT"]
correct = sum(a == b for a, b in zip(gold, predicted, strict=True))
needed = sum(a == "TOOL" for a in gold)
missed = sum(a == "TOOL" and b != "TOOL" for a, b in zip(gold, predicted, strict=True))
not_needed = len(gold) - needed
extra = sum(a != "TOOL" and b == "TOOL" for a, b in zip(gold, predicted, strict=True))
print("動作選對", correct, "/", len(gold))
print("需要工具卻沒選", missed, "/", needed)
print("不需要工具卻選了", extra, "/", not_needed)
