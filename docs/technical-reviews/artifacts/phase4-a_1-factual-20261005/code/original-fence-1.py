question = "請依本次規則回答：3→?"
examples = ["1→2", "2→4"]
zero = question
few = "示例：" + "；".join(examples) + "。\n" + question
changed = "示例：1→3；2→6。\n" + question
print("無例子：", zero)
print("兩例子：", few)
print("換規則：", changed)
