budget = {"history": 20, "question_and_format": 15, "retrieval": 40, "answer": 25}
limit = 100
used = sum(budget.values())
print("輸入", used - budget["answer"], "答案預留", budget["answer"])
print("總預算", used, "放得下", used <= limit)
budget["history"] += 30
used = sum(budget.values())
print("歷史變長後", used, "超出", max(0, used - limit))
