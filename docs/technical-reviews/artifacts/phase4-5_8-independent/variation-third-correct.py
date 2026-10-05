truth = [1, 2, 3, 4]
teacher_forced = [1, 2, 3, 4]
generated = [1, 2, 0, 0]
for name, prediction in [("正確前文下", teacher_forced), ("自由生成", generated)]:
    correct = sum(a == b for a, b in zip(truth, prediction, strict=True))
    token_accuracy = correct / len(truth)
    exact_match = prediction == truth
    print(name, "逐字正確率", token_accuracy, "整串相同", exact_match)
