truth = ["circle", "square", "circle", "square"]
text_only = ["circle"] * len(truth)
accuracy = sum(a == b for a, b in zip(truth, text_only)) / len(truth)
print("平衡資料的固定猜測", accuracy)
imbalanced = ["circle"] * 9 + ["square"]
print("九比一資料的固定猜測", imbalanced.count("circle") / len(imbalanced))
