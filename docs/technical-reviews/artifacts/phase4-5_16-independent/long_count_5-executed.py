counts = {"短答": 90, "長答": 5}
lengths = {"短答": 1, "長答": 20}
tokens = {}
for source in counts:
    tokens[source] = counts[source] * lengths[source]
total = sum(tokens.values())
ratios = {source: amount / total for source, amount in tokens.items()}
print("有效答案token數", tokens)
print("有效答案token比例", ratios)
