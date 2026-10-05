from tiny_perceptron.data import shifted

ids = list(range(10))
context = 3
for start in [0, 3, 6]:
    segment = ids[start : start + context + 1]
    x, y = shifted(segment)
    print("起點", start, "輸入", x.tolist(), "答案", y.tolist())
    assert len(x) == len(y) == context
