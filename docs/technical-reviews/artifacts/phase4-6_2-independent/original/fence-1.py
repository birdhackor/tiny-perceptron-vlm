from collections import Counter

corpus = [list("abab"), list("abac")]
pairs = Counter()
for row in corpus:
    for left, right in zip(row, row[1:]):
        pairs[(left, right)] += 1
pair = pairs.most_common(1)[0][0]


def merge(row, pair):
    out = []
    i = 0
    while i < len(row):
        if i + 1 < len(row) and tuple(row[i : i + 2]) == pair:
            out.append("".join(pair))
            i += 2
        else:
            out.append(row[i])
            i += 1
    return out


print(pairs)
print("選擇", pair, "結果", [merge(row, pair) for row in corpus])
