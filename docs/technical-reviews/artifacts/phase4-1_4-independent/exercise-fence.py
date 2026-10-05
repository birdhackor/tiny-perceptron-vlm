from collections import Counter

text = "貓貓貓狗狗狗鳥"
counts = Counter(text)
total = sum(counts.values())
probability = {}
for char, count in counts.items():
    probability[char] = count / total
print(counts)
print(probability)
assert abs(sum(probability.values()) - 1) < 1e-6
