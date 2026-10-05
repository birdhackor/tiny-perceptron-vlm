import math

counts = [8536, 25630, 4875, 215]
fractions = [count / sum(counts) for count in counts]
terms = [-f * math.log(f) if f > 0 else 0.0 for f in fractions]
print(round(sum(terms) / math.log(len(counts)), 5))
