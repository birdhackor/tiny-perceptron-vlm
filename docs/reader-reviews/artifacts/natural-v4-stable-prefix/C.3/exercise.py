p = 0.3
for n in (1, 2, 4, 8):
    coverage = 1 - (1 - p) ** n
    print("候選數", n, "獨立假設下包含正解", round(coverage, 6))
candidates = [3, 3, 3, 3]
print("本組候選", candidates, "包含真值4", 4 in candidates)
