truth = "1010"
predicted = "1000"
assert len(truth) == len(predicted)  # 本實驗只比較等長且對齊的字串
matched = sum(a == b for a, b in zip(truth, predicted))
print("等長逐位正確率", matched / len(truth))
print("整串正確", truth == predicted)
print("本例一次替換的CER", 1 / len(truth))
print("獨立錯誤假設下五位全對", round(0.9**5, 4))
