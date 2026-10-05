short = "答案是4。"
long = short * 10
candidates = [short, long]
lengths = [len(text) for text in candidates]
chosen = max(candidates, key=len)
print("字元數", lengths)
print("長度裁判選長版", chosen == long)
print("不同句子數", len(set(long.split("。")[:-1])))
