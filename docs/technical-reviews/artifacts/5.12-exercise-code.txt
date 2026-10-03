def grams(text, n=2):
    return {text[i : i + n] for i in range(len(text) - n + 1)}


a = grams("紅色圓形在左邊", n=3)
b = grams("紅色圓形在右邊", n=3)
common = a & b
all_parts = a | b
score = len(common) / len(all_parts)
print("共有片段", sorted(common))
print("共有數", len(common), "聯集數", len(all_parts), "相似度", score)
