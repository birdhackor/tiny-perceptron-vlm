answers = [
    {"名稱": "準確短答", "風格": 1, "正確": 1},
    {"名稱": "華麗錯答", "風格": 3, "正確": 0},
]
for style_weight in [0.2, 2.0]:
    print("風格權重", style_weight)
    for row in answers:
        total = style_weight * row["風格"] + row["正確"]
        print(row["名稱"], "分項", row["風格"], row["正確"], "總分", round(total, 1))
