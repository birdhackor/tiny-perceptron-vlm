pair = {"prompt": "2+2=?", "A": "4", "B": "4，就像兩雙筷子共有四根。"}
rule = "先正確，再遵循只回數字"
chosen = "A"
print("條件", pair["prompt"])
print("比較", pair["A"], "／", pair["B"])
print("理由", rule, "選擇", chosen)
