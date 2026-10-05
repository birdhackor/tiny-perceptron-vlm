records = []
for question, fact in [("1+1=?", "2"), ("2+2=?", "4")]:
    for style in ["生動", "簡短"]:
        reply = fact if style == "簡短" else fact + "；把兩組物件合起來數就知道了。"
        records.append({"提問": f"風格={style}；{question}", "回答": reply})
print("筆數", len(records))
for row in records:
    print(row["提問"], "→", row["回答"])
