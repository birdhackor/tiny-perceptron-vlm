import json
records=[]
for question,fact in [("1+1=?","2"),("2+2=?","4")]:
 for style in ["簡短","生動"]:
  reply=fact if style=="簡短" else fact+"；把兩組物件合起來數就知道了。"
  records.append({"提問":f"風格={style}；{question}","回答":reply})
print("筆數",len(records))
for row in records:print(row["提問"],"→",row["回答"])
assert len(records)==4
print(json.dumps({"records":records,"variation_condition_removed":[{"提問":r["提問"].split("；",1)[1],"回答":r["回答"]} for r in records]},ensure_ascii=False))
