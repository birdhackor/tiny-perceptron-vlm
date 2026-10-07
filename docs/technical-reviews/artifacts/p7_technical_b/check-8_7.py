import json
answers=[{"名稱":"準確短答","風格":1,"正確":1},{"名稱":"華麗錯答","風格":3,"正確":0}]
out={}
for style_weight in [.2,2.0,.5]:
 scores=[];print("風格權重",style_weight)
 for row in answers:
  total=style_weight*row["風格"]+row["正確"];scores.append(total)
  print(row["名稱"],"分項",row["風格"],row["正確"],"總分",round(total,1))
 out[str(style_weight)]=scores
assert out["0.2"][0]>out["0.2"][1] and out["2.0"][0]<out["2.0"][1] and out["0.5"][0]==out["0.5"][1]
print(json.dumps(out))
