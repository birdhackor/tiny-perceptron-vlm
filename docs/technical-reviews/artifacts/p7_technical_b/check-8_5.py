import json
answers=['{"answer":4}','答案是：{"answer":4}','{"answer":5}']
out=[]
for text in answers+['{"answer":4.0}','{"answer":4,"extra":0}','{"answer":"4"}']:
 try:
  value=json.loads(text);valid=isinstance(value,dict) and set(value)=={"answer"};correct=valid and value["answer"]==4
 except json.JSONDecodeError:valid=correct=False
 print(text,"格式",valid,"內容",correct)
 out.append({"text":text,"valid":valid,"correct":correct})
assert [(r["valid"],r["correct"]) for r in out[:3]]==[(True,True),(False,False),(True,False)]
print(json.dumps(out,ensure_ascii=False))
