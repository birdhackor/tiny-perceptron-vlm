import json
records=[{'numbers':[2,2],'answer':'4'},{'numbers':[2,2],'answer':'5'}];accepted=[]
for record in records:
 if record['answer']==str(sum(record['numbers'])):
  accepted.append({'messages':[{'role':'user','content':'2+2=?'},{'role':'assistant','content':record['answer']}],'teacher_revision':None,'provenance':'人工示例，非教師生成'})
print('保留筆數',len(accepted))
for record in accepted: print(json.dumps(record,ensure_ascii=False))
