import json
from pathlib import Path
x=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/distillation.json').read_text())['results']['tasks']
for n in ['attributes','style_transfer']:
 h=x[n]['hard_target_generation'];rows=h['audit'];correct=sum(t['teacher_ids']==[b+8 for b in t['gold_answer'].encode('utf-8')]+[2] for t in rows);eos=sum(bool(t['teacher_ids']) and t['teacher_ids'][-1]==2 for t in rows);invalid=sum(any(v<8 and v!=2 for v in t['teacher_ids']) for t in rows);zero=sum(not t['teacher_ids'] for t in rows);print(n,'rows',len(rows),'correct',correct,'eos',eos,'invalid',invalid,'zero',zero)
 assert correct==h['correct'] and len(rows)==h['records'] and invalid==0 and zero==0
 for t in rows:
  if not t['teacher_correct']:print('wrong',t['gold_answer'],t['teacher_answer'])
 for w in [16,32]:
  a=x[n]['runs'][f'w{w}_ce']['training'];b=x[n]['runs'][f'w{w}_teacher_hard']['training'];print(n,w,'same final',a['final_sha256']==b['final_sha256'],'same initial',a['initialization_sha256']==b['initialization_sha256'],'same batch',a['batch_plan_sha256']==b['batch_plan_sha256'])
