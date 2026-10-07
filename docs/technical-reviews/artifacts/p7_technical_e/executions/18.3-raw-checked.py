import json
from pathlib import Path
from tiny_perceptron.data import ByteTokenizer
x=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/distillation.json').read_text())['results']['tasks'];tok=ByteTokenizer()
for n in ['attributes','style_transfer']:
 h=x[n]['hard_target_generation'];rows=h['audit'];correct=sum(t['teacher_ids']==tok.encode(t['gold_answer'])+[tok.eos_id] for t in rows);eos=sum(bool(t['teacher_ids']) and t['teacher_ids'][-1]==tok.eos_id for t in rows);invalid=sum(any(v<tok.byte_offset and v!=tok.eos_id for v in t['teacher_ids']) for t in rows);zero=sum(not t['teacher_ids'] for t in rows);print(n,'rows',len(rows),'correct',correct,'eos',eos,'invalid',invalid,'zero',zero)
 assert correct==h['correct'] and len(rows)==h['records'] and invalid==0 and zero==0
 for t in rows:
  if not t['teacher_correct']:print('wrong',t['gold_answer'],t['teacher_answer'])
for w in [16,32]:
 a=x['attributes']['runs'][f'w{w}_ce']['training'];b=x['attributes']['runs'][f'w{w}_teacher_hard']['training'];print('attributes',w,'same final',a['final_sha256']==b['final_sha256'],'same initial',a['initialization_sha256']==b['initialization_sha256'],'same batch',a['batch_plan_sha256']==b['batch_plan_sha256'])
