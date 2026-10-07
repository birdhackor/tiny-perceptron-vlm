import json
from pathlib import Path
t=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/distillation.json').read_text())['results']['tasks']['attributes']
for n,r in [('teacher',{'storage':t['teacher_storage'],'test':t['teacher_test']}),*[(k,t['runs'][k]) for k in ['w32_ce','w32_ce_kl','w32_ce_kl_packed4']]]:
 print(n,'tensor_bytes',r['storage']['tensor_bytes'],'file_bytes',r['storage']['file_bytes'],'parameter_bytes',r['storage']['parameter_tensor_bytes'],'buffer_bytes',r['storage']['buffer_tensor_bytes'],'test',r['test']['correct'],r['test']['examples'],'NLL',r['test']['answer_nll'],'effective',r['test']['supervised_tokens'])
a=t['runs']['w32_ce_kl']['test']['generated_samples'];b=t['runs']['w32_ce_kl_packed4']['test']['generated_samples'];assert len(a)==len(b)==10
for x,y in zip(a,b):print(x.get('row'),x.get('target'),x['generated'],y['generated'],x['exact_match'],y['exact_match'])
print('same_student_architecture',t['runs']['w32_ce']['storage']['parameter_count']==t['runs']['w32_ce_kl']['storage']['parameter_count'])
