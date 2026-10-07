import json
from pathlib import Path
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/qat.json').read_text())['results']
for n,v in r['runs'].items():
 for split in ['test']:
  t=v[split];calc=t['nll_sum']/t['supervised_tokens'];s=t['generated_samples'];correct=sum(x['exact'] for x in s);eos=sum(x['ended_with_eos'] for x in s)
  print(n,'nll',round(calc,4),'correct',correct,'examples',len(s),'eos',eos)
  assert abs(calc-t['answer_nll'])<1e-12 and correct==t['correct'] and eos==t['eos_count']
for n in ['matched_ptq4','qat_packed4']:
 t=r['runs'][n]['validation'];print(n,'validation',t['correct'],t['examples'],t['nll_sum']/t['supervised_tokens'])
a,b=r['training'].values();print('same init',a['initialization_sha256']==b['initialization_sha256'],'same batch',a['batch_plan_sha256']==b['batch_plan_sha256'],'updates',a['optimizer_updates'],b['optimizer_updates'],'targets',a['effective_supervised_tokens'],b['effective_supervised_tokens'])
