import json
from pathlib import Path
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/distillation.json').read_text())['results']['tasks']['moe_to_dense'];print('data',r['data']);print('teacher-config',r['teacher_provenance']['config'])
for n in ['w32_ce','w32_ce_kl']:
 v=r['runs'][n];tr=v['training'];print(n,'records',tr['training_examples'],'init',tr['initialization_sha256'],'batch',tr['batch_plan_sha256'],'steps',tr['optimizer_updates'],'targets',tr['effective_supervised_tokens']);t=v['test'];print('test',t['examples'],t['supervised_tokens'],'NLL',t['nll_sum']/t['supervised_tokens'],'storedNLL',t['answer_nll']);assert abs(t['nll_sum']/t['supervised_tokens']-t['answer_nll'])<1e-12
 for s in t['generated_samples'][:2]:print('sample',s['generated'],'ids',s['generated_ids'])
