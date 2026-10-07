import json,hashlib
from pathlib import Path
q=Path('docs/course-experiments/results/quantization.json');s=Path('docs/course-experiments/results/sft.json')
d=json.loads(q.read_text())['results'];a=json.loads(s.read_text())['results']
print('source_sha256',hashlib.sha256(q.read_bytes()).hexdigest(),'sft_sha256',hashlib.sha256(s.read_bytes()).hexdigest())
t=d['training'];print('training',{k:t[k] for k in ['steps','optimizer_updates','batch_size','learning_rate','effective_supervised_tokens','weights_changed','initialization_sha256','final_sha256']})
assert t['steps']==t['optimizer_updates']==120 and t['effective_supervised_tokens']==13610
print('data',d['data'])
assert d['data']['counts']=={'train':45,'validation':5,'test':10}
assert d['data']['families']=={'train':9,'validation':1,'test':2} and d['data']['family_intersections']==0
print('config',d['teacher_provenance']['config'])
print('same_fp32_source',d['same_fp32_source'],'reloaded_packed_checkpoints_before_evaluation',d['reloaded_packed_checkpoints_before_evaluation'])
assert d['same_fp32_source'] and d['reloaded_packed_checkpoints_before_evaluation']
print('original SFT test',a['after']['test']['exact_match'],a['after']['test']['examples'])
r=d['runs']['fp32']['test'];print('updated fp32 test',r['correct'],r['examples'],r['exact_match'])
assert a['after']['test']['exact_match']*a['after']['test']['examples']==5
assert r['correct']==6 and r['examples']==10 and r['exact_match']==.6
print('Read historical raw measurements and generator source; training not rerun and private data/checkpoints not reacquired.')
