import json,torch
from tiny_perceptron.data import ByteTokenizer
print('torch',torch.__version__)
t=ByteTokenizer();r=json.load(open('docs/course-experiments/results/efficiency.json'))['results']['packing'];p=r['actual_updates']
assert r['padded_positions']==r['packed_positions']==sum(r['document_prefix_lengths'])==14
print('mechanism',[(k,r[k]) for k in ['logit_max_error','gradient_max_error','second_document_error_after_first_document_change_with_isolation','second_document_error_without_isolation']])
print('weight_error',p['weight_max_error_after_updates'])
for name in ['padded','packed']:
 d=p[name];assert d['steps']==d['optimizer_updates']==40;assert d['effective_tokens']==845
 print(name,'steps',d['steps'],'targets',d['effective_tokens'],'warm samples count from code',d['steps']-3,'stored_median_ms',d['warm_step_median_seconds']*1000)
 for sp in ['validation','test']:
  h=d['heldout'][sp];ss=h['samples'];matches=0
  for s in ss:
   ids=s['generated_ids'];ids=ids[:ids.index(2)] if 2 in ids else ids;exact=ids==t.encode(s['expected']);assert exact==s['exact'];matches+=exact
  assert matches==h['matches'];assert len(ss)==h['records']
  print(name,sp,'matches',matches,'/',len(ss))
  if sp=='test':print('shape_sample',ss[1]['expected'],ss[1]['generated'],t.decode(ss[1]['generated_ids']))
