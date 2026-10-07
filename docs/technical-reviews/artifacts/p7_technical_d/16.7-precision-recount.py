import json,statistics,torch,ast,hashlib
from pathlib import Path
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM,ModelConfig
print('torch',torch.__version__)
r=json.load(open('docs/course-experiments/results/precision.json'));t=ByteTokenizer()
old=Path('docs/technical-reviews/artifacts/p7_technical_d/modern-historical-scripts_course_experiments_architecture.py');new=Path('scripts/course_experiments/architecture.py')
assert hashlib.sha256(old.read_bytes()).hexdigest()==r['code_sha256'][str(new)]
def funcs(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(p.read_text()).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
f,g=funcs(old),funcs(new)
for n in ['run_precision','_train','_amp','_heldout','_nll']:
 assert f[n]==g[n];print('historical AST same',n)
for n,d in r['results']['variants'].items():
 m=TinyLM(ModelConfig(**d['model']['config']));params=sum(p.numel() for p in m.parameters());by=sum(p.numel()*p.element_size() for p in m.parameters());assert params==141568 and by==566272
 q=d['training'];assert q['steps']==q['optimizer_updates']==200 and q['skipped_updates']==0 and q['effective_tokens']==22493
 assert all(x['gradients_finite'] for x in q['history']);assert q['all_parameters_finite'] and d['logits_finite']
 print(n,'params/bytes',params,by,'weights/logits',d['weights_dtype'],d['observed_logits_dtype'],'updates/skip/targets',q['optimizer_updates'],q['skipped_updates'],q['effective_tokens'],'scaler',q['scaler_enabled'],'stored_step_ms',q['warm_step_median_seconds']*1000)
 inf=d['inference'];med=statistics.median(inf['samples_seconds']);assert abs(med-inf['median_seconds'])<1e-12;print('forward9 median_ms',med*1000)
 for sp in ['validation','test']:
  h=d['heldout'][sp];matches=0;targets=0
  for s in h['samples']:
   ids=s['generated_ids'];body=ids[:ids.index(2)] if 2 in ids else ids;ok=body==t.encode(s['expected']);assert ok==s['exact'];matches+=ok;targets+=len(t.encode(s['expected']))+1
  assert targets==h['effective_tokens'] and matches==h['matches'];nll=h['nll_sum']/targets;assert abs(nll-h['nll'])<1e-12
  print(sp,'NLL',round(nll,5),'targets',targets,'match',matches,'/',len(h['samples']))
