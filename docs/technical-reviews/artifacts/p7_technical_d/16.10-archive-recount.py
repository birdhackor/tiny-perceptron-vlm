import torch,json,ast
from pathlib import Path
from tiny_perceptron.data import ByteTokenizer
print('torch',torch.__version__);r=json.load(open('docs/course-experiments/results/efficiency.json'))['results'];t=ByteTokenizer();print('mechanism',r['activation_checkpoint']);print('weighterror',r['update_variants']['activation_checkpoint']['weight_max_error_from_ordinary_after_updates'])
for n in ['ordinary','activation_checkpoint']:
 d=r['update_variants'][n];q=d['training'];assert q['steps']==q['optimizer_updates']==40 and q['effective_tokens']==2328
 b=q['memory_allocated_before_bytes'];p=q['peak_memory_allocated_bytes'];a=q['peak_additional_allocated_bytes'];assert p-b==a
 print(n,'updates/targets',40,2328,'stored_step_ms',q['warm_step_median_seconds']*1000,'before/peak/add_MiB',[x/2**20 for x in [b,p,a]])
 for sp,h in d['heldout'].items():
  m=0
  for s in h['samples']:
   ids=s['generated_ids'];ids=ids[:ids.index(2)] if 2 in ids else ids;ok=ids==t.encode(s['expected']);assert ok==s['exact'];m+=ok
  assert m==h['matches'];print(n,sp,'matches',m,'/',len(h['samples']),'NLL',h['nll_sum']/h['effective_tokens'])
def f(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(Path(p).read_text()).body if isinstance(x,ast.FunctionDef)}
x=f('docs/technical-reviews/artifacts/p7_technical_d/modern-historical-scripts_course_experiments_architecture.py');y=f('scripts/course_experiments/architecture.py')
for n in ['_checkpoint_forward','_checkpoint_probe']:assert x[n]==y[n];print('historical AST same',n)
