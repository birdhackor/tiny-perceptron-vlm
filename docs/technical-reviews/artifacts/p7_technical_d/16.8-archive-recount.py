import json,statistics,torch,ast
from pathlib import Path
print('torch',torch.__version__);r=json.load(open('docs/course-experiments/results/efficiency.json'))['results'];d=r['manual_vs_sdpa']
assert d['shape']==[4,53];assert d['output_max_error']<1e-5 and d['gradient_max_error']<1e-8;print('shape/dtype/errors',d['shape'],d['dtype'],d['output_max_error'],d['gradient_max_error']);print('backend',d['backend'])
for n,x in d['timings'].items():
 v=x['samples_seconds'];m=statistics.median(v);assert abs(m-x['median_seconds'])<1e-12;print(n,'forward samples',len(v),'median_ms',m*1000)
for n in ['ordinary','sdpa']:
 x=r['update_variants'][n];q=x['training'];assert q['steps']==q['optimizer_updates']==40 and q['effective_tokens']==2328;print(n,'updates/targets',q['optimizer_updates'],q['effective_tokens'],'stored_step_ms',q['warm_step_median_seconds']*1000)
print('weight_error',r['update_variants']['sdpa']['weight_max_error_from_ordinary_after_updates'])
def f(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(Path(p).read_text()).body if isinstance(x,ast.FunctionDef)}
a=f('docs/technical-reviews/artifacts/p7_technical_d/modern-historical-scripts_course_experiments_architecture.py');b=f('scripts/course_experiments/architecture.py')
for n in ['_sdpa_probe','_gradients','_benchmark','_train']:
 assert a[n]==b[n];print('historical AST same',n)
