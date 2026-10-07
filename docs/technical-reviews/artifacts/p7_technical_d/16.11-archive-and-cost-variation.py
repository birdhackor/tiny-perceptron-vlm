import json,torch,statistics,ast
from pathlib import Path
print('torch',torch.__version__)
setup,plain,compiled=4.,.01,.006
print('hypothetical break_even',setup/(plain-compiled))
for calls in [100,1000]:print('hypothetical setup4',calls,'plain',plain*calls,'compiled',setup+compiled*calls)
r=json.load(open('docs/course-experiments/results/efficiency.json'))['results']['compile'];print('archive first/wall',r['first_compiled_call_seconds'],r['worker_wall_seconds'],'graphs/kernels/breaks',r['unique_graphs'],r['inductor_generated_kernel_count'],r['graph_breaks'],'outputmax',r['output_max_error'])
meds={}
for n in ['eager','compiled']:
 d=r[n];v=d['samples_seconds'];m=statistics.median(v);assert len(v)==9 and d['warmup_calls']==3 and abs(m-d['median_seconds'])<1e-12;assert sorted(v)[4]==m
 print(n,'samples9 medianms',m*1000);meds[n]=m
saving=meds['eager']-meds['compiled'];assert saving<0 and r['estimated_calls_to_amortize_first_call'] is None;print('saving_seconds',saving,'finite_break_even_exists',False)
def f(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(Path(p).read_text()).body if isinstance(x,ast.FunctionDef)}
x=f('docs/technical-reviews/artifacts/p7_technical_d/modern-historical-scripts_course_experiments_architecture.py');y=f('scripts/course_experiments/architecture.py')
for n in ['_compile_worker','_compile_probe']:assert x[n]==y[n];print('historical AST same',n)
