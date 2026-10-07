"""Owner's actual execution of read freeze-06 code and meaningful CPU variants."""
import json,sys,re,hashlib,io,contextlib,math
from pathlib import Path
b=Path('docs/technical-reviews/artifacts/p7_technical_a/freeze-06-callback')
m=json.load(open('docs/course-revision-20261007-phase7/reviews/freeze-06/manifest.json'));p=next(x for x in m['pages'] if x['page_id']=='1.9');body=Path(p['snapshot']).read_bytes();assert hashlib.sha256(body).hexdigest()==p['source_sha256']
codes=re.findall(r'```python\n(.*?)```',body.decode(),re.S);assert len(codes)==1;cp=b/'1-9-original-block.py'
with cp.open('x',encoding='utf-8') as f:f.write(codes[0])
ns={};stream=io.StringIO()
with contextlib.redirect_stdout(stream):exec(compile(codes[0],str(cp),'exec'),ns)
loss=ns['loss'];cases={}
for w in [1.,5.,3.]:
 h=.001;left=loss(w-h);right=loss(w+h);g=(right-left)/(2*h);analytic=2*(w-3);assert math.isclose(g,analytic,rel_tol=0.,abs_tol=1e-10)
 cases[str(w)]={'left':left,'right':right,'central_gradient':g,'analytic':analytic,'w_after_check':w}
table=[]
for w in [1.,1.1,1.01]:
 L=loss(w);row={'w':w,'loss':L,'change_vs_one':L-loss(1.)}
 if w!=1.:row['one_sided_ratio']=(L-loss(1.))/(w-1.)
 table.append(row)
assert abs(table[1]['loss']-3.61)<1e-12 and abs(table[2]['loss']-3.9601)<1e-12
assert abs(table[1]['one_sided_ratio']+3.9)<1e-12 and abs(table[2]['one_sided_ratio']+3.99)<1e-12
steps={}
for h in [.1,.01,.001,1e-8,1e-16]:steps[str(h)]={'left':loss(1.-h),'right':loss(1.+h),'gradient':(loss(1.+h)-loss(1.-h))/(2*h)}
# Large central step is exact for a quadratic in exact arithmetic, not a generic law.
assert math.isclose(steps['0.1']['gradient'],-4.,rel_tol=0.,abs_tol=1e-12)
try:z=(loss(1.)-loss(1.))/0.
except ZeroDivisionError:zero_h='ZeroDivisionError'
assert ns['w']==1.
out={'command':'PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/p7_technical_a/freeze-06-callback/check-1-9.py','python':sys.version,'device':'CPU Python float, no model training/GPU','source_sha256':p['source_sha256'],'code_path':str(cp),'code_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'original_stdout':stream.getvalue(),'original_w_unchanged':ns['w']==1.,'cases':cases,'table':table,'h_cases':steps,'zero_h':zero_h,'scope':'one scalar quadratic and finite difference only; no auto-update, no large-model derivative benchmark; central quadratic analytic exact for any nonzero h but floating-point cancellation can fail'}
rp=b/'cpu-execution.json'
with rp.open('x',encoding='utf-8') as f:f.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
