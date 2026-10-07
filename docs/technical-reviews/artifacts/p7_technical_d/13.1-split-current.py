import json,hashlib
from scripts.course_experiments.text import arithmetic_records
from scripts.course_experiments.common import split_records
from scripts.course_experiments.behavior import _preference_parts
p='docs/course-experiments/results/dpo.json';j=json.load(open(p));data=j['results']['data']
parts=split_records(arithmetic_records(),seed=j['seed']);pairs=_preference_parts(parts)
print('raw_result_sha256',hashlib.sha256(open(p,'rb').read()).hexdigest())
print('raw_recorded_code_sha256',j['code_sha256'])
for s,rs in parts.items():
 fs={r['family'] for r in rs};assert all(not fs.intersection(r['family'] for r in os) for n,os in parts.items() if n!=s)
 assert all(r['chosen']==str(x['a']+x['b']) and r['rejected']==str(x['a']+x['b']+1) for r,x in zip(pairs[s],rs))
 h=hashlib.sha256(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs).encode()).hexdigest()
 print(s,len(rs),len(fs),h,'matches raw',h==data[s]['sha256'])
 assert h==data[s]['sha256'] and len(rs)==data[s]['records']
for p in ['scripts/course_experiments/behavior.py','scripts/course_experiments/text.py','scripts/course_experiments/common.py']:print('inspected_source',p,hashlib.sha256(open(p,'rb').read()).hexdigest())
print('total',sum(map(len,parts.values())))
