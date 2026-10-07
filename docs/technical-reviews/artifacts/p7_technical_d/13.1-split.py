import json,hashlib
from scripts.course_experiments.text import arithmetic_records
from scripts.course_experiments.common import split_records
from scripts.course_experiments.behavior import _preference_parts
parts=split_records(arithmetic_records(),seed=42)
pairs=_preference_parts(parts)
for s,rs in parts.items():
 families={r['family'] for r in rs}
 for other,others in parts.items():
  if other!=s:assert not families.intersection(r['family'] for r in others)
 assert all(r['chosen']==str(x['a']+x['b']) and r['rejected']==str(x['a']+x['b']+1) for r,x in zip(pairs[s],rs))
 raw=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rs)
 print(s,len(rs),len(families),hashlib.sha256(raw.encode()).hexdigest())
print('total',sum(map(len,parts.values())))
