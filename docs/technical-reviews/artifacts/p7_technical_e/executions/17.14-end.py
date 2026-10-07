import json
from pathlib import Path
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/qat.json').read_text())['results']
a,b=[r['runs'][n] for n in ['matched_ptq4','qat_packed4']]
for v in [a,b]:
 s=v['storage'];assert s['tensor_bytes']==s['parameter_tensor_bytes']+s['buffer_tensor_bytes'];assert s['file_overhead_bytes']==s['file_bytes']-s['tensor_bytes'];print('storage',s['tensor_bytes'],s['file_bytes'],s['file_overhead_bytes'])
 x=next(t for t in v['test']['generated_samples'] if t['question']=='color=red;shape=square;pitch=high;describe');print('describe',x['expected'],x['generated'])
assert a['storage']['tensor_bytes']==b['storage']['tensor_bytes']
print('fake_deployed_max_logit_difference',r['fake_deployed_max_logit_difference'])
