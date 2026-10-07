import json,hashlib
from pathlib import Path
b=Path('docs/technical-reviews/artifacts/p7_technical_e/sources')
for name in ['joint','encoders']:
 p=b/(name+'.json');assert not p.exists();p.write_bytes(Path('docs/course-experiments/results/'+name+'.json').read_bytes());d=json.loads(p.read_text());r=d['results'];print(name,'sha256',hashlib.sha256(p.read_bytes()).hexdigest(),'code_sha256',d['code_sha256']['scripts/course_experiments/modalities.py'])
 ds=r['data'] if name=='joint' else r['audio']['data']
 for split,s in ds['splits'].items():print(name,split,'count',s['count'],'frequency',sorted({x['frequency'] for x in s['records']}))
