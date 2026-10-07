"""Check original PyPI release metadata and Python3.12 compatibility only."""
import json,urllib.request,hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pip._vendor.packaging.specifiers import SpecifierSet
pairs=[]
for line in Path('requirements-natural.txt').read_text().splitlines():
 if line and not line.startswith('#'):pairs.append(line.split('=='))
def fetch(pair):
 name,version=pair;url=f'https://pypi.org/pypi/{name}/{version}/json'
 with urllib.request.urlopen(url,timeout=30) as response:raw=response.read()
 v=json.loads(raw);p=v['info'].get('requires_python') or next((x.get('requires_python') for x in v['urls'] if x.get('requires_python')), '');assert v['info']['version']==version
 return {'name':name,'version':version,'url':url,'original_json_sha256':hashlib.sha256(raw).hexdigest(),'requires_python':p,'python3_12_allowed':SpecifierSet(p).contains('3.12.14'),'files':len(v['urls'])}
rows=list(ThreadPoolExecutor(max_workers=4).map(fetch,pairs));assert all(r['python3_12_allowed'] and r['files'] for r in rows);print(json.dumps({'rows':rows,'scope':'Original release availability and declared Python constraint only; no full dependency resolution, installation, model execution or CUDA'},indent=2))
