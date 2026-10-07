"""Execute Python fences of a page this owner has completely checkpointed."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

page_id, final_unit=sys.argv[1],int(sys.argv[2])
folder=Path('docs/technical-reviews/artifacts/p7_technical_c')
trace=Path('docs/course-revision-20261007-phase7/reviews/freeze-03/traces/technical/c/2bcb00175a2949dd82090328f77c573f.jsonl')
events=[json.loads(line) for line in trace.read_text().splitlines()]
assert events[0]['reviewer_task']=='/root/p7_technical_c'
seen={e['unit_index'] for e in events if e['event']=='checkpoint' and e['page_id']==page_id}
assert set(range(final_unit+1))<=seen, 'Page not yet fully checkpointed by this owner'
m=json.loads(Path('docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json').read_text())
meta=next(x for x in m['pages'] if x['page_id']==page_id)
body=Path(meta['snapshot']).read_text()
blocks=re.findall(r'```python\n(.*?)```',body,re.S)
assert blocks, 'No Python fences'
source=folder/f'{page_id}-fences.py'
assert not source.exists(), 'Keep original execution evidence'
prefix="import torch, sys, json\ntorch.set_num_threads(1)\nprint('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))\n"
source.write_text(prefix+'\n'.join(f"\nprint('fence {i}')\n{code}" for i,code in enumerate(blocks))+'\n')
if len(sys.argv)>3:
    extra=Path(sys.argv[3])
    assert extra.resolve().is_relative_to(folder.resolve())
    with source.open('a') as f:f.write('\n# Owner supplied proportional check\n'+extra.read_text()+'\n')
command=['.venv/bin/python',str(source)]
result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
out=folder/f'{page_id}-fences.stdout.txt';out.write_text(result.stdout)
print(result.stdout)
print('exit_code',result.returncode)
libpath=folder/'evidence-library.json';lib=json.loads(libpath.read_text())
id=f'{page_id}-run'
lib['artifacts'][id]={'id':id,'kind':'execution','path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'description':f'本人执行已完整读完的{page_id}全部Python短fences，精确代码另存同目录。','command':' '.join(command),'result':f'实际退出码{result.returncode}；完整原stdout与stderr见本artifact。','environment':{'python':'3.13.5','torch':'2.14.1+cpu','device':'CPU'}}
lib['sources'][f'{page_id}-exec']={'id':f'{page_id}-exec','kind':'execution','title':f'本轮{page_id}已读短代码CPU实际输出','verified':result.returncode==0,'artifact_id':id}
libpath.write_text(json.dumps(lib,ensure_ascii=False,indent=2)+'\n')
raise SystemExit(result.returncode)
