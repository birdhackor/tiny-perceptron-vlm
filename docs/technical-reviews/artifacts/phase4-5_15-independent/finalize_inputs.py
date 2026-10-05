"""Save actually read supporting context and installed contracts, plus execution identity."""
from pathlib import Path
import hashlib,inspect,json,re,sys,urllib.request
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
sha=lambda raw:hashlib.sha256(raw).hexdigest()
for filename,lesson in [('03.md','3.5'),('05.md','5.14')]:
 raw=(ROOT/'course/chapters'/filename).read_bytes()
 headers=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
 i=next(i for i,h in enumerate(headers) if h[0].startswith(('## '+lesson+' ').encode()))
 end=headers[i+1].start() if i+1<len(headers) else len(raw)
 body=raw[headers[i].start():end]
 (BASE/'inputs'/('context-'+lesson+'.md')).write_bytes(body)
provenance=json.loads((BASE/'input-provenance.json').read_text())
provenance['scope']='5.15 only; necessary context read: 3.5 standard-deviation cross-reference and 5.14 same-start learning-rate comparison. No chapter introduction, no old technical/reader reviews, no empirical benchmark result or weight.'
for lesson in ['3.5','5.14']:
 path=BASE/'inputs'/('context-'+lesson+'.md')
 provenance['inputs'].append({'input':'course/chapters/'+('03.md' if lesson=='3.5' else '05.md')+'#'+lesson,'sha256':sha(path.read_bytes()),'snapshot':path.relative_to(ROOT).as_posix()})
(BASE/'input-provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
import torch
contracts={
 '_manual_seed_impl':torch.random._manual_seed_impl,
 '_no_grad_normal_':torch.nn.init._no_grad_normal_,
 'Linear_reset_parameters':torch.nn.Linear.reset_parameters,
 'LayerNorm_reset_parameters':torch.nn.LayerNorm.reset_parameters,
}
records=[]
for name,obj in contracts.items():
 raw=inspect.getsource(obj).encode()
 path=BASE/'installed'/(name+'.py');path.write_bytes(raw)
 records.append({'name':name,'file':inspect.getsourcefile(obj),'line':inspect.getsourcelines(obj)[1],'sha256':sha(raw),'snapshot':path.relative_to(ROOT).as_posix()})
for name,obj in [('Tensor-item',torch.Tensor.item),('torch-std',torch.std),('torch-equal',torch.equal)]:
 raw=obj.__doc__.encode()
 path=BASE/'installed'/(name+'.txt');path.write_bytes(raw)
 records.append({'name':name,'kind':'installed API __doc__','sha256':sha(raw),'snapshot':path.relative_to(ROOT).as_posix()})
environment=json.loads((BASE/'installed-environment.json').read_text())
environment['additional_contracts']=records
(BASE/'installed-environment.json').write_text(json.dumps(environment,indent=2)+'\n')
url='https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.item.html'
with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'technical-review/5.15'}),timeout=25) as response:
 raw=response.read();resolved=response.url
path=BASE/'sources/pytorch-2.8-item.html';path.write_bytes(raw)
from bs4 import BeautifulSoup
soup=BeautifulSoup(raw,'html.parser');main=soup.find('article') or soup.find('main') or soup
for e in main.find_all(['script','style']):e.decompose()
(BASE/'sources/pytorch-2.8-item.txt').write_text(main.get_text('\n',strip=True)+'\n')
(BASE/'item-acquisition.json').write_text(json.dumps({'url':url,'resolved_url':resolved,'accessed_on':'2026-10-05','sha256':sha(raw),'saved':True},indent=2)+'\n')
commands=[
 {'command':provenance['original_extraction_command'],'exit_code':0,'bounds':'30 seconds helper timeout; one original Python fence; no guard events','stdout':'original/stdout.txt','stderr':'original/stderr.txt','receipt':'original/execution.json'},
 {'command':"CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout 30s .venv/bin/python docs/technical-reviews/artifacts/phase4-5_15-independent/cpu_probe.py",'exit_code':0,'bounds':'30 seconds shell timeout; CPU constructors, row lookup and 20-index permutations only','stdout':'cpu-probe.stdout.txt','stderr':'cpu-probe.stderr.txt','receipt':'cpu-probe-results.json'},
 {'command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-5_15-independent/prepare_evidence.py','exit_code':0,'bounds':'25 seconds each HTTPS request; snapshots and source acquisition only','stdout':'prepare.stdout.json','stderr':'prepare.stderr.txt','receipt':'acquisition.json'},
 {'command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-5_15-independent/fetch_variance_paper.py','exit_code':0,'bounds':'25 seconds each HTTPS request; original source only; pdftotext timeout20','stdout':'extra-acquisition.stdout.json','stderr':'extra-acquisition.stderr.txt','receipt':'extra-acquisition.json'},
]
(BASE/'commands.json').write_text(json.dumps({'cwd':str(ROOT),'shell':'bash, login:false','commands':commands,'environment':environment},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'context_snapshots':['3.5','5.14'],'additional_installed_contracts':records,'item_source_saved':True},indent=2))
