import concurrent.futures, datetime, hashlib, json, pathlib, urllib.request, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
A = ROOT / 'docs/technical-reviews/artifacts'
O = ROOT / 'outputs/natural-extension/fact-20.8-public'
O.mkdir(parents=True, exist_ok=True)
manifest = json.loads((ROOT/'docs/natural-assistant/public-release.json').read_text())
items = [
 ('hashlib', 'https://raw.githubusercontent.com/python/cpython/v3.12.14/Doc/library/hashlib.rst', A/'natural-final-fact-20.8-hashlib.rst'),
 ('python-bytes', 'https://raw.githubusercontent.com/python/cpython/v3.12.14/Doc/library/stdtypes.rst', A/'natural-final-fact-20.8-stdtypes.rst'),
 ('torch-dtypes', 'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/docs/source/tensor_attributes.rst', A/'natural-final-fact-20.8-torch-dtypes.rst'),
 ('torch-serialization', 'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/docs/source/notes/serialization.rst', A/'natural-final-fact-20.8-torch-serialization.rst'),
 ('hf-download', 'https://raw.githubusercontent.com/huggingface/huggingface_hub/v0.36.2/docs/source/en/guides/download.md', A/'natural-final-fact-20.8-hf-download.md'),
 ('pytorch-versions', 'https://pytorch.org/get-started/previous-versions/', A/'natural-final-fact-20.8-pytorch-versions.html'),
]
commit='1d5fccbc76a6ebbb3757f9b31df17a7c22ea171b'
for p in ['docs/natural-assistant/public-release.json','requirements-natural.txt','scripts/fetch_natural_release.py','tiny_perceptron/natural_assistant.py','tiny_perceptron/natural_ui.py']:
 items.append(('student-git-'+p,'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/'+commit+'/'+p,A/('natural-final-fact-20.8-student-'+p.replace('/','__'))))
for f in manifest['files']:
 items.append(('public-'+f['output'],'https://huggingface.co/'+manifest['repo']+'/resolve/'+manifest['revision']+'/'+f['path'],O/f['output']))
def fetch(item):
 name,url,path=item; start=time.monotonic()
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'independent-factual-review-20.8'}),timeout=45) as r:
  b=r.read(); status=r.status
 path.write_bytes(b)
 receipt={'id':name,'url':url,'path':str(path.relative_to(ROOT)), 'http_status':status,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'elapsed_seconds':time.monotonic()-start,'authentication_header_supplied':False}
 if name.startswith('public-'):
  f=next(f for f in manifest['files'] if f['output']==name[7:]);assert len(b)==f['bytes'] and receipt['sha256']==f['sha256'];receipt['exact_manifest_match']=True
 if name.startswith('student-git-'):
  local=ROOT/name[len('student-git-'):]; receipt['equals_current_local_bytes']=b==local.read_bytes();assert receipt['equals_current_local_bytes']
 return receipt
started=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 rows=list(pool.map(fetch,items))
result={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'executed':True,'elapsed_seconds':time.monotonic()-started,'anonymous_fixed_adapter_bytes':sum(r['bytes'] for r in rows if r['id'].startswith('public-')),'files':rows,'scope':'Read-only public fixed files; downloaded weights only ignored outputs; no model load/forward/GPU/install/Git mutation.'}
(A/'natural-final-fact-20.8-fetch.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'files':len(rows),'adapter_bytes':result['anonymous_fixed_adapter_bytes'],'git_files_match':all(r.get('equals_current_local_bytes',True) for r in rows),'elapsed':result['elapsed_seconds']}))
