import json,hashlib,urllib.request,datetime
from pathlib import Path
A=Path(__file__).resolve().parent
S=A/'sources';S.mkdir(exist_ok=True)
receipt=[]
repos=[('huggingface/transformers','v4.57.1',[('generation-config.py','src/transformers/generation/configuration_utils.py'),('generation-utils.py','src/transformers/generation/utils.py'),('stopping-criteria.py','src/transformers/generation/stopping_criteria.py'),('logits-process.py','src/transformers/generation/logits_process.py'),('chat-templating.md','docs/source/en/chat_templating.md'),('cache-explanation.md','docs/source/en/cache_explanation.md'),('pytorch-utils.py','src/transformers/pytorch_utils.py')]),('vllm-project/vllm','v0.8.5',[('vllm-config.py','vllm/config.py')])]
def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'Independent-factual-review-7.19'})
 with urllib.request.urlopen(req,timeout=20) as r:return r.read(),r.geturl(),r.status
for repo,tag,files in repos:
 api=f'https://api.github.com/repos/{repo}/git/ref/tags/{tag}'
 b,u,status=get(api); obj=json.loads(b); (S/(repo.split('/')[-1]+'-tag.json')).write_bytes(b)
 commit=obj['object']['sha']
 if obj['object']['type']=='tag':
  b,u,status=get(obj['object']['url']); (S/(repo.split('/')[-1]+'-annotated-tag.json')).write_bytes(b);commit=json.loads(b)['object']['sha']
 for name,path in files:
  tagged=f'https://raw.githubusercontent.com/{repo}/{tag}/{path}';immutable=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
  raw,final,status=get(tagged); fixed,_,_=get(immutable)
  assert raw==fixed,(tag,path)
  (S/name).write_bytes(raw)
  receipt.append({'name':name,'tag':tag,'commit':commit,'url':tagged,'immutable_url':immutable,'final_url':final,'http_status':status,'tag_equals_immutable':True,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'fetched_utc':datetime.datetime.now(datetime.UTC).isoformat(),'authority':'Maintainer-owned official GitHub repository; tag ref resolved through GitHub API and bytes compared with commit-pinned raw source.'})
  print(name,tag,commit,len(raw),hashlib.sha256(raw).hexdigest())
(A/'source-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
