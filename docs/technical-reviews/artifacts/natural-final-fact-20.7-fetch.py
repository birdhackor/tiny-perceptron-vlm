from pathlib import Path
import urllib.request,hashlib,json,concurrent.futures,struct
P=Path('docs/technical-reviews/artifacts/natural-final-fact-20.7-')
urls={
'whisper-card.md':'https://huggingface.co/openai/whisper-small/raw/973afd24965f72e36ca33b3055d56a652f456b4d/README.md',
'whisper-config.json':'https://huggingface.co/openai/whisper-small/raw/973afd24965f72e36ca33b3055d56a652f456b4d/config.json',
'whisper-generation-config.json':'https://huggingface.co/openai/whisper-small/raw/973afd24965f72e36ca33b3055d56a652f456b4d/generation_config.json',
'fleurs-card.md':'https://huggingface.co/datasets/google/fleurs/raw/d7c758a6dceecd54a98cac43404d3d576e721f07/README.md',
'fleurs-loader.py':'https://huggingface.co/datasets/google/fleurs/raw/d7c758a6dceecd54a98cac43404d3d576e721f07/fleurs.py',
'uax15.html':'https://www.unicode.org/reports/tr15/tr15-55.html',
'python-unicodedata.rst':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/unicodedata.rst',
'python-stdtypes.rst':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst',
'fleurs-v1.pdf':'https://arxiv.org/pdf/2205.12446v1',
'whisper-v1.pdf':'https://arxiv.org/pdf/2212.04356v1'}
def fetch(item):
 name,url=item
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'technical-source-review/1'})
  with urllib.request.urlopen(req,timeout=25) as r:
   raw=r.read(12*1024*1024); status=r.status; date=r.headers.get('Date'); resolved=r.url
  Path(str(P)+name).write_bytes(raw)
  return {'name':name,'url':url,'response_status':status,'response_date':date,'resolved_url':resolved,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 except Exception as e: return {'name':name,'url':url,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor: receipts=list(executor.map(fetch,urls.items()))
# Read only the safetensors JSON header, closing responses immediately afterward.
url='https://huggingface.co/openai/whisper-small/resolve/973afd24965f72e36ca33b3055d56a652f456b4d/model.safetensors'
try:
 with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-7'}),timeout=25) as r: first=r.read(8); headstatus=r.status
 size=struct.unpack('<Q',first)[0]; assert 0<size<1024*1024
 with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes=0-{size+7}'}),timeout=25) as r:
  chunk=r.read(size+8); hdrstatus=r.status
 header=json.loads(chunk[8:]);Path(str(P)+'whisper-header.json').write_text(json.dumps(header,indent=2)+'\n')
 count=0
 for name,t in header.items():
  if name=='__metadata__':continue
  elements=1
  for d in t['shape']:elements*=d
  count+=elements
 receipts.append({'name':'whisper-header.json','url':url,'head_bytes_read':8,'header_bytes_read':len(chunk),'header_size':size,'first_status':headstatus,'header_status':hdrstatus,'tensor_count':len(header)-('__metadata__' in header),'tensor_elements_sum':count,'sha256':hashlib.sha256(Path(str(P)+'whisper-header.json').read_bytes()).hexdigest(),'scope':'range-limited serialized tensor shapes; no model weights loaded/downloaded in full'})
except Exception as e: receipts.append({'name':'whisper-header.json','url':url,'error':str(e)})
Path(str(P)+'fetch-receipt.json').write_text(json.dumps(receipts,indent=2)+'\n'); print(json.dumps(receipts,indent=2))
raise SystemExit(int(any('error' in r for r in receipts)))
