from pathlib import Path
import urllib.request,hashlib,json
out=Path('docs/technical-reviews/artifacts/phase4-12_14-independent/sources');out.mkdir(exist_ok=True)
urls={
 'hf-chat-v4.57.1.md':'https://raw.githubusercontent.com/huggingface/transformers/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/docs/source/en/chat_templating.md',
 'torchmetrics-cer-v1.8.2.py':'https://raw.githubusercontent.com/Lightning-AI/torchmetrics/v1.8.2/src/torchmetrics/functional/text/cer.py',
 'python313-stdtypes.html':'https://docs.python.org/3.13/library/stdtypes.html',
 'python313-functions.html':'https://docs.python.org/3.13/library/functions.html'}
records=[]
for name,url in urls.items():
 try:
  with urllib.request.urlopen(url,timeout=30) as response: raw=response.read();status=response.status
  (out/name).write_bytes(raw);record={'name':name,'url':url,'http_status':status,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 except Exception as e:record={'name':name,'url':url,'error':repr(e)}
 records.append(record);print(json.dumps(record))
(out/'fetch.json').write_text(json.dumps(records,indent=2)+'\n')
