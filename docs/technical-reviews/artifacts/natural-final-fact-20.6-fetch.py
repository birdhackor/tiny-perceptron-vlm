"""Fetch original authority documents, retaining version and full-byte SHA receipts."""
import concurrent.futures, hashlib, json, urllib.request
from pathlib import Path
OUT=Path(__file__).resolve().parent
PREFIX='natural-final-fact-20.6-'
items={
 'uax15.html':'https://www.unicode.org/reports/tr15/tr15-57.html',
 'python-unicodedata.rst':'https://raw.githubusercontent.com/python/cpython/v3.12.14/Doc/library/unicodedata.rst',
 'python-collections.rst':'https://raw.githubusercontent.com/python/cpython/v3.12.14/Doc/library/collections.rst',
 'python-stdtypes.rst':'https://raw.githubusercontent.com/python/cpython/v3.12.14/Doc/library/stdtypes.rst',
 'cer.py':'https://raw.githubusercontent.com/Lightning-AI/torchmetrics/v1.8.2/src/torchmetrics/functional/text/cer.py',
 'cer-edit.py':'https://raw.githubusercontent.com/Lightning-AI/torchmetrics/v1.8.2/src/torchmetrics/functional/text/helper.py',
 'CC-OCR-README.md':'https://huggingface.co/datasets/wulipc/CC-OCR/raw/c64517e92179991d509776064174776700cdd5a2/README.md',
 'CC-OCR-LICENSE.txt':'https://huggingface.co/datasets/wulipc/CC-OCR/raw/c64517e92179991d509776064174776700cdd5a2/LICENSE',
 'Noto-Sans-OFL.txt':'https://raw.githubusercontent.com/notofonts/noto-cjk/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/LICENSE',
 'Noto-Serif-OFL.txt':'https://raw.githubusercontent.com/notofonts/noto-cjk/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Serif/LICENSE',
 'UnicodeData.txt':'https://www.unicode.org/Public/15.0.0/ucd/UnicodeData.txt'
}
def fetch(pair):
 name,url=pair
 try:
  with urllib.request.urlopen(url,timeout=45) as r:raw=r.read();final=r.url;status=r.status
  digest=hashlib.sha256(raw).hexdigest()
  if name=='UnicodeData.txt':
   raw='\n'.join(line for line in raw.decode().splitlines() if line.split(';')[0] in ['FF21','0041','81FA','53F0','4E00','9FFF']).encode()+b'\n'
  path=OUT/(PREFIX+name);path.write_bytes(raw)
  return {'name':name,'url':url,'final_url':final,'status':status,'downloaded_sha256':digest,'snapshot_sha256':hashlib.sha256(raw).hexdigest(),'snapshot_bytes':len(raw),'path':str(path),'accessed_on':'2026-10-04'}
 except Exception as e:return {'name':name,'url':url,'error':repr(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:result=list(pool.map(fetch,items.items()))
(OUT/(PREFIX+'fetch-receipt.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if any('error' in r for r in result):raise SystemExit(1)
