"""Anonymous published bytes acquisition only; no model is constructed or run."""
from pathlib import Path
import datetime,hashlib,json,platform,sys,urllib.request
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from scripts.omni_revision.public_download import descriptor,download
BASE=Path('/workspace/work/reference/omni-P5-training-whole-originals-v1')
manifest=ROOT/'docs/omni-products/v1/public-release.json';spec=descriptor(manifest)
url=f"https://api.github.com/repos/{spec['repository']}/releases/tags/{spec['release_tag']}"
request=urllib.request.Request(url,headers={'Accept':'application/vnd.github+json','User-Agent':'P5-training-independent-review'})
with urllib.request.urlopen(request,timeout=45) as response:raw=response.read()
(BASE/'provider-release.json').write_bytes(raw);provider=json.loads(raw)
assets={row['name']:row for row in provider['assets']};observed={}
for kind,row in spec['components'].items():
 p=assets[row['name']];assert p['id']==row['asset_id'] and p['size']==row['bytes'] and p['digest']=='sha256:'+row['sha256']
 observed[kind]={'id':p['id'],'name':p['name'],'bytes':p['size'],'digest':p['digest'],'state':p['state'],'browser_download_url':p['browser_download_url']}
receipt=download(manifest,BASE/'public-installed-v1')
assert set(receipt['assets'])=={'av','choice','codec','talker'}
assert not any(name.endswith('.pt') for row in receipt['assets'].values() for name in row['files'])
result={'created_at':datetime.datetime.now(datetime.UTC).isoformat(),'environment':{'python':platform.python_version(),'device':'no model execution; CPU-compatible loader only inspected'},'provider':{'url':url,'release_id':provider['id'],'tag_name':provider['tag_name'],'draft':provider['draft'],'published_at':provider['published_at'],'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'assets':observed},'download_receipt':receipt,'scope':'Anonymous GET/install/digest only. Archives and weights remain Git-external. Full continuation .pt is skipped, no model inference and no final family reopened. Native Windows T.11 remains pending book PR CI.'}
p=Path(__file__).parent/'public-download-results-v1.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'output':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'components':list(receipt['assets'])}))
