from pathlib import Path
import urllib.request,hashlib,json,concurrent.futures
ROOT=Path(__file__).resolve().parents[5];D=ROOT/'outputs/natural-v4/factual-research/training-course'
urls={
 'tinystories-card':'https://huggingface.co/datasets/roneneldan/TinyStories/raw/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md',
 'poetry-license':'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/b8594f81a89752241442f2ce267d6f66f96704ee/LICENSE',
 'fsdd-readme':'https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/26eb9aaf76e81b692f806f9140c2d2777410d7a1/README.md',
 'pku-card':'https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF/raw/9421ffafec3fa40a1f1a7d567b4d525079477ecb/README.md',
 'fashion-readme':'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/README.md',
 'gsm-readme':'https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/README.md',
 'public-models-card':'https://huggingface.co/birdhackor/tiny-perceptron-course-models/raw/fbbff36990db0d95a6e0af5ecdbc593f920938d8/README.md',
}
def fetch(kv):
 k,u=kv
 try:
  with urllib.request.urlopen(u,timeout=30) as r:b=r.read(150001);resolved=r.url
  assert len(b)<=150000
  p=D/(k+'.md');p.write_bytes(b)
  return {'id':k,'url':u,'resolved_url':resolved,'status':'retrieved','path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'accessed_on':'2026-10-04'}
 except Exception as e:return {'id':k,'url':u,'status':'retrieval_failed','error':repr(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as p:rs=list(p.map(fetch,urls.items()))
(Path(__file__).parent/'card-retrieval-receipts.json').write_text(json.dumps(rs,indent=2)+'\n');print(json.dumps(rs,indent=2))
