from pathlib import Path
import concurrent.futures
import hashlib
import json
import urllib.request

OUT = Path(__file__).resolve().parent / 'sources'
OUT.mkdir(exist_ok=True)
torch_commit = '5c4886908584029761b579af026dcfb627c84070'
targets = {
    'vaswani-1706.03762v7.pdf': 'https://arxiv.org/pdf/1706.03762v7',
    'deeplearning-chapter5-ml.html': 'https://www.deeplearningbook.org/contents/ml.html',
    'deeplearning-chapter6-mlp.html': 'https://www.deeplearningbook.org/contents/mlp.html',
    'torch-linear.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{torch_commit}/torch/nn/modules/linear.py',
    'torch-normalization.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{torch_commit}/torch/nn/modules/normalization.py',
    'torch-sparse.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{torch_commit}/torch/nn/modules/sparse.py',
    'torch-module.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{torch_commit}/torch/nn/modules/module.py',
    'torch-tensor-docs.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{torch_commit}/torch/_tensor_docs.py',
}

def fetch(item):
    name,url=item
    request = urllib.request.Request(url, headers={'User-Agent':'Independent factual review of course section 4.8'})
    with urllib.request.urlopen(request,timeout=35) as response:
        raw=response.read()
        info={'name':name,'url':url,'final_url':response.url,'status':response.status,'headers':dict(response.headers),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'accessed_on':'2026-10-05'}
    (OUT/name).write_bytes(raw)
    return info

results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for item,result in zip(targets.items(),pool.map(lambda item: fetch(item),targets.items())):
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='headers'}))
(OUT/'fetch-receipt.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
