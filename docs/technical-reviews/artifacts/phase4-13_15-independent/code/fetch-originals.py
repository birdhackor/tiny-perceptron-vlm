import urllib.request, concurrent.futures, json, hashlib
from pathlib import Path
A=Path(__file__).resolve().parents[1]
revision='5c4886908584029761b579af026dcfb627c84070'
items={
 'ppo-v2.pdf':'https://arxiv.org/pdf/1707.06347v2',
 'instructgpt-v1.pdf':'https://arxiv.org/pdf/2203.02155v1',
 'published-posttraining.json':'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/posttraining.json',
}
for file,path in {'categorical.py':'torch/distributions/categorical.py','grad_mode.py':'torch/autograd/grad_mode.py','optimizer.py':'torch/optim/optimizer.py','sgd.py':'torch/optim/sgd.py','tensor.py':'torch/_tensor.py','functional.py':'torch/nn/functional.py','torch_docs.py':'torch/_torch_docs.py','LossNLL.cpp':'aten/src/ATen/native/LossNLL.cpp','module.py':'torch/nn/modules/module.py'}.items(): items[file]='https://raw.githubusercontent.com/pytorch/pytorch/'+revision+'/'+path

def fetch(pair):
 name,url=pair; data=urllib.request.urlopen(url,timeout=35).read(); (A/'sources'/name).write_bytes(data)
 return {'file':name,'url':url,'accessed_on':'2026-10-05','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex: result=list(ex.map(fetch,items.items()))
(A/'sources/retrieval.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
