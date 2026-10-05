import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCES = {
 'hf-causal-lm-v4.57.1.md': 'https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/tasks/language_modeling.md',
 'hf-perplexity-v4.57.1.md': 'https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/perplexity.md',
 'sklearn-group-split-1.7.2.rst': 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst',
 'sklearn-pitfalls-1.7.2.rst': 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/common_pitfalls.rst',
 'python-3.13-sequence.rst': 'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst',
 'pytorch-tolist-v2.9.0.py': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_tensor_docs.py',
 'hf-tokenizers-bpe-v0.22.1.rs': 'https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.1/tokenizers/src/models/bpe/model.rs',
 'gpt3-arxiv-v1.pdf': 'https://arxiv.org/pdf/2005.14165v1',
}
(OUT/'sources').mkdir(exist_ok=True)
def get(item):
 name,url=item
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'independent-factual-review/6.8'}),timeout=45) as response:
  raw=response.read()
  receipt={'file':name,'url':url,'resolved_url':response.url,'accessed_on':'2026-10-05','http_status':response.status,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 (OUT/'sources'/name).write_bytes(raw)
 return receipt
receipts=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for future in concurrent.futures.as_completed([pool.submit(get,item) for item in SOURCES.items()]):
  try:
   receipts.append(future.result())
  except Exception as error:
   receipts.append({'error':str(error)})
(OUT/'sources/acquisition.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps(receipts,indent=2))
assert all('error' not in r for r in receipts)
