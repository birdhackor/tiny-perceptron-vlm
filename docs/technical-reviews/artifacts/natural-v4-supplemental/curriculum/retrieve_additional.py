from pathlib import Path
import concurrent.futures, hashlib, json, urllib.request
ROOT=Path(__file__).resolve().parents[5]
HERE=Path(__file__).resolve().parent
RAW=ROOT/'outputs/natural-v4/factual-research/curriculum'
urls={
 'miniv-language':'https://raw.githubusercontent.com/jingyaogong/minimind-v/1862b633fc082a723e78dbead9777545f09d960c/model/model_minimind.py',
 'hf-v1-tree':'https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/fbbff36990db0d95a6e0af5ecdbc593f920938d8/course/course-v1?recursive=true&expand=false&limit=1000',
 'hf-capstone-tree':'https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed/course/course-integration-v2?recursive=true&expand=false&limit=1000',
 'qwen-card':'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/README.md',
 'qwen-config':'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/config.json',
 'whisper-card':'https://huggingface.co/openai/whisper-large-v3-turbo/raw/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/README.md',
 'whisper-config':'https://huggingface.co/openai/whisper-large-v3-turbo/raw/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/config.json',
 'literal-preview-source':'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/f8ca78bc3ffef82b0faa352c464909b76bfd8976/docs/curriculum.md',
}
for name in ['torch.nn.Module','torch.nn.CrossEntropyLoss','torch.nn.Embedding','torch.nn.LayerNorm','torch.optim.AdamW','torch.optim.Adam','torch.optim.SGD','torch.utils.checkpoint.checkpoint','torch.compile']:
 urls[name]=f'https://pytorch.org/docs/2.14/generated/{name}.html'
urls['autograd']='https://pytorch.org/docs/2.14/notes/autograd.html'
urls['amp']='https://pytorch.org/docs/2.14/amp.html'
urls['tokenizers']='https://huggingface.co/docs/tokenizers/api/models'
urls['unicode']='https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/'
def get(pair):
 name,url=pair;p=RAW/f'{name}.txt';row={'id':name,'url':url,'accessed_on':'2026-10-04','path':str(p.relative_to(ROOT))}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'factual-review/1.0'}),timeout=30) as response:
   b=response.read();row.update(status=response.status,final_url=response.url)
  p.write_bytes(b);row.update(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
 except Exception as e:row['error']=type(e).__name__+': '+str(e)
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: rows=list(pool.map(get,urls.items()))
(HERE/'retrieval.additional.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(rows,ensure_ascii=False,indent=2))
