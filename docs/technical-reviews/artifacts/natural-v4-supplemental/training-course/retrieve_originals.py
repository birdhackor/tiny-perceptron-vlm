from pathlib import Path
import hashlib, json, urllib.request, subprocess, concurrent.futures

ROOT = Path(__file__).resolve().parents[5]
DEST = ROOT / 'outputs/natural-v4/factual-research/training-course'
DEST.mkdir(parents=True, exist_ok=True)
sources = {
 'dpo': 'https://arxiv.org/pdf/2305.18290v1',
 'ppo': 'https://arxiv.org/pdf/1707.06347v1',
 'lora': 'https://arxiv.org/pdf/2106.09685v1',
 'distillation': 'https://arxiv.org/pdf/1503.02531v1',
 'rlhf': 'https://arxiv.org/pdf/2203.02155v1',
 'rope': 'https://arxiv.org/pdf/2104.09864v1',
 'rmsnorm': 'https://arxiv.org/pdf/1910.07467v1',
 'swiglu': 'https://arxiv.org/pdf/2002.05202v1',
 'switch': 'https://arxiv.org/pdf/2101.03961v1',
 'passk': 'https://arxiv.org/pdf/2107.03374v1',
 'ce': 'https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html',
 'sdpa': 'https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html',
 'amp': 'https://docs.pytorch.org/docs/2.14/amp.html',
 'memory': 'https://docs.pytorch.org/docs/2.14/generated/torch.cuda.max_memory_allocated.html',
 'tf32': 'https://docs.pytorch.org/docs/2.14/notes/cuda.html',
 'checkpoint': 'https://docs.pytorch.org/docs/2.14/checkpoint.html',
}
def fetch(item):
    key, url = item
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'Factual-review/1.0'}), timeout=40) as response:
            data = response.read()
            resolved = response.url
            content_type = response.headers.get('content-type')
        path = DEST / (key + ('.pdf' if data[:4] == b'%PDF' else '.html'))
        path.write_bytes(data)
        if path.suffix == '.pdf':
            subprocess.run(['pdftotext','-layout',str(path),str(path.with_suffix('.txt'))], check=True)
        else:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(data,'html.parser')
            main = soup.select_one('article') or soup.select_one('main') or soup
            path.with_suffix('.txt').write_text(main.get_text('\n',strip=True),encoding='utf-8')
        return {'id':key,'url':url,'resolved_url':resolved,'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'content_type':content_type,'accessed_on':'2026-10-04','status':'retrieved'}
    except Exception as e:
        return {'id':key,'url':url,'status':'retrieval_failed','error':repr(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    receipts = list(pool.map(fetch, sources.items()))
(Path(__file__).parent/'original-retrieval-receipts.json').write_text(json.dumps(receipts, indent=2)+'\n')
print(json.dumps(receipts,indent=2))
