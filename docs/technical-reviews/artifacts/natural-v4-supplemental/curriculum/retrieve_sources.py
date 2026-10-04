"""Fresh original-source retrieval; long raw material remains in ignored outputs."""
from pathlib import Path
import concurrent.futures, hashlib, json, re, shutil, sys, urllib.request

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
RAW = ROOT / 'outputs/natural-v4/factual-research/curriculum'
RAW.mkdir(parents=True, exist_ok=True)
report = HERE / 'report.json'
if report.exists() and not (HERE / 'preexisting-report-unread.json').exists():
    shutil.copyfile(report, HERE / 'preexisting-report-unread.json')
shutil.copyfile(ROOT / 'docs/curriculum.md', HERE / 'curriculum.snapshot.md')
urls = list(dict.fromkeys(re.findall(r'https://github.com/[^)\s]+/blob/[^)\s]+', (ROOT/'docs/curriculum.md').read_text())))
for repo, version, paths in [
 ('stanford-cs336/lectures','de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e',['lecture_12.py','lecture_14.py','lecture_17.py']),
 ('stanford-cs336/stanford-cs336.github.io','25d740fd9060cc6613163b5b88ca88a5f64138ff',['index.html','spring2025/index.html']),
 ('Harvard-IACS/2025-AC215','a0d3ed05aa749aa46c95c70a2f519db0eef2bb30',['README.md','_modules/week-06.md']),
 ('Harvard-IACS/2026-AC215','f00978fe6d008935ab87c7fda589b461bec6d236',['README.md']),
 ('MITDeepLearning/introtodeeplearning.com','c3841a160a77d336391107cc71fd3b80555bfa22',['2025/index.html','2026/index.html']),
 ('MITDeepLearning/introtodeeplearning','535e0862f52c81288258fae0516c40968d07f603',['lab3/LLM_Finetuning.ipynb','lab2/PT_Part2_Debiasing.ipynb']),
]:
    urls.extend(f'https://github.com/{repo}/blob/{version}/{p}' for p in paths)
papers = {'transformer':'1706.03762v1','dpo':'2305.18290v1','ppo':'1707.06347v1','switch':'2101.03961v1','distill':'1503.02531v1','lora':'2106.09685v1','rope':'2104.09864v1','rmsnorm':'1910.07467v1','clip':'2103.00020v1','qknorm':'2010.04245v1','glu':'2002.05202v1','gqa':'2305.13245v1','flash':'2205.14135v1','qlora':'2305.14314v1','rag':'2005.11401v1','icl':'2005.14165v1'}
urls.extend(f'https://arxiv.org/pdf/{p}' for p in papers.values())
urls.extend(['https://pytorch.org/docs/2.14/generated/torch.nn.Linear.html','https://pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html','https://pytorch.org/docs/2.14/generated/torch.stft.html','https://pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html'])

def retrieve(pair):
    i, url = pair
    raw_url = url.replace('https://github.com/','https://raw.githubusercontent.com/').replace('/blob/','/')
    suffix = '.pdf' if '/pdf/' in url else '.txt'
    p = RAW / f'original-{i:02d}{suffix}'
    row = {'id':f'o{i:02d}', 'url':url, 'retrieval_url':raw_url, 'accessed_on':'2026-10-04', 'path':str(p.relative_to(ROOT))}
    try:
        req = urllib.request.Request(raw_url, headers={'User-Agent':'factual-review/1.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            row.update(status=response.status, final_url=response.url, headers={k:response.headers.get(k) for k in ['ETag','Last-Modified','Content-Type']})
        p.write_bytes(raw)
        row.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    except Exception as exc:
        row['error'] = type(exc).__name__ + ': ' + str(exc)
    return row

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    records=list(pool.map(retrieve, enumerate(urls)))
(HERE/'retrieval.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(records,ensure_ascii=False,indent=2))
