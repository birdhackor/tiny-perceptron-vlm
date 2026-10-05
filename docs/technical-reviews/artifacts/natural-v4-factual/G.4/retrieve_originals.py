from pathlib import Path
import concurrent.futures,urllib.request,hashlib,json,subprocess
root=Path(__file__).resolve().parents[5]
research=root/'outputs/natural-v4/factual-research/G.4'
research.mkdir(parents=True,exist_ok=True)
public=Path(__file__).resolve().parent
specs=[
 ('moe','https://arxiv.org/pdf/1701.06538v1','1701.06538v1.pdf'),
 ('flash','https://arxiv.org/pdf/2205.14135v2','2205.14135v2.pdf'),
 ('quant','https://arxiv.org/pdf/1712.05877v1','1712.05877v1.pdf'),
 ('hinton','https://arxiv.org/pdf/1503.02531v1','1503.02531v1.pdf'),
 ('minillm','https://arxiv.org/pdf/2306.08543v6','2306.08543v6.pdf'),
 ('ppo','https://arxiv.org/pdf/1707.06347v2','1707.06347v2.pdf'),
 ('instruct','https://arxiv.org/pdf/2203.02155v1','2203.02155v1.pdf'),
 ('dpo','https://arxiv.org/pdf/2305.18290v3','2305.18290v3.pdf'),
 ('rag','https://arxiv.org/pdf/2005.11401v4','2005.11401v4.pdf'),
 ('cache','https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation','cache-v4.57.1.html'),
 ('tool','https://developers.openai.com/api/docs/guides/function-calling','function-calling.html'),
 ('safety','https://model-spec.openai.com/2025-12-18.html','model-spec-2025-12-18.html')]
def fetch(spec):
 ident,url,name=spec
 row={'id':ident,'url':url,'accessed_on':'2026-10-04'}
 try:
  cached=root/'outputs/natural-v4/factual-original-cache'/name
  if ident in ('instruct','dpo') and cached.exists():
   data=cached.read_bytes();row['retrieval_method']='personally read shared original PDF bytes only; no donor analysis'
  else:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=40) as response:
    data=response.read();row['response_url']=response.url;row['content_type']=response.headers.get('Content-Type');row['retrieval_method']='HTTPS original retrieval'
  path=research/name;path.write_bytes(data)
  row.update(path=str(path.relative_to(root)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),retrieval_status='retrieved')
  if name.endswith('.pdf'):
   subprocess.run(['pdftotext','-layout',str(path),str(path.with_suffix('.txt'))],check=True,capture_output=True)
   row['first_lines']=path.with_suffix('.txt').read_text().splitlines()[:5]
  print(ident,row['retrieval_status'],row['bytes'],flush=True)
 except Exception as error:
  row.update(retrieval_status='failed',error=repr(error));print(ident,'FAILED',repr(error),flush=True)
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 rows=list(pool.map(fetch,specs))
(public/'original-retrieval-receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
