from pathlib import Path
import concurrent.futures, hashlib, json, subprocess, urllib.request

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / 'outputs/natural-v4/factual-research/entries'
OUT.mkdir(parents=True, exist_ok=True)
URLS = {
 'attention.pdf':'https://arxiv.org/pdf/1706.03762',
 'switch.pdf':'https://arxiv.org/pdf/2101.03961',
 'distillation.pdf':'https://arxiv.org/pdf/1503.02531',
 'torch-autograd.html':'https://docs.pytorch.org/docs/2.14/autograd.html',
 'uv-projects.html':'https://docs.astral.sh/uv/concepts/projects/sync/',
 'ipykernel.html':'https://ipython.readthedocs.io/en/stable/install/kernel_install.html',
 'git-lfs.md':'https://raw.githubusercontent.com/git-lfs/git-lfs/v3.7.0/docs/man/git-lfs-config.adoc',
 'poetry-license':'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/b8594f81a89752241442f2ce267d6f66f96704ee/LICENSE',
 'fashion-license':'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/LICENSE',
 'fashion-readme.md':'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/README.md',
 'fsdd-readme.md':'https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/26eb9aaf76e81b692f806f9140c2d2777410d7a1/README.md',
 'gsm8k-readme.md':'https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/README.md',
 'gsm8k-license':'https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE',
 'tiny-card.md':'https://huggingface.co/datasets/roneneldan/TinyStories/raw/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md',
 'ultrachat-card.md':'https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/raw/8049631c405ae6576f93f445c6b8166f76f5505a/README.md',
 'ultrafeedback-card.md':'https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/raw/3949bf5f8c17c394422ccfab0c31ea9c20bdeb85/README.md',
 'pku-card.md':'https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF/raw/9421ffafec3fa40a1f1a7d567b4d525079477ecb/README.md',
 'qwen-card.md':'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/README.md',
 'qwen-config.json':'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/config.json',
 'whisper-card.md':'https://huggingface.co/openai/whisper-large-v3-turbo/raw/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/README.md',
 'assets-run.json':'https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/actions/runs/36946663981',
 'ci-run.json':'https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/actions/runs/37071683289',
 'hf-course-tree.json':'https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed/course/course-integration-v2?recursive=true&expand=false',
}
def fetch(pair):
 name,url = pair
 row={'name':name,'url':url,'accessed_on':'2026-10-04'}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Independent factual reviewer'}),timeout=15) as r:
   data=r.read(8*1024*1024+1)
   assert len(data)<=8*1024*1024
   (OUT/name).write_bytes(data)
   row.update(status=r.status,final_url=r.url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  if name.endswith('.pdf'):
   cp=subprocess.run(['pdftotext','-layout',str(OUT/name),str(OUT/(name+'.txt'))],capture_output=True,text=True)
   row['text_conversion_exit']=cp.returncode
 except Exception as e: row['error']=str(e)
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 rows=list(pool.map(fetch,URLS.items()))
(Path(__file__).parent/'authority-retrieval.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows: print(row['name'],row.get('status',row.get('error')),row.get('sha256',''))
