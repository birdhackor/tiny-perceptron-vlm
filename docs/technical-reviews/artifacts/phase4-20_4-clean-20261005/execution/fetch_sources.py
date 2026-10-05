import requests,json,hashlib,concurrent.futures
from pathlib import Path
base=Path(__file__).resolve().parents[1]/"sources"
urls={
"docci-readme":"https://raw.githubusercontent.com/google/docci/c190366529943195d90c141899abb2ec68dc345d/README.md",
"docci-hf-card":"https://huggingface.co/datasets/google/docci/resolve/a0a43eaf34676ffd008fb6565dd8c2ba00d09100/README.md",
"ocr-card":"https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1/resolve/69696a1cc543ef3a0f8e9892a89c17293e915263/README.md",
"oasst2-card":"https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/179dd21fc55192153d94adb0e0ce8f69e222bf75/README.md",
"fleurs-card":"https://huggingface.co/datasets/google/fleurs/resolve/d7c758a6dceecd54a98cac43404d3d576e721f07/README.md",
"aishell-openslr":"https://www.openslr.org/33",
"docci-website":"https://raw.githubusercontent.com/google/docci/c190366529943195d90c141899abb2ec68dc345d/index.html"}
def fetch(k,u):
 r=requests.get(u,timeout=20);r.raise_for_status();path=base/(k+".txt");path.write_bytes(r.content)
 return {"id":k,"url":u,"final_url":r.url,"status":r.status_code,"accessed_on":"2026-10-05","sha256":hashlib.sha256(r.content).hexdigest(),"bytes":len(r.content),"path":str(path)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 results=list(pool.map(lambda kv:fetch(*kv),urls.items()))
(base/"retrieval.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
for r in results: print(r["id"],r["status"],r["bytes"],r["sha256"])
