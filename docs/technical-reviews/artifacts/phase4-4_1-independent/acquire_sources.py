"""Preserve original sources and raw lesson bytes for independent 4.1 review."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, UTC

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SHA = lambda b: hashlib.sha256(b).hexdigest()

original = OUT / "original"
original.mkdir(exist_ok=True)
for name in ("section.md", "fence-1.py", "bootstrap.py", "extraction.json", "execution.json", "stdout.txt", "stderr.txt", "environment.json"):
    shutil.copyfile(Path("/tmp/phase4-4_1-independent-original") / name, original / name)
shutil.copyfile(ROOT / "docs/review-tools/section_facts.py", original / "section_facts.py")
raw = (ROOT / "course/chapters/04.md").read_bytes()
intro = raw[:raw.index(b"## 4.1 ")]
(OUT / "chapter-intro.md").write_bytes(intro)
(OUT / "intro-receipt.json").write_text(json.dumps({
    "source": "course/chapters/04.md", "read_range": "raw UTF-8 bytes from chapter start to immediately before ## 4.1",
    "sha256": SHA(intro), "bytes": len(intro),
    "intro_summary": "本章把字與位置特徵、因果注意力、各位置的特徵加工與候選字打分串成可重複的 Transformer 層；先採完整 Dense 配方，檢查輸入、答案與求導，再討論層數。",
    "reviewer_task": "/root/phase4_factual_coordinator/factual_4_1",
    "policy": "Original bytes; no newline normalization; independently read current introduction.",
}, ensure_ascii=False, indent=2) + "\n")
shutil.copyfile(ROOT / "tiny_perceptron/model.py", OUT / "model.py")

SOURCES = {
    "vaswani-2017-v7.pdf": "https://arxiv.org/pdf/1706.03762v7",
    "nope-2022-v1.pdf": "https://arxiv.org/pdf/2203.16634v1",
    "gpt2-model.py": "https://raw.githubusercontent.com/openai/gpt-2/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py",
    "pytorch-sparse.py": "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/sparse.py",
    "pytorch-torch-docs.py": "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_torch_docs.py",
    "pytorch-tensor-docs.py": "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_tensor_docs.py",
    "pytorch-grad-mode.py": "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/autograd/grad_mode.py",
}
(OUT / "sources").mkdir(exist_ok=True)
def fetch(item):
    name, url = item
    with urllib.request.urlopen(url, timeout=35) as response:
        blob = response.read()
        receipt = {"filename": name, "url": url, "resolved_url": response.url,
                   "status": response.status, "sha256": SHA(blob), "bytes": len(blob),
                   "accessed_at": datetime.now(UTC).isoformat(), "content_type": response.headers.get("Content-Type")}
    (OUT / "sources" / name).write_bytes(blob)
    return receipt
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
    receipts = list(executor.map(fetch, SOURCES.items()))
(OUT / "sources/acquisition.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps({"sources": receipts, "intro_sha256": SHA(intro), "original_helper_exit": json.loads((original / "execution.json").read_text())["exit_code"]}, indent=2))
