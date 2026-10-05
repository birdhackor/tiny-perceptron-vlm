"""Read-only source retrieval; print hashes, never result annotations."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
PIN = "1df335318bda03fd771807f66976953231d5a00b"
TORCH = "5c4886908584029761b579af026dcfb627c84070"
jobs = []
for name in ("pretrain", "sft", "joint", "preference"):
    rel = f"docs/course-experiments/results/capstone_{name}.json"
    jobs.append((f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{PIN}/{rel}", rel, ART / "remote" / rel))
for stage in ("pretrain", "sft", "joint", "dpo"):
    rel = f"docs/course-experiments/capstone-evidence/{stage}/validation.json"
    jobs.append((f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{PIN}/{rel}", rel, ART / "remote" / rel))
for rel in ("tiny_perceptron/capstone.py", "scripts/course_experiments/capstone.py"):
    jobs.append((f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{PIN}/{rel}", rel, ART / "remote" / rel))
for rel, name in (("torch/nn/functional.py", "torch-functional.py"), ("docs/source/notes/autograd.md", "torch-autograd.md"), ("torch/random.py", "torch-random.py")):
    jobs.append((f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/{rel}", None, ART / "authority" / name))

def fetch(job):
    url, local, dest = job
    try:
        with urlopen(url, timeout=30) as response:
            raw = response.read()
            result = {"url": url, "http_status": response.status, "accessed_on": "2026-10-05", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        if local:
            result["local_path"] = local
            result["local_sha256"] = hashlib.sha256((ROOT / local).read_bytes()).hexdigest()
            result["same_as_local"] = raw == (ROOT / local).read_bytes()
        elif dest.exists():
            result["same_as_prior_original_copy"] = raw == dest.read_bytes()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        result["saved"] = str(dest.relative_to(ART))
        return result
    except Exception as error:
        return {"url": url, "error_type": type(error).__name__, "error": str(error)}

with ThreadPoolExecutor(max_workers=5) as pool:
    results = list(pool.map(fetch, jobs))
(ART / "fetch-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
for result in results:
    print(json.dumps(result, ensure_ascii=False))
