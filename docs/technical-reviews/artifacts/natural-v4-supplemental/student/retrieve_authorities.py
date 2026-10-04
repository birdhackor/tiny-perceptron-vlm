"""Read-only retrieval of the original official authorities; no installs/models."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "outputs/natural-v4/factual-research/student"
OUT.mkdir(parents=True, exist_ok=True)
URLS = {
    "git-clone": "https://git-scm.com/docs/git-clone",
    "git-lfs": "https://raw.githubusercontent.com/git-lfs/git-lfs/v3.7.1/docs/man/git-lfs-config.adoc",
    "python-venv": "https://docs.python.org/3.12/library/venv.html",
    "pytorch-install": "https://pytorch.org/get-started/previous-versions/",
    "torch-init": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/__init__.py",
    "torch-cuda": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/cuda/__init__.py",
    "qwen-card": "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/README.md",
    "whisper-card": "https://huggingface.co/openai/whisper-large-v3-turbo/raw/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/README.md",
    "hf-download": "https://raw.githubusercontent.com/huggingface/huggingface_hub/v0.36.2/src/huggingface_hub/file_download.py",
    "transformers-modeling": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/modeling_utils.py",
    "rfc1122": "https://www.rfc-editor.org/rfc/rfc1122.txt",
    "python-hashlib": "https://docs.python.org/3.12/library/hashlib.html",
}

def retrieve(item):
    name, url = item
    record = {"id": name, "url": url, "accessed_on": datetime.datetime.now(datetime.UTC).isoformat()}
    try:
        response = requests.get(url, timeout=35)
        record.update(status_code=response.status_code, final_url=response.url)
        raw = response.content
        path = OUT / (name + (".html" if "text/html" in response.headers.get("Content-Type", "") else ".txt"))
        path.write_bytes(raw)
        record.update(path=str(path.relative_to(ROOT)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        response.raise_for_status()
        body = BeautifulSoup(raw, "html.parser").get_text("\n", strip=True) if path.suffix == ".html" else raw.decode("utf-8")
        textpath = OUT / (name + ".readable.txt")
        textpath.write_text(body, encoding="utf-8")
        record["readable_path"] = str(textpath.relative_to(ROOT))
    except Exception as exc:
        record["error"] = repr(exc)
    return record

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    records = list(pool.map(retrieve, URLS.items()))
receipt = Path(__file__).with_name("authority-retrieval.json")
receipt.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(records, ensure_ascii=False, indent=2))
