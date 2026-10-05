"""Read-only authority retrieval; no dataset, model, or repository report download."""
import hashlib
import json
from pathlib import Path
import sys
import requests

BASE = Path(__file__).resolve().parent
SOURCES = {
    "minds14-readme.md": "https://huggingface.co/datasets/PolyAI/minds14/raw/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md",
    "minds14-loader.py": "https://huggingface.co/datasets/PolyAI/minds14/raw/40ce77cb32a384e4d50a568e1ec39ac804019d33/minds14.py",
    "minds14-2104.08524v1.pdf": "https://arxiv.org/pdf/2104.08524v1",
    "slu-1904.03670v1.pdf": "https://arxiv.org/pdf/1904.03670v1",
    "librosa-0.11.0-melspectrogram.html": "https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html",
    "librosa-0.11.0-power_to_db.html": "https://librosa.org/doc/0.11.0/generated/librosa.power_to_db.html",
    "sklearn-1.7-GroupKFold.html": "https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GroupKFold.html",
    "pytorch-2.9-CrossEntropyLoss.html": "https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html",
    "pytorch-2.9-Linear.html": "https://docs.pytorch.org/docs/2.9/generated/torch.nn.Linear.html",
}

def main():
    target = sys.argv[1]
    url = SOURCES[target]
    out = BASE / "sources"
    out.mkdir(exist_ok=True)
    try:
        response = requests.get(url, timeout=40)
        receipt = {
            "requested_url": url, "resolved_url": response.url,
            "http_status": response.status_code,
            "accessed_on": "2026-10-05", "tls_verification": "requests default verify=True",
            "requests_version": requests.__version__, "python": sys.version,
        }
        if response.status_code == 200:
            content = response.content
            (out / target).write_bytes(content)
            receipt.update(path=str(out / target), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    except requests.RequestException as error:
        receipt = {"requested_url": url, "exception": type(error).__name__, "message": str(error)}
    (out / (target + ".receipt.json")).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))

if __name__ == "__main__":
    main()
