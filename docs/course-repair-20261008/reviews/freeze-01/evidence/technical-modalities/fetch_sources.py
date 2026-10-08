import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

dest = Path(__file__).parent / "original-sources"
dest.mkdir(exist_ok=True)
sources = {
    "clip-radford21a.pdf": "https://proceedings.mlr.press/v139/radford21a/radford21a.pdf",
    "torchaudio-functional-v2.8.0.py": "https://raw.githubusercontent.com/pytorch/audio/v2.8.0/src/torchaudio/functional/functional.py",
    "scipy-signaltools-v1.16.1.py": "https://raw.githubusercontent.com/scipy/scipy/v1.16.1/scipy/signal/_signaltools.py",
    "smith-sampling-theorem.html": "https://ccrma.stanford.edu/~jos/resample/Sampling_Theorem.html",
    "dpo-2305.18290v2.pdf": "https://arxiv.org/pdf/2305.18290v2",
}

def fetch(entry):
    name, url = entry
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read()
            final_url = response.url
        (dest / name).write_bytes(data)
        return {"path": str(dest / name), "url": url, "final_url": final_url,
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "downloaded": True}
    except Exception as exc:
        return {"url": url, "downloaded": False, "error": repr(exc)}

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    rows = list(pool.map(fetch, sources.items()))
(dest / "fetch-manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
