"""Read original public sources and retain versioned, small evidence snapshots."""

import concurrent.futures
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
TMP = Path("/tmp/fact_v2_18_01")
TMP.mkdir(exist_ok=True)
PREFIX = "fact_v2_18_01_"


def fetch(item):
    name, url = item
    try:
        with urllib.request.urlopen(url, timeout=25) as response:
            data = response.read()
            parsed = urlsplit(response.url)
            final_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
            result = {"name": name, "url": url, "final_url_without_query": final_url,
                      "status": response.status, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()}
        (TMP / name).write_bytes(data)
        if name.endswith(".json"):
            (OUT / (PREFIX + name)).write_bytes(data)
        if name.endswith(".pdf"):
            subprocess.run(["pdftotext", "-layout", str(TMP / name),
                            str(TMP / name.replace(".pdf", ".txt"))], check=True)
        return result
    except Exception as error:
        return {"name": name, "url": url, "error": repr(error)}


def main():
    release = json.loads((ROOT / "docs/course-experiments/public-releases/distillation.json").read_text())
    manifest = release["release"]["public_manifest"]
    hf = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/"
    requests = [("minillm.pdf", "https://arxiv.org/pdf/2306.08543v6"),
                ("hinton.pdf", "https://arxiv.org/pdf/1503.02531v1")]
    torch_root = f"https://raw.githubusercontent.com/pytorch/pytorch/{torch.version.git_version}/torch/"
    requests += [(name, torch_root + name) for name in ("_torch_docs.py", "_tensor_docs.py", "nn/functional.py")]
    requests += [(row["output"], hf + row["path"]) for row in manifest["files"]
                 if row["output"] in ("sft-teacher.pt", "style-teacher.pt", "moe-teacher.pt", "export-manifest.json", "evaluation.json")]
    # The slash in functional.py is used as a temporary directory only.
    (TMP / "nn").mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, requests))
    for item in results:
        if "sha256" in item and item["name"].endswith(".pt"):
            row = next(row for row in manifest["files"] if row["output"] == item["name"])
            item["manifest_sha256"] = row["sha256"]
            item["hash_match"] = row["sha256"] == item["sha256"]
    (OUT / (PREFIX + "fetch_receipt.json")).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
