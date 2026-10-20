import concurrent.futures
import hashlib
import html
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ART = Path(__file__).resolve().parent
SPECS = [
    ("lora", "https://arxiv.org/html/2106.09685v2", "arXiv:2106.09685v2", ["For a pre-trained weight", "We still need the 350GB"]),
    ("splitting", "https://scikit-learn.org/1.9/modules/cross_validation.html", "scikit-learn 1.9 documentation", ["test set", "GroupKFold"]),
    ("docci", "https://arxiv.org/html/2404.19753v1", "arXiv:2404.19753v1", ["All 15k annotated", "We grouped images into 149"]),
    ("docci_license", "https://google.github.io/docci/", "publisher website; response SHA-256 recorded", ["The annotations and images are licensed"]),
    ("nvidia", "https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1/raw/69696a1cc543ef3a0f8e9892a89c17293e915263/README.md", "69696a1cc543ef3a0f8e9892a89c17293e915263", ["Large-scale synthetically", "Dataset Governing Terms"]),
    ("oasst2", "https://huggingface.co/datasets/OpenAssistant/oasst2/raw/179dd21fc55192153d94adb0e0ce8f69e222bf75/README.md", "179dd21fc55192153d94adb0e0ce8f69e222bf75", ["license: apache-2.0", "message_tree_id"]),
    ("fleurs", "https://huggingface.co/datasets/google/fleurs/raw/d7c758a6dceecd54a98cac43404d3d576e721f07/README.md", "d7c758a6dceecd54a98cac43404d3d576e721f07", ["license:", "particular focus on read-speech"]),
    ("aishell", "https://www.openslr.org/33/", "SLR33 publisher website; response SHA-256 recorded", ["Apache License", "400 people"]),
    ("cer", "https://huggingface.co/spaces/evaluate-metric/cer/raw/main/cer.py", "metric implementation main; response SHA-256 recorded", ["cer = (s + d + i)", "substitutions", "Character error rate can be computed"]),
    ("lfs", "https://git-lfs.github.com/spec/v1", "Git LFS pointer specification; response SHA-256 recorded", ["version", "oid"]),
    ("pillow", "https://pillow.readthedocs.io/en/stable/reference/Image.html", "Pillow stable Image API; response SHA-256 recorded", ["Returns a rectangular region", "Returns a resized copy"]),
]


def fetch(spec):
    identifier, url, version, terms = spec
    cmd = ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2", "--max-time", "45", url]
    proc = subprocess.run(cmd, capture_output=True)
    raw = proc.stdout
    text = raw.decode(errors="replace")
    if "<html" in text[:2000].lower() or "<!doctype html" in text[:2000].lower():
        text = html.unescape(re.sub("<[^>]+>", " ", text))
    text = re.sub(r"\s+", " ", text)
    viewed = []
    for term in terms:
        match = re.search(re.escape(term), text, re.I)
        if match:
            viewed.append({"term": term, "context": text[max(0, match.start() - 100):match.start() + 800]})
    excerpt = " ".join(viewed[0]["context"][viewed[0]["context"].lower().find(terms[0].lower()):].split()[:22]) if viewed else ""
    record = {"id": identifier, "url": url, "version": version, "accessed_on": datetime.now(UTC).date().isoformat(),
              "command": cmd, "curl_exit": proc.returncode, "stderr": proc.stderr.decode(errors="replace"),
              "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "excerpt": excerpt,
              "locators_found": [v["term"] for v in viewed]}
    return record, viewed


records = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for record, viewed in pool.map(fetch, SPECS):
        records.append(record)
        print(json.dumps({"receipt": record, "inspected_context": viewed}, ensure_ascii=False), flush=True)
output = ART / "original-sources.json"
output.write_text(json.dumps({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ART.parents[3])),
                            "environment": {"python": sys.version, "transport": "curl anonymous HTTPS GET; full responses held in memory only"},
                            "sources": records}, ensure_ascii=False, indent=2) + "\n")
