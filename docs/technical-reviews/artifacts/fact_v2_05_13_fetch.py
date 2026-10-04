"""Persist original-source snapshots inspected by the 5.13 reviewer."""

import hashlib
import json
import platform
import shutil
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_05_13_"
records = []
for name, url in {
    "nist-factorial.html": "https://www.itl.nist.gov/div898/handbook/pri/section3/pri3331.htm",
    "torch-linear.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{torch.version.git_version}/torch/nn/modules/linear.py",
    "torch-loss.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{torch.version.git_version}/torch/nn/modules/loss.py",
}.items():
    target = OUT / (PREFIX + name)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            target.write_bytes(response.read())
            records.append({"url": url, "resolved_url": response.url, "path": str(target.relative_to(ROOT))})
    except Exception as error:
        records.append({"url": url, "error": str(error)})
for name, origin in {
    "chinchilla-v1.pdf": "outputs/technical-sources/text-behavior/chinchilla2022.pdf",
    "chinchilla-v1.txt": "outputs/technical-sources/text-behavior/chinchilla2022.txt",
    "chinchilla-metadata.html": "outputs/technical-sources/text-behavior/chinchilla2022-metadata.html",
    "cawley2010.txt": "outputs/technical-sources/5.10-fresh/cawley2010.txt",
    "cawley2010.pdf": "outputs/technical-sources/5.10-fresh/cawley2010.pdf",
    "sklearn-cv.txt": "outputs/technical-sources/5.10-fresh/sklearn-cross-validation.txt",
    "sklearn-cv.html": "outputs/technical-sources/5.10-fresh/sklearn-cross-validation.html",
    "sklearn-pitfalls.txt": "outputs/technical-sources/5.10-fresh/sklearn-common-pitfalls.txt",
    "sklearn-pitfalls.html": "outputs/technical-sources/5.10-fresh/sklearn-common-pitfalls.html",
    "text-foundation-raw.json": "docs/course-experiments/results/text_foundation.json",
}.items():
    target = OUT / (PREFIX + name)
    shutil.copyfile(ROOT / origin, target)
    records.append({"copied_original": origin, "path": str(target.relative_to(ROOT))})
for record in records:
    if "path" in record:
        path = ROOT / record["path"]
        record.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
print(json.dumps({"environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                  "device": "cpu", "torch_git_version": torch.version.git_version},
                  "records": records}, ensure_ascii=False, indent=2))
