"""Read-only primary-source downloads for the R.4 review; never fetch weights."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "originals"


def fetch(item):
    name, url = item
    try:
        with urlopen(url, timeout=30) as response:
            raw = response.read()
            status = response.status
        destination = OUT / name
        destination.write_bytes(raw)
        return {
            "path": str(destination.relative_to(ROOT)),
            "url": url,
            "status": status,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    except Exception as error:
        return {"url": url, "error": str(error)}


def main():
    OUT.mkdir(exist_ok=True)
    capstone = json.loads((ROOT / "docs/course-experiments/capstone-public.json").read_text())
    earlier = json.loads((ROOT / "docs/course-experiments/public-models.json").read_text())
    tasks = [
        ("attention.pdf", "https://arxiv.org/pdf/1706.03762v7"),
        ("moe.pdf", "https://arxiv.org/pdf/1701.06538v1"),
        ("distillation.pdf", "https://arxiv.org/pdf/1503.02531v1"),
        ("pytorch-autograd.html", "https://docs.pytorch.org/docs/2.11/autograd.html"),
    ]
    for model in earlier["models"]:
        directory = Path(model["files"][0]["path"]).parent.as_posix()
        tasks.append(
            (
                f"hf-{model['id']}.json",
                f"https://huggingface.co/api/models/{model['repo']}/tree/{model['revision']}/{directory}?recursive=true&expand=false",
            )
        )
    parents = {Path(model["files"][0]["path"]).parent.parent.as_posix() for model in capstone["models"]}
    for index, directory in enumerate(sorted(parents)):
        tasks.append(
            (
                f"hf-capstone-{index}.json",
                f"https://huggingface.co/api/models/{capstone['repo']}/tree/{capstone['revision']}/{directory}?recursive=true&expand=false",
            )
        )
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(fetch, tasks))
    (OUT / "fetch-receipt.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
