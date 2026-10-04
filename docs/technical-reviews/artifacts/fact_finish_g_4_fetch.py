"""Fetch original sources for this reviewer, retaining bytes and retrieval metadata."""

import concurrent.futures
import hashlib
import json
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
PREFIX = "fact_finish_g_4"
SOURCES = {
    "switch": "https://arxiv.org/pdf/2101.03961v3",
    "flash": "https://arxiv.org/pdf/2205.14135v2",
    "quant": "https://arxiv.org/pdf/1712.05877v1",
    "distill": "https://arxiv.org/pdf/1503.02531v1",
    "minillm": "https://arxiv.org/pdf/2306.08543v6",
    "ppo": "https://arxiv.org/pdf/1707.06347v2",
    "instructgpt": "https://arxiv.org/pdf/2203.02155v1",
    "dpo": "https://arxiv.org/pdf/2305.18290v3",
    "rag": "https://arxiv.org/pdf/2005.11401v4",
    "cache": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/cache_explanation.md",
    "function_calling": "https://developers.openai.com/api/docs/guides/function-calling",
    "model_spec": "https://model-spec.openai.com/2025-12-18.html",
}


def fetch(item):
    name, url = item
    request = urllib.request.Request(url, headers={"User-Agent": "G4-technical-review/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
        content_type = response.headers.get("Content-Type", "")
        final_url = response.url
    suffix = ".pdf" if data.startswith(b"%PDF") else (".md" if name == "cache" else ".html")
    path = BASE / f"{PREFIX}_{name}_original{suffix}"
    path.write_bytes(data)
    result = {
        "name": name,
        "requested_url": url,
        "final_url": final_url,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "content_type": content_type,
        "path": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if suffix == ".pdf":
        output = path.with_suffix(".txt")
        subprocess.run(["pdftotext", "-layout", str(path), str(output)], check=True)
        result["text_path"] = output.name
        result["text_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    return result


def main():
    receipts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name, future in [(name, pool.submit(fetch, (name, url))) for name, url in SOURCES.items()]:
            try:
                receipt = future.result()
            except Exception as error:
                receipt = {"name": name, "requested_url": SOURCES[name], "error": str(error)}
            receipts.append(receipt)
            print(json.dumps(receipt, ensure_ascii=False))
    (BASE / f"{PREFIX}_fetch_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()
