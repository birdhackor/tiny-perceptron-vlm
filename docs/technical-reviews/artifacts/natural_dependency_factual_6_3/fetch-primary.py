"""Fetch fixed primary sources for this replacement review, with TLS verification."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
BASE = "https://raw.githubusercontent.com/huggingface/tokenizers/88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3/"
URLS = {
    "byte_level.rs": BASE + "tokenizers/src/pre_tokenizers/byte_level.rs",
    "trainer.rs": BASE + "tokenizers/src/models/bpe/trainer.rs",
    "tokenizer-mod.rs": BASE + "tokenizers/src/tokenizer/mod.rs",
    "tokenizers-api.pyi": BASE + "bindings/python/py_src/tokenizers/__init__.pyi",
    "bpe-model.rs": BASE + "tokenizers/src/models/bpe/model.rs",
    "gpt2-report.pdf": "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
}


def fetch(item):
    name, url = item
    with urlopen(url, timeout=30) as response:
        raw = response.read()
    (HERE / name).write_bytes(raw)
    return {"path": name, "url": url, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


with ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(fetch, URLS.items()))
(HERE / "primary-downloads.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
