"""Fetch original version-pinned official documentation, preserving TLS checks."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = Path(__file__).resolve().parent
DEST = BASE / "official"
DEST.mkdir(exist_ok=True)
items = [
    ("python-introduction.rst", "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/tutorial/introduction.rst", "CPython v3.13.5"),
    ("python-datastructures.rst", "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/tutorial/datastructures.rst", "CPython v3.13.5"),
    ("python-controlflow.rst", "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/tutorial/controlflow.rst", "CPython v3.13.5"),
    ("python-simple-stmts.rst", "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/reference/simple_stmts.rst", "CPython v3.13.5"),
    ("python-functions.rst", "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/functions.rst", "CPython v3.13.5"),
    ("jupyterlab-notebook.rst", "https://raw.githubusercontent.com/jupyterlab/jupyterlab/v4.4.0/docs/source/user/notebook.rst", "JupyterLab v4.4.0"),
    ("jupyterlab-tracker.json", "https://raw.githubusercontent.com/jupyterlab/jupyterlab/v4.4.0/packages/notebook-extension/schema/tracker.json", "JupyterLab v4.4.0"),
    ("python-expressions.rst", "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/reference/expressions.rst", "CPython v3.13.5"),
]

def fetch(item):
    name, url, version = item
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
        status, final_url = response.status, response.url
    (DEST / name).write_bytes(raw)
    return {"file": name, "url": url, "final_url": final_url, "version": version,
            "accessed_on": "2026-10-05", "status": status, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "tls_verification": "urllib default verified HTTPS"}

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    records = list(pool.map(fetch, items))
(BASE / "official-fetch.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
for record in records:
    print(json.dumps(record, ensure_ascii=False))
