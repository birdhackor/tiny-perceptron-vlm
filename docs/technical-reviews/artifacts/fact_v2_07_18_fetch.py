"""Fetch the exact original-paper versions cited in section 7.18."""

import hashlib
import json
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PAPERS = {
    "instructgpt": "2203.02155v1",
    "deepseek_r1": "2501.12948v1",
    "dpo": "2305.18290v1",
    "ppo": "1707.06347v1",
}


def fetch(item):
    name, version = item
    url = f"https://arxiv.org/pdf/{version}"
    path = OUT / f"fact_v2_07_18_{name}.pdf"
    request = Request(url, headers={"User-Agent": "technical-review/1.0"})
    with urlopen(request, timeout=60) as response:
        data = response.read()
        final_url = response.url
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"Not a PDF: {url}")
    path.write_bytes(data)
    text_path = path.with_suffix(".txt")
    command = ["pdftotext", "-layout", str(path), str(text_path)]
    subprocess.run(command, check=True)
    return {
        "name": name,
        "version": version,
        "url": url,
        "resolved_url": final_url,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "text_path": str(text_path.relative_to(ROOT)),
        "text_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
        "conversion_command": " ".join(command),
        "conversion_exit_code": 0,
    }


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        originals = list(pool.map(fetch, PAPERS.items()))
    poppler = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True, check=True)
    poppler_version = poppler.stderr.splitlines()[0]
    result = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_07_18_fetch.py",
        "environment": {"python": platform.python_version(), "device": "CPU", "pdftotext": poppler_version},
        "accessed_at": datetime.now(UTC).isoformat(),
        "result": "Four HTTPS version-pinned original PDFs fetched and converted successfully.",
        "originals": originals,
    }
    (OUT / "fact_v2_07_18_fetch.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
