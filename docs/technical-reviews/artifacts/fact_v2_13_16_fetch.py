"""Fetch exact primary versions; persist text and download provenance."""

import hashlib
import html
import json
import platform
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "docs/technical-reviews/artifacts"


def main():
    records = []
    for name, identifier in [
        ("r1", "2501.12948v1"),
        ("dpo", "2305.18290v3"),
        ("ppo", "1707.06347v2"),
        ("instructgpt", "2203.02155v1"),
    ]:
        url = f"https://arxiv.org/pdf/{identifier}"
        with urlopen(url, timeout=40) as response:
            data = response.read()
            final_url = response.url
        with tempfile.TemporaryDirectory() as temporary:
            pdf = Path(temporary) / f"{name}.pdf"
            pdf.write_bytes(data)
            output = ARTIFACTS / f"fact_v2_13_16_{name}_original.txt"
            subprocess.run(["pdftotext", "-layout", str(pdf), str(output)], check=True)
        records.append({
            "url": url,
            "final_url": final_url,
            "pdf_sha256": hashlib.sha256(data).hexdigest(),
            "pdf_bytes": len(data),
            "text_path": str(output.relative_to(ROOT)),
            "text_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        })
    url = "https://docs.python.org/3.13/library/functions.html"
    with urlopen(url, timeout=40) as response:
        data = response.read()
    original = data.decode()
    excerpts = []
    for identifier in ["len", "max"]:
        start = original.index(f'id="{identifier}"')
        end = original.index('<dl class="py function">', start)
        excerpt = html.unescape(re.sub(r"<[^>]+>", "", original[start:end]))
        excerpts.append(f"URL: {url}#{identifier}\n{excerpt.strip()}")
    output = ARTIFACTS / "fact_v2_13_16_python_docs.txt"
    output.write_text("\n\n".join(excerpts) + "\n", encoding="utf-8")
    records.append({"url": url, "html_sha256": hashlib.sha256(data).hexdigest(),
                    "text_path": str(output.relative_to(ROOT)), "text_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    result = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_16_fetch.py",
        "environment": {"python": platform.python_version(), "pdf_extractor": "pdftotext -layout (system)"},
        "result": "Downloaded HTTPS version-specific originals and extracted complete texts successfully",
        "records": records,
    }
    (ARTIFACTS / "fact_v2_13_16_fetch.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
