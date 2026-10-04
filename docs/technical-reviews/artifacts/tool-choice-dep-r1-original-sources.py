"""Fetch the two original documents cited by the independent R.1 review."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

from bs4 import BeautifulSoup

DEST = Path(__file__).parent
SOURCES = [
    ("nbformat", "https://raw.githubusercontent.com/jupyter/nbformat/v5.10.4/docs/format_description.rst", ["Top-level structure", "Markdown cells", "Code cells"]),
    ("colab", "https://research.google.com/colaboratory/faq.html", ["What is Colab", "What is the difference between Jupyter and Colab"]),
]
for name, url, headings in SOURCES:
    response = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "30", url], check=True, capture_output=True)
    raw = response.stdout
    metadata = {"requested_url": url, "http_status": 200, "transport": "curl --fail --location with TLS verification"}
    soup = BeautifulSoup(raw, "html.parser")
    text = raw.decode("utf-8") if name == "nbformat" else soup.get_text(" ", strip=True)
    excerpts = []
    for heading in headings:
        found = None if name == "nbformat" else next((h for h in soup.find_all(["h1", "h2", "h3", "dt"]) if heading.lower() in h.get_text(" ", strip=True).lower()), None)
        if found is not None:
            container = found.parent if name == "nbformat" else found
            if name == "colab":
                parts = [found.get_text(" ", strip=True)]
                for sibling in found.next_siblings:
                    if getattr(sibling, "name", None) in {"h1", "h2", "h3", "dt"}:
                        break
                    if hasattr(sibling, "get_text"):
                        parts.append(sibling.get_text(" ", strip=True))
                excerpt = "\n".join(parts)
            else:
                excerpt = container.get_text(" ", strip=True)
            excerpts.append({"heading": heading, "excerpt": excerpt[:10000]})
        else:
            at = text.lower().find(heading.lower())
            excerpts.append({"heading": heading, "excerpt": text[max(0, at):max(0, at)+2600], "fallback_offset": at})
    record = {**metadata, "read_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "document_sha256": hashlib.sha256(raw).hexdigest(), "title": "The Notebook file format (nbformat v5.10.4)" if name == "nbformat" else soup.title.get_text(" ", strip=True), "excerpts": excerpts}
    path = DEST / f"tool-choice-dep-r1-{name}-original.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"snapshot": path.name, **record}, ensure_ascii=False))
