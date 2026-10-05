"""Retrieve original authority sources only; no datasets, model files, or reviews."""
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
import json

OUT = Path(__file__).resolve().parent
SOURCES = {
    "slp3-ngram.pdf": "https://web.stanford.edu/~jurafsky/slp3/3.pdf",
    "python-313-collections.html": "https://docs.python.org/3.13/library/collections.html",
    "python-313-floatingpoint.html": "https://docs.python.org/3.13/tutorial/floatingpoint.html",
    "python-313-random.html": "https://docs.python.org/3.13/library/random.html",
    "python-313-functions.html": "https://docs.python.org/3.13/library/functions.html",
}

class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        if tag in {"p", "div", "li", "dt", "dd", "h1", "h2", "h3", "pre", "tr"}:
            self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip -= 1
        if tag in {"p", "div", "li", "dt", "dd", "h1", "h2", "h3", "pre", "tr"}:
            self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

records = []
for name, url in SOURCES.items():
    request = Request(url, headers={"User-Agent": "Technical factual review; original source inspection"})
    with urlopen(request, timeout=30) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Source exceeded the bounded document size")
        record = {"url": url, "resolved_url": response.url, "content_type": response.headers.get("Content-Type"),
                  "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag"),
                  "accessed_at": datetime.now(timezone.utc).isoformat(), "file": name,
                  "sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
    (OUT / name).write_bytes(raw)
    if name.endswith(".html"):
        parser = PlainText()
        parser.feed(raw.decode("utf-8"))
        (OUT / name.replace(".html", ".txt")).write_text("".join(parser.parts), encoding="utf-8")
    records.append(record)
    print(json.dumps(record, ensure_ascii=False))
(OUT / "source-acquisition.json").write_text(json.dumps(records, indent=2) + "\n")
