"""Fetch primary official documents; record failures without inventing evidence."""
import hashlib
import json
from pathlib import Path
import urllib.request
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "nist-finite-difference": "https://dlmf.nist.gov/3.4",
    "openstax-derivative": "https://openstax.org/books/calculus-volume-1/pages/3-1-defining-the-derivative",
    "openstax-linear-approximation": "https://openstax.org/books/calculus-volume-1/pages/4-2-linear-approximations-and-differentials",
    "pytorch-autograd": "https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html",
    "python-floatingpoint": "https://docs.python.org/3.13/tutorial/floatingpoint.html",
}

class Reader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag in ("p", "div", "section", "li", "h1", "h2", "h3", "h4", "pre", "br"):
            self.parts.append("\n")
        if tag == "img":
            self.parts.append(dict(attrs).get("alt", ""))
    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1
        if tag in ("p", "div", "section", "li", "h1", "h2", "h3", "h4", "pre"):
            self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

receipt = []
for name, url in SOURCES.items():
    entry = {"name": name, "url": url, "accessed_on": "2026-10-05"}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            entry.update(status=response.status, final_url=response.url)
        (ROOT / f"{name}.html").write_bytes(raw)
        reader = Reader()
        reader.feed(raw.decode("utf-8"))
        readable = "\n".join(line.strip() for line in "".join(reader.parts).splitlines() if line.strip())
        (ROOT / f"{name}.txt").write_text(readable + "\n", encoding="utf-8")
        entry.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    except Exception as error:
        entry.update(error=repr(error))
    receipt.append(entry)
(ROOT / "fetch-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
