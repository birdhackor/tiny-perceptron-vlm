"""Acquire first-party API documentation and preserve installed first-party source."""
import hashlib
import importlib
import json
import sys
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DEST = HERE / "sources"
DEST.mkdir(exist_ok=True)

class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag in ("p", "h1", "h2", "h3", "li", "pre", "div"):
            self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

urls = {
    "pytorch-randomness": "https://docs.pytorch.org/docs/2.14/notes/randomness.html",
    "pytorch-saving-loading": "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html",
    "pytorch-serialization": "https://docs.pytorch.org/docs/2.14/notes/serialization.html",
    "python-tempfile": "https://docs.python.org/3.13/library/tempfile.html",
    "python-pathlib": "https://docs.python.org/3.13/library/pathlib.html",
}
receipts = []
for name, url in urls.items():
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "lesson-5.7-independent-factual-review"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            resolved = response.url
        (DEST / f"{name}.html").write_bytes(raw)
        parser = Text()
        parser.feed(raw.decode("utf-8"))
        (DEST / f"{name}.txt").write_text("".join(parser.parts))
        receipts.append({"name": name, "url": url, "resolved_url": resolved, "accessed_on": date.today().isoformat(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "success": True})
    except Exception as error:
        receipts.append({"name": name, "url": url, "success": False, "error": str(error)})
for module_name in ("torch.optim.adam", "torch.optim.adamw", "torch.optim.optimizer", "torch.serialization", "torch.random"):
    module = importlib.import_module(module_name)
    path = Path(module.__file__)
    raw = path.read_bytes()
    target = DEST / (module_name.replace(".", "-") + ".py")
    target.write_bytes(raw)
    receipts.append({"name": module_name, "origin": str(path), "url": f"https://github.com/pytorch/pytorch/blob/{torch.version.git_version}/{str(path.relative_to(Path(torch.__file__).parent.parent))}", "version": str(torch.__version__), "git_version": torch.version.git_version, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "success": True})
(HERE / "source-acquisition.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipts, ensure_ascii=False, indent=2))
