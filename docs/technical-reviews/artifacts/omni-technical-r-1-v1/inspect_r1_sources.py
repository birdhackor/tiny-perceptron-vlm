"""Inspect original cached sources and the first-lesson public notebook links."""

from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import platform
import re
import sys
from urllib.parse import urljoin
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[4]
CACHE = Path("/workspace/work/reference/omni-technical-r-1-v1")
OUT = Path(__file__).with_name("source-inspection.json")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attributes):
        if tag == "a":
            self.links.append(dict(attributes).get("href", ""))


def source(name, url, version, locator, excerpts):
    path = CACHE / name
    raw = path.read_bytes()
    text = raw.decode()
    original_path = path.with_suffix(".html") if path.suffix == ".plain" else path
    for excerpt in excerpts:
        assert excerpt in text, (name, excerpt)
    return {
        "url": url,
        "version": version,
        "cached_original": str(original_path),
        "original_sha256": sha256(original_path.read_bytes()).hexdigest(),
        "inspected_text_path": str(path),
        "inspected_text_sha256": sha256(raw).hexdigest(),
        "locator": locator,
        "short_excerpts": excerpts,
        "checked_original": True,
    }


inspected = [
    source(
        "python-datastructures.plain",
        "https://docs.python.org/3.13/tutorial/datastructures.html",
        "Python 3.13 official documentation; original HTML SHA is in fetch-receipt.json",
        "section 5.5 Dictionaries; reviewer-extracted text lines 380-400",
        [
            "with the requirement that the keys are unique (within one dictionary).",
            "extracting the value given the key.",
        ],
    ),
    source(
        "nbformat-description.rst",
        "https://raw.githubusercontent.com/jupyter/nbformat/v5.10.4/docs/format_description.rst",
        "nbformat v5.10.4",
        "lines 62-110: Cell Types, Markdown cells, Code cells",
        [
            "There are a few basic cell types for encapsulating code and text.",
            "Markdown cells are used for body-text",
            "They contain source code in the language of the document's associated kernel,",
        ],
    ),
    source(
        "colab-faq.plain",
        "https://research.google.com/colaboratory/faq.html",
        "Live FAQ retrieved 2026-10-29; original HTML SHA is in fetch-receipt.json",
        "What is Colaboratory?; Where are my notebooks stored?; Where is my code executed?; How can I get my data out?",
        [
            "Colab is a hosted Jupyter Notebook service",
            "Colab notebooks are stored in Google Drive, or can be loaded from GitHub.",
            "Code is executed in a virtual machine private to your account.",
            "All Colab notebooks are stored in the open source Jupyter notebook format ( .ipynb).",
        ],
    ),
]

raw_receipts = json.loads((CACHE / "fetch-receipt.json").read_text())
receipts = [r for r in raw_receipts if r.get("status") == 200]
for receipt in receipts:
    assert sha256(Path(receipt["path"]).read_bytes()).hexdigest() == receipt["sha256"]

colab_html = (CACHE / "colab-overview.html").read_text()
colab_links = Links()
colab_links.feed(colab_html)
sign_in_links = [u for u in colab_links.links if u.startswith("https://accounts.google.com/ServiceLogin")]
assert sign_in_links

public_path = CACHE / "public-1.1.html"
public_plain = (CACHE / "public-1.1.plain").read_text()
assert "實際執行結果 · CPU" in public_plain
assert "[3, 2, 1, 4, 1, 2, 3, 0]" in public_plain
assert "['。', '狗', '看', '貓', '，']" in public_plain
links = Links()
links.feed(public_path.read_text())
download_link = next(u for u in links.links if u == "notebooks/01/1.1.ipynb")
colab_link = next(u for u in links.links if u.startswith("https://colab.research.google.com/github/") and u.endswith("/notebooks/01/1.1.ipynb"))
download_url = urljoin("https://birdhackor.github.io/tiny-perceptron-vlm/1.1.html", download_link)
raw_url = colab_link.replace("https://colab.research.google.com/github/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
notebook_hash = sha256((ROOT / "notebooks/01/1.1.ipynb").read_bytes()).hexdigest()
notebook_downloads = []
for name, url in [("public-1.1.ipynb", download_url), ("colab-pinned-1.1.ipynb", raw_url)]:
    with urlopen(url, timeout=25) as response:
        raw = response.read()
        final_url = response.url
        status = response.status
    path = CACHE / name
    with path.open("xb") as handle:
        handle.write(raw)
    actual_hash = sha256(raw).hexdigest()
    assert actual_hash == notebook_hash, (url, actual_hash, notebook_hash)
    notebook_downloads.append({
        "url": url,
        "final_url": final_url,
        "status": status,
        "sha256": actual_hash,
        "accessed_at": datetime.now(UTC).isoformat(),
        "matches_local_notebook_bytes": True,
    })

steps = (ROOT / "course/first-steps.md").read_text()
headings = {}
for lesson in ["W.1", "W.2", "W.3"]:
    headings[lesson] = re.search(r"^## " + re.escape(lesson) + r" .+$", steps, re.M).group()
assert (ROOT / "course/glossary.md").is_file()
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/inspect_r1_sources.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; HTTPS GET and source inspection only"},
    "original_fetch_receipts": receipts,
    "sources_inspected": inspected,
    "colab_account_evidence": {
        "locator": "Official Colab intro HTML line 97 Sign in link; FAQ Where is my code executed?",
        "google_sign_in_host": "accounts.google.com",
        "faq_scope": "Hosted execution belongs to a signed-in account; this is not a claim that anonymous public notebook viewing requires login.",
        "cloud_execution_performed": False,
    },
    "public_page": {
        "url": "https://birdhackor.github.io/tiny-perceptron-vlm/1.1.html",
        "html_sha256": sha256(public_path.read_bytes()).hexdigest(),
        "text_locator": "reviewer-extracted lines 441-462",
        "cpu_label_and_expected_three_lines_present": True,
        "download_link": download_url,
        "colab_link": colab_link,
        "notebook_downloads": notebook_downloads,
    },
    "navigation": {"headings": headings, "glossary_exists": True},
    "scope": "Original source text and exact first-lesson public download/Colab target inspected. No cloud runtime, whole-book comprehension audit, or rendered figure inspection was performed.",
}
with OUT.open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
