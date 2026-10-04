"""Persist independently fetched primary documentation and site responses."""

import concurrent.futures
import hashlib
import json
import platform
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

OUT = Path(__file__).resolve().parent
URLS = {
    "uv-install": "https://docs.astral.sh/uv/getting-started/installation/",
    "uv-sync": "https://docs.astral.sh/uv/concepts/projects/sync/",
    "uv-python": "https://docs.astral.sh/uv/concepts/python-versions/",
    "uv-pytorch": "https://docs.astral.sh/uv/guides/integration/pytorch/",
    "git-install": "https://git-scm.com/install/",
    "git-clone": "https://git-scm.com/docs/git-clone/2.52.0",
    "python-venv": "https://docs.python.org/3.13/library/venv.html",
    "ipykernel-install": "https://ipython.readthedocs.io/en/stable/install/kernel_install.html",
    "jupyterlab-notebook": "https://jupyterlab.readthedocs.io/en/stable/user/notebook.html",
    "jupyterlab-start": "https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html",
    "colab-faq": "https://research.google.com/colaboratory/faq.html",
    "site-home": "https://birdhackor.github.io/tiny-perceptron-vlm/",
    "site-W.1": "https://birdhackor.github.io/tiny-perceptron-vlm/W.1.html",
    "site-1.1": "https://birdhackor.github.io/tiny-perceptron-vlm/1.1.html",
    "site-first-steps": "https://birdhackor.github.io/tiny-perceptron-vlm/first-steps.html",
    "jupyterlab-commands": "https://raw.githubusercontent.com/jupyterlab/jupyterlab/v4.6.4/packages/notebook-extension/src/index.ts",
    "jupyterlab-schema": "https://raw.githubusercontent.com/jupyterlab/jupyterlab/v4.6.4/packages/notebook-extension/schema/tracker.json",
}


class Readable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.text = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href")
        if tag in {"p", "li", "h1", "h2", "h3", "pre", "tr", "section"}:
            self.text.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1
        if tag in {"p", "li", "h1", "h2", "h3", "pre", "tr", "section"}:
            self.text.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


def fetch(item):
    name, url = item
    record = {"name": name, "url": url, "accessed_at": datetime.now(UTC).isoformat()}
    try:
        with urlopen(Request(url, headers={"User-Agent": "W.1-primary-source-review/1.0"}), timeout=25) as response:
            raw = response.read()
            record.update(status=response.status, final_url=response.url, headers=dict(response.headers))
        (OUT / f"{name}.html").write_bytes(raw)
        parser = Readable()
        parser.feed(raw.decode("utf-8"))
        plain = "\n".join(line.strip() for line in "".join(parser.text).splitlines() if line.strip()) + "\n"
        (OUT / f"{name}.txt").write_text(plain, encoding="utf-8")
        record.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), links=parser.links)
    except Exception as error:
        record["error"] = repr(error)
    return record


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(fetch, URLS.items()))
    receipt = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_w_1/fetch_sources.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "tls": "verified"},
        "results": results,
    }
    (OUT / "fetch-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    for result in results:
        print(json.dumps({key: value for key, value in result.items() if key not in {"links", "headers"}}))
    one = next(result for result in results if result["name"] == "site-1.1")
    print(json.dumps({"sample_notebook_links": [link for link in one.get("links", []) if "ipynb" in link]}))
    additional = []
    for link in one.get("links", []):
        if "colab.research" in link:
            additional.append(fetch(("colab-entry", link)))
        elif link == "notebooks/01/1.1.ipynb":
            additional.append(fetch(("site-download-1.1", "https://birdhackor.github.io/tiny-perceptron-vlm/" + link)))
    receipt["results"].extend(additional)
    (OUT / "fetch-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    for result in additional:
        print(json.dumps({key: value for key, value in result.items() if key not in {"links", "headers"}}))


if __name__ == "__main__":
    main()
