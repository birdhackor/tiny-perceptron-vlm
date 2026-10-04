"""Persist the original researchers' official explanation of specification gaming."""

import hashlib
import json
import platform
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen


class PageText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skipped = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skipped += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skipped -= 1

    def handle_data(self, data):
        if not self.skipped and data.strip():
            self.text.append(data.strip())


def main():
    url = "https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/"
    with urlopen(url, timeout=30) as response:
        data = response.read()
        final_url = response.url
    page = PageText()
    page.feed(data.decode())
    text = "\n".join(page.text)
    start = text.index("April 21, 2020")
    end = text.index("\nNotes", start)
    directory = Path(__file__).resolve().parent
    output = directory / "fact_v2_13_16_deepmind_original.txt"
    output.write_text(f"Original URL: {url}\n" + text[start:end] + "\n", encoding="utf-8")
    provenance = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_16_deepmind_fetch.py",
        "environment": {"python": platform.python_version()},
        "result": "Fetched original official article; persisted dated article text from title through conclusion",
        "url": url, "final_url": final_url, "html_bytes": len(data),
        "html_sha256": hashlib.sha256(data).hexdigest(),
        "text_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (directory / "fact_v2_13_16_deepmind_fetch.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == "__main__":
    main()
