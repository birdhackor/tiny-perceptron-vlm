"""Fetch versioned official Python documentation; preserve original HTTPS bytes."""
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import json
import sys
from datetime import datetime, UTC
from lxml import html

OUT = Path(__file__).resolve().parent
PAGES = {
    "functions": ("https://docs.python.org/3.13/library/functions.html", ["enumerate", "sorted", "ord"]),
    "stdtypes": ("https://docs.python.org/3.13/library/stdtypes.html", ["text-sequence-type-str", "set-types-set-frozenset", "mapping-types-dict", "str.join", "common-sequence-operations"]),
    "expressions": ("https://docs.python.org/3.13/reference/expressions.html", ["comparisons", "displays-for-lists-sets-and-dictionaries"]),
    "simple-stmts": ("https://docs.python.org/3.13/reference/simple_stmts.html", ["the-assert-statement"]),
    "exceptions": ("https://docs.python.org/3.13/library/exceptions.html", ["KeyError"]),
}
records = []
for name, (url, anchors) in PAGES.items():
    with urlopen(Request(url, headers={"User-Agent": "Codex factual reviewer; Python tutorial verification"}), timeout=30) as response:
        raw = response.read()
        final_url = response.url
    (OUT / f"python-3.13-{name}.html").write_bytes(raw)
    root = html.fromstring(raw)
    title = root.xpath("string(//title)").strip()
    extracted = []
    for anchor in anchors:
        elements = root.xpath("//*[@id=$anchor]", anchor=anchor)
        if len(elements) != 1:
            raise ValueError(f"Expected one {anchor}; got {len(elements)}")
        element = elements[0]
        # An API dt is paired with its immediate dd in the same dl.
        if element.tag == "dt":
            content = element.getparent()
        else:
            content = element
        text = "\n".join(line.strip() for line in content.text_content().splitlines() if line.strip())
        extracted.append(f"URL: {url}#{anchor}\nHTML anchor: {anchor}\n{text}\n")
    (OUT / f"python-3.13-{name}-excerpts.txt").write_text("\n".join(extracted), encoding="utf-8")
    records.append({"url":url,"final_url":final_url,"title":title,"version":"Python 3.13 documentation (versioned /3.13/)","accessed_on":datetime.now(UTC).date().isoformat(),"anchors":anchors,"raw_sha256":hashlib.sha256(raw).hexdigest()})
    print(json.dumps(records[-1],ensure_ascii=False))
(OUT / "source-fetch.json").write_text(json.dumps({"command":[sys.executable,str(Path(__file__).resolve())],"sources":records},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
