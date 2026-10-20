"""以 curl 讀取官方原文，只保存短摘錄與取得收據。"""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


ITEMS = [
    ("softmax", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html",
     "Softmax is defined as", "sum to 1", "torch.nn.functional.softmax"),
    ("cross_entropy", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html",
     "The unreduced", "unnormalized logits", "CrossEntropyLoss; class indices and unreduced loss"),
    ("argmax", "https://docs.pytorch.org/docs/2.14/generated/torch.argmax.html",
     "If there are multiple maximal values", "the indices of the first maximal value are returned", "torch.argmax; Note"),
    ("dict", "https://docs.python.org/3.13/library/stdtypes.html#mapping-types-dict",
     "Dictionaries preserve insertion order", "Dictionaries preserve insertion order", "Mapping Types — dict; insertion order"),
    ("builtins", "https://docs.python.org/3.13/library/functions.html",
     "Return a reverse iterator", "Return a reverse iterator", "reversed; round"),
    ("tolist", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.tolist.html",
     "Returns the tensor as a", "Returns the tensor as a", "torch.Tensor.tolist"),
]


def fetch(item):
    name, url, anchor, quote, locator = item
    command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
    raw = subprocess.check_output(command)
    parser = Text()
    parser.feed(raw.decode())
    text = " ".join(" ".join(parser.parts).split())
    index = text.find(anchor)
    assert index >= 0, (name, anchor)
    assert quote in text, (name, quote)
    inspection = text[max(0, index - 100):index + 2200]
    extra = ""
    if name == "builtins":
        pos = text.find("Return number rounded")
        assert pos >= 0
        extra = text[pos:pos + 1200]
    return {
        "id": name,
        "url": url,
        "command_argv": command,
        "accessed_on": datetime.now(timezone.utc).date().isoformat(),
        "version": "PyTorch 2.14" if "pytorch" in url else "Python 3.13",
        "retrieved_html_sha256": hashlib.sha256(raw).hexdigest(),
        "locator": locator,
        "short_excerpt": quote,
        "inspection": inspection,
        "extra_inspection": extra,
    }


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=6) as executor:
        inspected = list(executor.map(fetch, ITEMS))
    for record in inspected:
        print(record["id"], record["url"])
        print(record["inspection"])
        if record["extra_inspection"]:
            print(record["extra_inspection"])
    saved = [{k: v for k, v in record.items() if k not in {"inspection", "extra_inspection"}} for record in inspected]
    destination = Path(__file__).with_name("official-source-receipts.json")
    with destination.open("x") as handle:
        json.dump(saved, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
