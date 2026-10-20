"""讀取官方來源，只保存定位所需的短摘錄與下載收據。"""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden -= 1

    def handle_data(self, text):
        if not self.hidden:
            self.parts.append(text)


SOURCES = [
    ("python_sequences", "https://docs.python.org/3.13/library/stdtypes.html",
     ["If i or j is greater", "range(stop)", "The slice of s from i to j"]),
    ("torch_tolist", "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html",
     ["Returns the tensor as a", "PyTorch 2."]),
    ("causal_lm", "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/tasks/language_modeling.md",
     ["Causal language modeling predicts", "block_size =", "Drop the small remainder", "shifted to the right"]),
    ("grouped_split", "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst",
     ["The i.i.d. assumption is broken", "all the samples in the validation fold", "same group is not represented"]),
    ("bpe_merges", "https://raw.githubusercontent.com/huggingface/course/main/chapters/en/chapter6/5.mdx",
     ["GPT-2", "merg", "bytes"]),
]


def main():
    receipts = []
    for name, url, needles in SOURCES:
        command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "40", url]
        result = subprocess.run(command, check=True, capture_output=True)
        raw = result.stdout
        decoded = raw.decode("utf-8")
        if "<html" in decoded[:2000].lower() or "<!doctype html" in decoded[:2000].lower():
            parser = PlainText()
            parser.feed(decoded)
            text = " ".join(parser.parts)
            titles = re.findall(r"<title>(.*?)</title>", decoded, flags=re.S)
            title = re.sub(r"\s+", " ", titles[0]).strip() if titles else name
        else:
            text, title = decoded, name
        text = re.sub(r"\s+", " ", text)
        excerpts = []
        for needle in needles:
            match = re.search(re.escape(needle), text, flags=re.I)
            if match is None:
                excerpts.append({"locator": needle, "found": False})
            else:
                excerpts.append({"locator": needle, "found": True,
                                 "excerpt": text[max(0, match.start()-90):match.end()+1050]})
        receipt = {"id": name, "url": url, "title": title,
                   "accessed_at": datetime.now(timezone.utc).isoformat(),
                   "command_argv": command, "exit_code": result.returncode,
                   "retrieved_bytes": len(raw), "retrieved_sha256": hashlib.sha256(raw).hexdigest(),
                   "excerpts": excerpts}
        receipts.append(receipt)
        print(json.dumps(receipt, ensure_ascii=False))
    target = Path(__file__).with_name("external-sources.json")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(receipts, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
