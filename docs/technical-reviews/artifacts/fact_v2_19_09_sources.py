"""Fetch the exact original versions inspected in the fresh 19.9 review."""

import hashlib
import json
import subprocess
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
TMP = Path("/tmp/fact_v2_19_09")
TMP.mkdir(exist_ok=True)
COMMIT = "1df335318bda03fd771807f66976953231d5a00b"
RECEIPTS = []


class PlainText(HTMLParser):
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
        if tag in {"p", "h1", "h2", "h3", "li", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    RECEIPTS.append({"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return data


def main():
    pdf = fetch("https://arxiv.org/pdf/2305.13245v3")
    (TMP / "gqa-v3.pdf").write_bytes(pdf)
    subprocess.run(["pdftotext", "-layout", str(TMP / "gqa-v3.pdf"), str(TMP / "gqa-v3.txt")], check=True)
    paper = (TMP / "gqa-v3.txt").read_text()
    start = paper.index("2.1")
    end = paper.index("\f", paper.index("\f") + 1)
    (OUT / "fact_v2_19_09_gqa-original-excerpt.txt").write_text(
        "Original PDF: https://arxiv.org/pdf/2305.13245v3\n"
        + "Accessed 2026-10-04; pdftotext -layout; sections 2.1-2.2 and adjacent page 2 text\n\n"
        + paper[start:end]
    )
    url = "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/functional.py"
    source = fetch(url).decode()
    start = source.index("scaled_dot_product_attention =")
    end = source.index("def _mha_shape_check", start)
    line = source[:start].count("\n") + 1
    (OUT / "fact_v2_19_09_torch-original-excerpt.txt").write_text(
        f"Original source: {url}\nAccessed 2026-10-04; v2.14.1; starting line {line}\n\n" + source[start:end]
    )
    url = "https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation"
    page = fetch(url).decode()
    parser = PlainText()
    parser.feed(page)
    plain = "".join(parser.parts)
    (TMP / "cache-explanation.txt").write_text(plain)
    start = plain.index("Imagine you’re having a conversation")
    end = plain.index("Cache storage implementation", start)
    (OUT / "fact_v2_19_09_cache-original-excerpt.txt").write_text(
        f"Original official docs: {url}\nAccessed 2026-10-04; Transformers v4.57.1\n\n" + plain[start:end]
    )
    paths = {
        "cache-consistency": "docs/course-experiments/capstone-evidence/deployment/cache-consistency.json",
        "generation-benchmark": "docs/course-experiments/capstone-evidence/deployment/generation-benchmark.json",
        "deployment-result": "docs/course-experiments/results/capstone_deployment.json",
    }
    for name, path in paths.items():
        data = fetch(f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{COMMIT}/{path}")
        assert data == (ROOT / path).read_bytes(), path
        (OUT / f"fact_v2_19_09_{name}.json").write_bytes(data)
        RECEIPTS[-1]["equal_current_repository_bytes"] = True
    (OUT / "fact_v2_19_09_source-receipts.json").write_text(
        json.dumps({"accessed_on": "2026-10-04", "receipts": RECEIPTS}, indent=2) + "\n"
    )
    print(json.dumps(RECEIPTS, indent=2))


if __name__ == "__main__":
    main()
