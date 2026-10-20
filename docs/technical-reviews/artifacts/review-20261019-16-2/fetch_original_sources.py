"""讀取原始來源，只保存取得收據與支持主張的短摘錄。"""

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


SOURCES = [
    {
        "id": "cache",
        "url": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/cache_explanation.md",
        "version": "Transformers v4.57.1",
        "locators": [
            ("Attention matrices", "At inference time, you only need the last token's query"),
            ("Cache storage implementation", "Caches are structured as a list of layers"),
        ],
    },
    {
        "id": "attention",
        "url": "https://arxiv.org/html/1706.03762v7",
        "version": "Attention Is All You Need, arXiv:1706.03762v7",
        "locators": [
            ("Scaled Dot-Product Attention", "The output is computed as a weighted sum of the values"),
            ("Multi-Head Attention", "On each of these projected versions of queries"),
        ],
    },
    {
        "id": "autograd",
        "url": "https://docs.pytorch.org/docs/2.14/notes/autograd.html",
        "version": "PyTorch 2.14 documentation; local runtime 2.14.1",
        "locators": [
            ("No-grad Mode", "Computations in no-grad mode behave as if none of the inputs require grad"),
            ("Evaluation Mode", "are completely orthogonal to no-grad mode and inference mode"),
        ],
    },
    {
        "id": "metrics",
        "url": "https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics",
        "version": "NVIDIA NIM Benchmarking LLM 1.0.0",
        "locators": [
            ("Time to First Token", "Time to first token generally includes both request queuing time"),
            ("Inter-token Latency", "Generation_time is the duration from the first token received"),
        ],
    },
    {
        "id": "tokenizer",
        "url": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/tokenizer_summary.md",
        "version": "Transformers v4.57.1",
        "locators": [
            ("Subword tokenization", "Subword tokenization algorithms rely on the principle"),
            ("Byte-level BPE", "uses bytes as the base vocabulary"),
        ],
    },
]


def main():
    receipts = []
    for source in SOURCES:
        command = ["curl", "--fail", "--location", "--silent", "--show-error", source["url"]]
        result = subprocess.run(command, check=True, capture_output=True)
        raw = result.stdout
        text = raw.decode("utf-8")
        if not source["url"].endswith(".md"):
            parser = Text()
            parser.feed(text)
            text = " ".join(parser.parts)
        text = re.sub(r"\s+", " ", text)
        excerpts = []
        for locator, prefix in source["locators"]:
            assert prefix in text, prefix
            start = text.index(prefix)
            excerpt = " ".join(text[start:].split()[:12])
            excerpts.append({"locator": locator, "excerpt": excerpt})
        receipts.append(
            {
                "id": source["id"],
                "url": source["url"],
                "version": source["version"],
                "accessed_at": datetime.now(UTC).isoformat(),
                "command": command,
                "exit_code": result.returncode,
                "original_bytes": len(raw),
                "original_sha256": hashlib.sha256(raw).hexdigest(),
                "excerpts": excerpts,
                "storage": "原始全文只在記憶體讀取；持久檔僅含收據與短摘錄。",
            }
        )
    destination = Path(__file__).with_name("original-source-receipts.json")
    destination.write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
