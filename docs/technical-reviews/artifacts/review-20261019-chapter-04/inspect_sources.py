"""直接讀取原始論文，只保存支撐本頁主張的短摘錄與取得收據。"""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current = None
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.current = []

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "p" and self.current is not None:
            self.paragraphs.append(" ".join(" ".join(self.current).split()))
            self.current = None


specifications = [
    {
        "url": "https://arxiv.org/html/1706.03762v7",
        "version": "arXiv:1706.03762v7，2023-08-02",
        "needles": [
            "stacked self-attention",
            "prevent positions",
            "compatibility function",
            "In addition to attention sub-layers",
            "different parameters from layer to layer",
            "learned linear transformation",
            "positional encodings",
        ],
        "excerpt": "While the linear transformations are the same across different positions, they use different parameters from layer to layer.",
        "locator": "Model Architecture；Attention；Position-wise Feed-Forward Networks；Embeddings and Softmax；Positional Encoding",
    },
    {
        "url": "https://arxiv.org/html/1807.03819v3",
        "version": "arXiv:1807.03819v3，2019-03-05",
        "needles": [
            "In each recurrent time-step",
            "Encoder: Given",
            "When running for a fixed number of steps",
        ],
        "excerpt": "both the self-attention and transition weights are tied across layers.",
        "locator": "Model Description；The Universal Transformer；Discussion",
    },
]

records = []
for specification in specifications:
    command = ["curl", "-fsSL", "--max-time", "30", specification["url"]]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode())
    parser = Paragraphs()
    parser.feed(completed.stdout.decode())
    relevant = [
        text
        for text in parser.paragraphs
        if any(needle.lower() in text.lower() for needle in specification["needles"])
    ]
    normalized = " ".join(parser.paragraphs)
    if specification["excerpt"] not in normalized:
        raise RuntimeError("原文中找不到待保存的短摘錄")
    if specification["url"].rsplit("/", 1)[-1] not in completed.stdout.decode():
        raise RuntimeError("回應中找不到指定版本的識別字")
    records.append(
        {
            "url": specification["url"],
            "version": specification["version"],
            "accessed_on": datetime.now(timezone.utc).astimezone().date().isoformat(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "exit_code": completed.returncode,
            "stderr": completed.stderr.decode(),
            "response_bytes": len(completed.stdout),
            "response_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "locator": specification["locator"],
            "excerpt": specification["excerpt"],
            "excerpt_present": specification["excerpt"] in normalized,
            "version_marker_present": True,
            "relevant_paragraphs_found": len(relevant),
            "full_text_saved": False,
        }
    )

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync --offline python docs/technical-reviews/artifacts/review-20261019-chapter-04/inspect_sources.py",
    "environment": {"python": platform.python_version(), "platform": platform.platform()},
    "sources": records,
    "result": "兩份原始論文皆成功以 curl 讀取，版本標記與短摘錄已核對。",
}, ensure_ascii=False, indent=2))
