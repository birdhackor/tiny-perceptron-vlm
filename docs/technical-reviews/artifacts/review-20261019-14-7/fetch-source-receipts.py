"""以 curl 讀取原始來源；全文只留在記憶體，輸出版本與內容雜湊。"""

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


sources = [
    ("roformer", "https://arxiv.org/pdf/2104.09864v5", "arXiv:2104.09864v5", "Rotary position embedding；General form；公式 14–16"),
    ("position-interpolation", "https://arxiv.org/pdf/2306.15595v2", "arXiv:2306.15595v2", "Direct Extrapolation；Position Interpolation；公式 4"),
    ("ruler", "https://arxiv.org/pdf/2404.06654v3", "arXiv:2404.06654v3", "Experiments & Results：Effective Context Size；Limitations"),
    ("transformer", "https://arxiv.org/pdf/1706.03762v7", "arXiv:1706.03762v7", "Scaled Dot-Product Attention；Applications of Attention in our Model"),
    ("sentencepiece", "https://arxiv.org/pdf/1808.06226v1", "arXiv:1808.06226v1", "Introduction；subword segmentation"),
    ("embedding", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html", "PyTorch 2.14 文件；本機執行版 2.14.1", "Embedding：num_embeddings；weight shape"),
]
records = []
for identifier, url, version, locator in sources:
    argv = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    completed = subprocess.run(argv, capture_output=True, check=True)
    raw = completed.stdout
    record = {"id": identifier, "url": url, "version": version, "locator": locator, "accessed_at": datetime.now(UTC).isoformat(), "command_argv": argv, "exit_code": completed.returncode, "bytes": len(raw), "original_sha256": hashlib.sha256(raw).hexdigest(), "stderr": completed.stderr.decode()}
    if url.endswith(".html"):
        parser = PlainText()
        parser.feed(raw.decode())
        text = " ".join(" ".join(parser.parts).split())
        excerpt = "size of the dictionary of embeddings"
        assert excerpt in text
        record["excerpt"] = excerpt
        record["shape_signature_present"] = "num_embeddings, embedding_dim" in text
        assert record["shape_signature_present"]
    else:
        assert raw.startswith(b"%PDF-")
        record["pdf_signature_verified"] = True
    records.append(record)
print(json.dumps({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-7/fetch-source-receipts.py", "environment": {"python": platform.python_version(), "curl": subprocess.run(["curl", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0], "device": "cpu；HTTPS 來源擷取"}, "records": records, "result": "已以 curl 讀取指定版本，全文未寫入檔案。論文支持範圍由審閱者讀取原始 PDF 判斷。"}, ensure_ascii=False, indent=2))
