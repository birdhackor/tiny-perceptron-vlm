"""以 curl 讀取原始來源；只保存短摘錄、版本與取得收據。"""

from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


requests = [
    {
        "id": "paper",
        "url": "https://arxiv.org/html/2307.03172v3",
        "version": "arXiv:2307.03172v3，20 Nov 2023",
        "locators": ["Multi-Document Question Answering / Experimental Setup", "Models", "Results and Discussion", "How Well Can Language Models Retrieve From Input Contexts?", "Effect of Query-Aware Contextualization"],
        "excerpts": ["we see a distinctive U-shaped performance curve", "Claude-1.3 and Claude-1.3 (100K) do nearly perfectly"],
    },
    {
        "id": "python_lists",
        "url": "https://docs.python.org/3.13/tutorial/datastructures.html",
        "version": "Python 3.13.16 documentation",
        "locators": ["More on Lists / list.index", "List Comprehensions"],
        "excerpts": ["Return zero-based index of the first occurrence of value", "then zero or more for or if clauses"],
    },
    {
        "id": "python_types",
        "url": "https://docs.python.org/3.13/library/stdtypes.html",
        "version": "Python 3.13.16 documentation",
        "locators": ["Common Sequence Operations / slice", "Text Sequence Type / str.join"],
        "excerpts": ["The slice of s from i to j", "Return a string which is the concatenation of the strings in iterable"],
    },
    {
        "id": "bpe_source",
        "url": "https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/tokenizers/src/models/bpe/model.rs",
        "version": "huggingface/tokenizers v0.23.2",
        "locators": ["BPE::merge_word", "BPE::tokenize_with_cache", "Model for BPE::tokenize"],
        "excerpts": ["word.merge_all(&self.merges, self.dropout);", "let word = self.merge_word(sequence)?;"],
    },
]
for request in requests:
    argv = ["curl", "-L", "--fail", "--silent", "--show-error", request["url"]]
    result = subprocess.run(argv, check=True, capture_output=True)
    text = result.stdout.decode("utf-8")
    if "/html/" in request["url"] or "docs.python.org" in request["url"]:
        parser = TextParser()
        parser.feed(text)
        text = " ".join(" ".join(parser.parts).split())
    for excerpt in request["excerpts"]:
        assert excerpt in text, request["id"] + ": 找不到指定原文摘錄"
    request.update({
        "command_argv": argv,
        "exit_code": result.returncode,
        "stderr": result.stderr.decode(),
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrieved_bytes": len(result.stdout),
        "retrieved_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "checked_original": True,
        "fulltext_saved": False,
    })

receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-A-8/inspect_sources.py",
    "environment": {
        "python": sys.version,
        "curl": subprocess.run(["curl", "--version"], capture_output=True, check=True, text=True).stdout.splitlines()[0],
        "device": "cpu；僅取得與核對來源，未執行論文模型",
    },
    "result": "四個原始來源均成功取得，短摘錄均與本次原文相符。",
    "sources": requests,
}
target = Path(__file__).with_name("source-inspection.json")
with target.open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
