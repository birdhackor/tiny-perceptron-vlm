"""讀取官方原文；僅保存短摘錄、版本及取得收據。"""

import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


class Article(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.active = True

    def handle_endtag(self, tag):
        if tag == "article":
            self.active = False

    def handle_data(self, data):
        if self.active and data.strip():
            self.parts.append(data.strip())


targets = [
    ("compile_api", "https://docs.pytorch.org/docs/2.14/generated/torch.compile.html", "2.14", "cache the compiled result on the code object for future"),
    ("compiler_overview", "https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html", "2.14", "TorchInductor"),
    ("cuda_timing", "https://docs.pytorch.org/docs/2.14/notes/cuda.html", "2.14", "time measurements without synchronizations are not accurate"),
    ("eager_backend", "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/_dynamo/backends/debugging.py", "v2.14.1", "return gm.forward"),
    ("compile_source", "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/__init__.py", "v2.14.1", "return torch._dynamo.optimize("),
]
receipts = []
for identifier, url, version, excerpt in targets:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", url]
    result = subprocess.run(command, check=True, capture_output=True)
    raw = result.stdout
    text = raw.decode()
    if url.endswith(".html"):
        parser = Article()
        parser.feed(text)
        inspected = " ".join(parser.parts)
    else:
        tree = ast.parse(text)
        name = "eager" if identifier == "eager_backend" else "compile"
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        node = max(nodes, key=lambda n: n.end_lineno - n.lineno)
        inspected = "\n".join(text.splitlines()[node.lineno - 1:node.end_lineno])
    inspected = " ".join(inspected.split())
    assert excerpt in inspected, identifier
    receipts.append({
        "id": identifier,
        "url": url,
        "version": version,
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "command_argv": command,
        "exit_code": result.returncode,
        "download_bytes": len(raw),
        "download_sha256": hashlib.sha256(raw).hexdigest(),
        "excerpt": excerpt,
        "inspected_sha256": hashlib.sha256(inspected.encode()).hexdigest(),
    })
    if identifier == "compile_source":
        print(json.dumps({"id": identifier, "inspected_original": inspected}, ensure_ascii=False))
    else:
        print(json.dumps({"id": identifier, "excerpt": excerpt}, ensure_ascii=False))

Path(__file__).with_name("source-receipts.json").write_text(
    json.dumps(receipts, ensure_ascii=False, indent=2) + "\n"
)
