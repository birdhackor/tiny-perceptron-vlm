"""以 curl 讀官方原文，保存定位、全文指紋及短摘錄，不保存第三方全文。"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, value):
        if not self.skip:
            self.parts.append(value)


sources = [
    ("serialization", "https://docs.pytorch.org/docs/2.14/notes/serialization.html", "PyTorch 2.14", ["not allowed to dynamically import", "uncompressed ZIP64"]),
    ("eval", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html", "PyTorch 2.14", ["Set the module in evaluation mode", "Dropout"]),
    ("randomness", "https://docs.pytorch.org/docs/2.14/notes/randomness.html", "PyTorch 2.14", ["Completely reproducible results are not guaranteed"]),
    ("mps", "https://docs.pytorch.org/docs/2.14/notes/mps.html", "PyTorch 2.14", ["Metal Performance Shaders", "GPU"]),
    ("checkpoint", "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html", "讀取當日官方教程", ["optimizer’s state_dict", "resume training"]),
    ("hf", "https://huggingface.co/docs/huggingface_hub/guides/download", "讀取當日 huggingface_hub 官方文件", ["commit hash", "revision"]),
    ("http", "https://www.rfc-editor.org/rfc/rfc9110.html", "RFC 9110, June 2022", ["application-level protocol", "request/response"]),
]
records = []
for name, url, version, needles in sources:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "40", url]
    value = subprocess.run(command, capture_output=True, timeout=45)
    parser = Text()
    parser.feed(value.stdout.decode("utf-8", errors="replace"))
    content = " ".join(" ".join(parser.parts).split())
    excerpts = []
    for needle in needles:
        position = content.lower().find(needle.lower())
        if position >= 0:
            excerpts.append({"locator": needle, "excerpt": " ".join(content[position:].split()[:10])})
    records.append({
        "id": name, "url": url, "version": version,
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "exit_code": value.returncode,
        "stderr": value.stderr.decode(), "original_sha256": hashlib.sha256(value.stdout).hexdigest(),
        "original_bytes": len(value.stdout), "excerpts": excerpts,
        "full_text_retained": False,
    })
    print(json.dumps({"source": name, "exit_code": value.returncode, "excerpt_count": len(excerpts)}, ensure_ascii=False), flush=True)
Path(__file__).with_name("official-source-excerpts.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
