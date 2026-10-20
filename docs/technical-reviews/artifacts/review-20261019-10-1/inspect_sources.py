"""用 curl 讀官方原文，只保存支持主張的短摘錄。"""

import hashlib
import html
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

SOURCES = [
    ("tensor", "https://docs.pytorch.org/docs/2.14/tensors.html", "A torch.Tensor is a multi-dimensional matrix"),
    ("meshgrid", "https://docs.pytorch.org/docs/2.14/generated/torch.meshgrid.html", "If “ij” is selected"),
    ("unsqueeze", "https://docs.pytorch.org/docs/2.14/generated/torch.unsqueeze.html", "Returns a new tensor with a dimension of size one inserted"),
    ("tolist", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.tolist.html", "Returns the tensor as a (nested) list"),
    ("zeros", "https://docs.pytorch.org/docs/2.14/generated/torch.zeros.html", "Returns a tensor filled with the scalar value 0"),
    ("broadcast", "https://docs.pytorch.org/docs/2.14/notes/broadcasting.html", "Two tensors are"),
    ("permute", "https://docs.pytorch.org/docs/2.14/generated/torch.permute.html", "Returns a view of the original tensor"),
    ("python_defaults", "https://docs.python.org/3.13/tutorial/controlflow.html", "The most useful form is to specify a default value"),
    ("python_dictionary", "https://docs.python.org/3.13/tutorial/datastructures.html", "Unlike sequences, which are indexed"),
    ("python_division", "https://docs.python.org/3.13/reference/expressions.html", "The / (division) and // (floor division) operators"),
    ("png", "https://www.w3.org/TR/2025/REC-png-3-20250624/", "Each pixel is an R,G,B triple"),
    ("linear", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html", "Applies an affine linear transformation"),
]


def inspect_source(item):
    identifier, url, needle = item
    command = ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", "45", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    document = raw.decode("utf-8")
    title_match = re.search(r"<title[^>]*>(.*?)</title>", document, re.S | re.I)
    title = html.unescape(re.sub("<[^>]+>", " ", title_match[1])) if title_match else ""
    main = re.search(r'<(?:main|article)\b[^>]*>(.*?)</(?:main|article)>', document, re.S | re.I)
    body = main[1] if main else document
    body = re.sub(r"<(script|style)\b.*?</\1>", "", body, flags=re.S | re.I)
    text = re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", " ", body))).strip()
    text = text.replace("¶", "")
    found = text.casefold().find(needle.casefold())
    if found < 0:
        return {"id": identifier, "url": url, "title": title, "error": "官方文件找不到指定定位", "locator_search": needle}
    excerpt = " ".join(text[found:].split()[:24])
    return {"id": identifier, "url": url, "title": title, "accessed_on": datetime.now().astimezone().date().isoformat(), "command_argv": command, "http_result": "curl exit 0", "read_body_sha256": hashlib.sha256(raw).hexdigest(), "locator_search": needle, "excerpt": excerpt}


with ThreadPoolExecutor(max_workers=5) as pool:
    records = list(pool.map(inspect_source, SOURCES))
print(json.dumps({"scope": "官方文件的相關局部原文；不保存第三方全文。", "sources": records}, ensure_ascii=False, indent=2))
