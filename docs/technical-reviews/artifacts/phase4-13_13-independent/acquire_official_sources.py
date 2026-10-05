"""Retrieve exact installed PyTorch source and retain own API excerpts."""
import ast
import hashlib
import json
import urllib.request
from pathlib import Path

import torch

out = Path(__file__).parent / "sources"
out.mkdir(exist_ok=True)
commit = torch.version.git_version
manifest = []
targets = {
    "torch/_torch_docs.py": {"tensor", "full", "log", "exp", "clamp", "minimum", "sum", "mean"},
    "torch/_tensor_docs.py": {"log", "exp", "clamp", "sum", "mean", "tolist"},
    "torch/_tensor.py": {"backward", "detach", "tolist"},
}
for path, names in targets.items():
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/{path}"
    with urllib.request.urlopen(url, timeout=20) as response:
        raw = response.read()
    filename = path.replace("/", "--")
    (out / filename).write_bytes(raw)
    text = raw.decode()
    lines = text.splitlines()
    tree = ast.parse(text)
    ranges = []
    if path.endswith("_tensor.py"):
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name in names:
                ranges.append((n.name, n.lineno, n.end_lineno))
    else:
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "add_docstr" and n.args:
                arg = n.args[0]
                if isinstance(arg, ast.Attribute) and arg.attr in names:
                    ranges.append((arg.attr, n.lineno, n.end_lineno))
    excerpt = "\n\n".join(
        f"API {name} ORIGINAL LINES {start}-{end}\n" +
        "\n".join(f"{i}: {lines[i-1]}" for i in range(start, end+1))
        for name, start, end in ranges)
    (out / (filename + ".inspected.txt")).write_text(excerpt + "\n")
    item = {"url": url, "version": commit, "accessed_on": "2026-10-05",
            "file": filename, "sha256": hashlib.sha256(raw).hexdigest(),
            "inspected_ranges": [{"api": name, "start": start, "end": end} for name, start, end in ranges]}
    manifest.append(item)
    print(json.dumps(item))
(out / "official-source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
