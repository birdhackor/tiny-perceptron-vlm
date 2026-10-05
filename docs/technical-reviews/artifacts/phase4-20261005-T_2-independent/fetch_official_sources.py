import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT = "5c4886908584029761b579af026dcfb627c84070"
OUT = Path(__file__).resolve().parent / "official-sources"
OUT.mkdir(exist_ok=True)
TARGETS = {
    "torch/_tensor.py": [("Tensor", "backward")],
    "torch/optim/optimizer.py": [("Optimizer", "step")],
    "torch/nn/utils/clip_grad.py": [(None, "_get_total_norm"), (None, "clip_grad_norm_")],
}
manifest = []
for file, selectors in TARGETS.items():
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/{file}"
    with urlopen(url, timeout=20) as response:
        raw = response.read()
    tree = ast.parse(raw.decode("utf-8"))
    lines = raw.decode("utf-8").splitlines(keepends=True)
    excerpts = []
    for parent, name in selectors:
        nodes = tree.body if parent is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent).body
        node = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name][-1]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        snippet = "".join(lines[start-1:node.end_lineno]).encode("utf-8")
        output = OUT / (file.replace("/", "__") + "--" + name + ".txt")
        output.write_bytes(snippet)
        excerpts.append({"selector": f"{parent + '.' if parent else ''}{name}", "lines": [start, node.end_lineno], "path": output.name, "sha256": hashlib.sha256(snippet).hexdigest()})
    manifest.append({"url": url, "version": COMMIT, "accessed_on": "2026-10-05", "complete_response_sha256": hashlib.sha256(raw).hexdigest(), "complete_response_bytes": len(raw), "retained_scope": "Only the personally inspected original functions, preserving their bytes; full response hash retained for provenance.", "excerpts": excerpts})
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
