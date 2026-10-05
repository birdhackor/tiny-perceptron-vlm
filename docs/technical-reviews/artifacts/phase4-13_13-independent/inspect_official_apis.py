"""AST selection of remaining tensor API contracts in original source snapshots."""
import ast
import hashlib
import json
from pathlib import Path

out = Path(__file__).parent / "sources"
targets = {
    "torch--_tensor_docs.py": {"log", "exp", "clamp", "sum", "mean", "tolist"},
    "torch--_tensor.py": {"detach"},
}
manifest = []
for filename, names in targets.items():
    raw = (out / filename).read_bytes()
    text = raw.decode()
    lines = text.splitlines()
    ranges = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            if node.args[0].value in names:
                ranges.append((node.args[0].value, node.lineno, node.end_lineno))
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    ranges.append((target.id, node.lineno, node.end_lineno))
    excerpt = "\n\n".join(
        f"API {name} ORIGINAL LINES {start}-{end}\n" +
        "\n".join(f"{i}: {lines[i-1]}" for i in range(start, end+1))
        for name, start, end in ranges)
    (out / (filename + ".additional-inspected.txt")).write_text(excerpt + "\n")
    manifest.append({"file": filename, "sha256": hashlib.sha256(raw).hexdigest(),
                     "ranges": [{"api": name, "start": start, "end": end} for name, start, end in ranges]})
    print(excerpt)
(out / "additional-api-inspection-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
