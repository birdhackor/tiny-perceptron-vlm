"""Execute the fully read frozen 1.1 notebook cold and its stated input variations."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[4]
base = Path(__file__).parent
source = root / "notebooks/01/1.1.ipynb"
original = nbformat.read(source, as_version=4)
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
if source_sha != "74258e6136b58a771ca54e7cd0f85bdf3dd857774de08f037177b4e22dafede3":
    raise SystemExit("Notebook does not match the reviewed freeze")
cases = []
for name in ["original", "reverse", "bird"]:
    book = copy.deepcopy(original)
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        if name == "reverse":
            cell.source = cell.source.replace("chars = sorted(set(text))", "chars = sorted(set(text), reverse=True)")
        if name == "bird":
            cell.source = cell.source.replace('text = "貓看狗，狗看貓。"', 'text = "鳥看狗，狗看鳥。"')
        cell.outputs = []
        cell.execution_count = None
    client = NotebookClient(book, timeout=60, kernel_name="tiny-perceptron", resources={"metadata": {"path": str(root / "notebooks/01")}})
    client.execute()
    target = base / f"1-1-{name}-executed.ipynb"
    if target.exists():
        raise SystemExit("Refusing to overwrite executed notebook evidence")
    nbformat.write(book, target)
    output = [{"cell": i, "execution_count": cell.execution_count, "stdout": "".join(o.get("text", "") for o in cell.get("outputs", []) if o.output_type == "stream"), "errors": [o.get("ename") for o in cell.get("outputs", []) if o.output_type == "error"]} for i, cell in enumerate(book.cells) if cell.cell_type == "code"]
    case = {"name": name, "path": str(target.relative_to(root)), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "outputs": output}
    cases.append(case)
    print(json.dumps(case, ensure_ascii=False), flush=True)
record = {"command": "/tmp/p7-technical-a-w1-g4z24kt8/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p7_technical_a/w1-and-1-1-notebook.py", "source": str(source.relative_to(root)), "source_sha256": source_sha, "environment": {"python": sys.version.split()[0], "kernel": "tiny-perceptron", "device": "Linux CPU"}, "cases": cases, "scope": "fresh local kernels executing every current 1.1 cell, including local preparation and displayed figure; no Colab login or Windows execution"}
dest = base / "w1-and-1-1-notebook-result.json"
dest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print("sha256", hashlib.sha256(dest.read_bytes()).hexdigest(), flush=True)
