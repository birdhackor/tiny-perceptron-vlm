"""Save precise source excerpts directly from version-pinned upstream PyTorch."""
import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

artifact = Path(__file__).resolve().parent
version = "v2.9.0"
items = []
for rel in ["torch/_torch_docs.py", "torch/ao/quantization/observer.py", "torch/_tensor_docs.py", "docs/source/tensor_attributes.rst"]:
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{version}/{rel}"
    with urlopen(url, timeout=30) as response:
        raw = response.read()
    text = raw.decode()
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text) if rel.endswith('.py') else None
    ranges = []
    if rel.endswith('_torch_docs.py'):
        wanted = {'linspace', 'arange', 'round', 'numel', 'clamp', 'abs', 'mean', 'max'}
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if call.args and isinstance(call.args[0], ast.Attribute) and call.args[0].attr in wanted:
                    ranges.append((call.args[0].attr, node.lineno, node.end_lineno))
    elif rel.endswith('observer.py'):
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in {'UniformQuantizationObserverBase', 'MinMaxObserver'}:
                if node.name == 'MinMaxObserver':
                    doc = node.body[0]
                    ranges.append(('MinMaxObserver documentation', doc.lineno, doc.end_lineno))
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name in {'_calculate_qparams', 'calculate_qparams'}:
                        ranges.append((node.name+'.'+child.name, child.lineno, child.end_lineno))
    elif rel.endswith('_tensor_docs.py'):
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value in {'element_size', 'float', 'to'}:
                    ranges.append((call.args[0].value, node.lineno, node.end_lineno))
    else:
        ranges.append(('dtype definitions and table', 1, 108))
    assert ranges
    saved = artifact / (Path(rel).stem + '-v2.9.0-excerpts.txt')
    with saved.open('w') as out:
        out.write(f'SOURCE: {url}\nVERSION: {version}\nORIGINAL_SHA256: {hashlib.sha256(raw).hexdigest()}\n')
        for label, start, end in ranges:
            out.write(f'\nLOCATOR: {label}, original lines {start}-{end}\n')
            for number in range(start, end + 1):
                out.write(f'{number}: {lines[number-1]}')
    item = {'url': url, 'version': version, 'accessed_on': '2026-10-05',
            'original_sha256': hashlib.sha256(raw).hexdigest(), 'original_bytes': len(raw),
            'snapshot': saved.name, 'snapshot_sha256': hashlib.sha256(saved.read_bytes()).hexdigest(),
            'locators': ranges}
    items.append(item)
    print(json.dumps(item))
(artifact/'official-fetch-provenance.json').write_text(json.dumps(items,indent=2)+'\n')
