"""Retain original authorities and precise AST-selected primary excerpts only."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


records = json.loads((ROOT / 'outputs/course-revision-20261005/original-source-locators.json').read_bytes())['locators']
primary = OUT / 'primary-snapshots'
primary.mkdir(exist_ok=True)
manifest = []
for index, names, target in [
    (44, ['torch.tensor', 'torch.cat', 'torch.matmul', 'torch.numel'], 'pytorch-tensor-operations.txt'),
    (43, ['numel', 'tolist'], 'pytorch-tensor-methods.txt'),
    (46, ['relu'], 'pytorch-relu.txt'),
    (42, ['__len__'], 'pytorch-tensor-length.txt'),
    (48, ['Linear'], 'pytorch-linear.txt'),
]:
    record = records[index]
    raw = (ROOT / record['original_path']).read_bytes()
    assert sha(raw) == record['sha256']
    lines = raw.splitlines(keepends=True)
    tree = ast.parse(raw)
    selected = {}
    for node in ast.walk(tree):
        key = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            key = node.name
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('add_docstr', 'add_docstr_all') and node.args:
            key = node.args[0].value if isinstance(node.args[0], ast.Constant) else ast.unparse(node.args[0])
        if key in names:
            selected[key] = node
    assert set(selected) == set(names), (target, set(selected))
    excerpts = []
    locators = []
    for name in names:
        node = selected[name]
        locators.append({'name': name, 'first_line': node.lineno, 'last_line': node.end_lineno})
        excerpts.append(f'ORIGINAL {record["url"]} lines {node.lineno}-{node.end_lineno}\n'.encode())
        excerpts.append(b''.join(lines[node.lineno-1:node.end_lineno]))
        excerpts.append(b'\n\n')
    snapshot = primary / target
    snapshot.write_bytes(b''.join(excerpts))
    manifest.append({'url': record['url'], 'version': record['version_label'], 'original_sha256': sha(raw), 'snapshot': str(snapshot.relative_to(ROOT)), 'snapshot_sha256': sha(snapshot.read_bytes()), 'read_locators': locators, 'origin': 'cached unmodified upstream source; no prior reviewer prose read'})

for source, filename, identifier, excerpt, first, last, url in [
    ('docs/technical-reviews/artifacts/fact_finish_r_3-transformer.original.pdf', 'transformer-v7.pdf', '1706.03762v7', '/tmp/factual-15_1-transformer-first5.txt', 239, 253, 'https://arxiv.org/pdf/1706.03762v7'),
    ('docs/technical-reviews/artifacts/fact_finish_r_4/originals/moe.pdf', 'sparsely-gated-moe-v1.pdf', '1701.06538v1', '/tmp/factual-15_1-moe-first4.txt', 120, 177, 'https://arxiv.org/pdf/1701.06538v1'),
]:
    raw = (ROOT / source).read_bytes()
    dest = primary / filename
    dest.write_bytes(raw)
    text = Path(excerpt).read_text()
    assert identifier in text
    chunk = '\n'.join(text.splitlines()[first-1:last]) + '\n'
    text_dest = primary / (filename.removesuffix('.pdf') + '-read-excerpt.txt')
    text_dest.write_text(chunk)
    manifest.append({'url': url, 'version': identifier, 'original_sha256': sha(raw), 'snapshot': str(dest.relative_to(ROOT)), 'snapshot_sha256': sha(raw), 'version_personally_checked': identifier, 'read_locators': 'Transformer: title/version and section 3.3 eq. (2), p.5; MoE: title/version, section 1.2 pp.2-3 and section 2 eq.(1), p.3; also section 2.1 p.4 read', 'text_excerpt': str(text_dest.relative_to(ROOT))})

(OUT / 'primary-source-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print('Verified cached original SHA values; retained five precise upstream source excerpts and two exact primary papers.')
print(json.dumps([{x['snapshot']: x['read_locators']} for x in manifest], ensure_ascii=False, indent=2))
