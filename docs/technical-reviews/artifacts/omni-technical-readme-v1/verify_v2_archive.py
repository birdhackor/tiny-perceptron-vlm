"""Read-only recount of the actual fixed v2 archive; no model or training."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import platform
import tarfile

root = Path('/workspace/tiny-perceptron-vlm')
manifest = json.loads((root / 'docs/selftrained/v2-manifest.json').read_text())
archive = root / manifest['package']['path']
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest == manifest['package']['sha256']
assert archive.stat().st_size == manifest['package']['bytes']
entries = {x['path']: x for x in manifest['records'] + manifest['assets']}
record_names = {x['path'] for x in manifest['records']}
counts, rows = Counter(), {}
with tarfile.open(archive, 'r:gz') as tar:
    members = [x for x in tar if x.isfile()]
    names = [x.name for x in members]
    assert len(names) == len(set(names)) == 8962
    assert set(names) == set(entries)
    for member in members:
        if member.name not in record_names:
            continue
        raw = tar.extractfile(member).read()
        assert hashlib.sha256(raw).hexdigest() == entries[member.name]['sha256']
        rows[member.name] = len(raw.splitlines())
        for line in raw.splitlines():
            counts[json.loads(line)['split']] += 1
print(json.dumps({
    'command': 'python docs/technical-reviews/artifacts/omni-technical-readme-v1/verify_v2_archive.py',
    'environment': {'python': platform.python_version(), 'device': 'cpu', 'libraries': 'Python standard library'},
    'archive_sha256': digest, 'archive_bytes': archive.stat().st_size, 'files': len(names),
    'record_rows': rows, 'splits': dict(counts),
    'scope': 'Actual tar hash/size, unique file names and every record JSONL hash/count; no new training or claim of individually rehashing all image/audio contents.'
}, ensure_ascii=False, indent=2))
