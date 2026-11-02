"""Read-only inspection of small pinned public course payloads; no model forward."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
DESTINATION = Path('/workspace/work/reference/omni-p6-t3-fresh-20261030/public-models')
manifest_path = ROOT / 'docs/course-experiments/public-models.json'
manifest = json.loads(manifest_path.read_text())
result = {
    'created_at': datetime.now(UTC).isoformat(),
    'scope': 'Fetch only small pinned course files, inspect payload keys with weights_only=True; zero model constructors, forward, generation, training, teacher, or final-evaluation calls.',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu'},
    'manifest': {'path': str(manifest_path.relative_to(ROOT)), 'sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
    'models': [],
}
for name in ('simple_models', 'text_foundation'):
    spec = next(x for x in manifest['models'] if x['id'] == name)
    assert sum(x['bytes'] for x in spec['files']) < 2_000_000
    command = [sys.executable, 'scripts/fetch_course_models.py', '--model', name, '--output', str(DESTINATION)]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=45)
    item = {'id': name, 'repo': spec['repo'], 'revision': spec['revision'],
            'command': command, 'returncode': process.returncode,
            'stdout': process.stdout, 'stderr': process.stderr, 'files': []}
    assert process.returncode == 0, process.stderr
    for file in spec['files']:
        path = DESTINATION / name / file['output']
        raw = path.read_bytes()
        assert len(raw) == file['bytes'] and hashlib.sha256(raw).hexdigest() == file['sha256']
        entry = {'url': f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{file['path']}",
                 'outside_git_path': str(path), 'bytes': len(raw), 'sha256': file['sha256']}
        if path.suffix == '.pt':
            saved = torch.load(path, map_location='cpu', weights_only=True)
            entry['payload_keys'] = sorted(saved)
            entry['format_version'] = saved.get('format_version')
            entry['config'] = saved.get('config')
            entry['tokenizer'] = saved.get('tokenizer')
            assert not {'optimizer', 'torch_rng', 'python_rng', 'cuda_rng', 'mps_rng', 'training_state'} & set(saved)
        if path.name == 'README.md':
            # Index presence only; do not treat model-card prose as technical authority.
            entry['is_markdown_model_card'] = raw.startswith(b'---') or b'#' in raw[:600]
        item['files'].append(entry)
    result['models'].append(item)
result['completed_at'] = datetime.now(UTC).isoformat()
print(json.dumps(result, ensure_ascii=False, indent=2))
