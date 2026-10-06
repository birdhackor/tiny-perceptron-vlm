import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

root = Path.cwd()
base = root / 'docs/technical-reviews/artifacts/phase4-9_7-callback-20261006'
frozen = root / 'docs/technical-reviews/artifacts/phase4-9_7-independent'
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
raw = (root / 'course/chapters/09.md').read_bytes()
headings = list(re.finditer(rb'(?m)^## [^\r\n]+', raw))
index = next(i for i, h in enumerate(headings) if h[0].startswith(b'## 9.7 '))
body = raw[headings[index].start():headings[index + 1].start() if index + 1 < len(headings) else len(raw)]
assert hashlib.sha256(body).hexdigest() == 'bc667c1fa23874df19644ab3b9799c391a05fa6f75203aecaeb787f0182ab55f'
assert body == (base / 'current-section.md').read_bytes()
code = re.search(rb'```python\n(.*?)```', body, re.S)[1]
assert code == (frozen / 'fence-1.py').read_bytes()
assert code == (base / 'current-fence.py').read_bytes()
receipt = json.loads((base / 'reuse-fingerprint-receipt.json').read_bytes())
for entry in receipt['checks']:
    assert digest(root / entry['path']) == entry['expected_sha256']
for entry in receipt['source_comparisons']:
    assert digest(root / entry['current_path']) == entry['current_sha256']
    assert digest(root / entry['frozen_path']) == entry['frozen_sha256']
    assert entry['equal'] is True
render = json.loads((base / 'render-receipt.json').read_bytes())
assert render['returncode'] == 0
assert digest(root / 'course/figures/rewrite-09-07-data-task.svg') == render['svg_sha256']
assert digest(base / 'current-figure.png') == render['png_sha256']
preservation = json.loads((base / 'prior-preservation.json').read_bytes())
assert digest(root / preservation['opaque_file']) == preservation['sha256']
assert preservation['sha256'] == '2c627979f6561e2b5c329fa5e1a5d7a837fc20a9b38fd0b910619953a079a633'
print(json.dumps({
    'current_source_sha256': hashlib.sha256(body).hexdigest(),
    'current_figure_sha256': render['svg_sha256'],
    'current_fence_equals_original_executed_fence': True,
    'all_original_formal_artifact_hashes_valid': True,
    'all_relevant_repository_files_equal_frozen_executed_version': True,
    'current_render_receipt_valid': True,
    'prior_report_preserved_opaquely': preservation,
    'reused_cpu_results_date': '2026-10-05',
    'no_fence_or_model_execution_in_callback': True,
    'environment': {'python': sys.version, 'python_executable': sys.executable, 'torch_installed': importlib.metadata.version('torch'), 'platform': platform.platform(), 'device': 'CPU metadata/fingerprint check only'}
}, ensure_ascii=False, indent=2))
