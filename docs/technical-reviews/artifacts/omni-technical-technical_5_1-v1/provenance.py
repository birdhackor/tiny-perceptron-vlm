"""Inspect primary run/export provenance and CPU optimizer diagnostic controls."""

import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path.cwd()
CACHE = Path('/workspace/work/reference/omni-technical-technical_5_1-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


runpath = Path('docs/course-experiments/results/text_foundation.json')
run = json.loads(runpath.read_text())
releasepath = Path('docs/course-experiments/releases/text_foundation.json')
release = json.loads(releasepath.read_text())
exportpath = Path('checkpoints/course/text_foundation/export-manifest.json')
export = json.loads(exportpath.read_text())
publicpath = Path('docs/course-experiments/public-releases/text_foundation.json')
public = json.loads(publicpath.read_text())
published = {v['output']: v for v in public['release']['public_manifest']['files']}
originals = {v['path']: v for v in run['artifacts']}
exports = {v['output']: v for v in export['files']}
assert run['gpu'] == 'NVIDIA L4' and run['device'] == 'cuda'
assert run['step_scale'] == 1 and run['evidence_status'] == 'complete_run'
assert run['results']['training']['steps'] == 600
assert run['results']['training']['history'][-1]['step'] == 600
assert export['provenance']['revision'] == run['revision'] == release['revision']
assert release['private_source'] == {k: run['hf'][k] for k in ('repo', 'revision', 'prefix')}
checks = {}
for name in ['start.pt', 'model.pt']:
    item = exports[name]
    assert item['source_sha256'] == originals[name]['sha256']
    assert item['sha256'] == sha(exportpath.parent / name) == published[name]['sha256']
    assert sha(exportpath) == published['export-manifest.json']['sha256']
    checks[name] = {'original_full_checkpoint_sha256': item['source_sha256'],
                    'public_export_sha256': item['sha256'], 'public_export_bytes': item['bytes']}
controls = {}
for mode in ['only_backward', 'clear_before_step', 'wrong_parameter']:
    p = torch.nn.Parameter(torch.tensor(2.0))
    q = torch.nn.Parameter(torch.tensor(3.0))
    optimizer = torch.optim.AdamW([q] if mode == 'wrong_parameter' else [p], lr=.03, weight_decay=0.)
    p.square().backward()
    initial_gradient = p.grad.item()
    if mode == 'clear_before_step':
        optimizer.zero_grad()
    if mode != 'only_backward':
        optimizer.step()
    assert p.item() == 2.0 and initial_gradient == 4.0
    controls[mode] = {'parameter_before': 2.0, 'initial_gradient': initial_gradient,
                      'parameter_after': p.item(), 'optimizer_state_count': len(optimizer.state),
                      'parameter_gradient_after': None if p.grad is None else p.grad.item()}
help_command = [sys.executable, '-m', 'scripts.course_experiments.run', '--help']
help_result = subprocess.run(help_command, capture_output=True, text=True, check=True)
assert '--experiment' in help_result.stdout and '--device' in help_result.stdout
assert '{cpu,cuda}' in help_result.stdout
api = subprocess.run([sys.executable, '-m', 'scripts.course_experiments.run', '--list-assets', 'text_foundation'],
                     capture_output=True, text=True, check=True)
assert api.stdout == ''
print(json.dumps({
    'checked_at': datetime.datetime.now(datetime.UTC).isoformat(),
    'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'device': 'cpu'},
    'primary_files': {str(p): sha(p) for p in [runpath, releasepath, exportpath, publicpath]},
    'export_to_original_full_checkpoint_links': checks,
    'primary_run_environment': {k: run[k] for k in ['gpu', 'device', 'seed', 'step_scale', 'evidence_status']},
    'recorded_steps': run['results']['training']['steps'],
    'last_history_step': run['results']['training']['history'][-1]['step'],
    'raw_step600_checkpoint_manifest_entry': originals['model-step-600.pt'],
    'diagnostic_scalar_controls': controls,
    'cli_help_command': help_command, 'cli_help_exit': help_result.returncode,
    'cli_help_stdout': help_result.stdout, 'list_assets_exit': api.returncode,
    'list_assets_stdout': api.stdout,
    'original_download_receipts': json.loads((CACHE / 'download-receipts-v1.json').read_text()),
    'book_source_receipt': {'url': 'https://www.deeplearningbook.org/contents/ml.html',
        'version': 'Goodfellow, Bengio, Courville, Deep Learning (2016), chapter 5 official HTML read 2026-10-29',
        'cache': str(CACHE / 'ml.html'), 'sha256': sha(CACHE / 'ml.html'),
        'locator': '5.2, p.108-109: optimization/training error versus new-input error; 5.3, p.119: separate validation subset',
        'short_quote': 'The ability to perform well on previously unobserved inputs is called generalization.'},
    'access_limits': 'Private full model-step-600.pt returned HTTP 401. No private optimizer tensors were inspected. '
                     'The recorded L4/600 metadata is checked against the primary run record, original hashed code, '
                     'and published start/end weights, not a newly executed GPU run.'
}, ensure_ascii=False, indent=2) + '\n')
