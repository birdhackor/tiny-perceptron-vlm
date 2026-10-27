"""Bounded read-only/CPU verification of README claims; scratch outputs are isolated."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys

ROOT = Path('/workspace/tiny-perceptron-vlm')
SCRATCH = Path('/workspace/work/tokenizer-dynamic-20261027/readme')
OUT = ROOT / 'docs/technical-reviews/artifacts/tokenizer-20261027/readme'
sys.path.insert(0, str(ROOT))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads((ROOT / path).read_bytes())

def run(args):
    r = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=60)
    return {'argv': args, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}

import torch
torch.set_num_threads(1)
result = {'environment': {'python': sys.version, 'torch': torch.__version__, 'device': 'cpu', 'platform': platform.platform()}}
index = load('course/lesson-index.json')
nb = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'notebooks').rglob('*.ipynb'))
extra = {}
for name in ['course/README.md', 'course/first-steps.md', 'course/training.md', 'course/glossary.md']:
    extra[name] = re.findall(r'^## ([\dA-Z]+\.\d+) ', (ROOT / name).read_text(), re.M)
result['counts'] = {'lesson_index': len(index), 'notebooks': len(nb), 'index_paths_equal_notebooks': sorted(x['notebook'] for x in index) == nb, 'extra_numbered_ids': extra, 'extra_count': sum(map(len, extra.values())), 'numbered_total': len(index) + sum(map(len, extra.values()))}
bootstrap_errors = []
code_notebooks = 0
for item in index:
    notebook = load(item['notebook'])
    code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    if code:
        code_notebooks += 1
        first = ''.join(code[0]['source'])
        if not all(x in first for x in ['import google.colab', 'git', 'clone', 'pip', 'install', 'GIT_LFS_SKIP_SMUDGE']):
            bootstrap_errors.append(item['id'])
result['bootstrap'] = {'code_notebooks': code_notebooks, 'first_code_bootstrap_errors': bootstrap_errors}
source = (ROOT / 'README.md').read_text()
local_links = []
for destination in re.findall(r'\]\(([^)]+)\)', source):
    if '://' in destination or destination.startswith('#'):
        continue
    file, _, anchor = destination.partition('#')
    path = ROOT / file
    valid = path.exists()
    if anchor and valid:
        valid = bool(re.search(r'^## ' + re.escape(anchor) + r'\s', path.read_text(), re.M))
    local_links.append({'target': destination, 'exists_and_numbered_anchor': valid})
result['local_links'] = local_links
result['commands'] = [run([sys.executable, 'scripts/check_env.py']), run([sys.executable, 'scripts/fetch_training_assets.py', '--list']), run([sys.executable, 'scripts/fetch_course_models.py', '--list']), run([sys.executable, '-m', 'ipykernel', 'install', '--help']), run(['uv', 'export', '--frozen', '--offline', '--extra', 'cpu', '--group', 'notebook', '--no-hashes', '--output-file', str(SCRATCH / 'cpu-notebook-requirements.txt')])]

# Existing preparation CLI with only the destination redirected to scratch.
prep = run([sys.executable, 'scripts/prepare_data.py', '--kind', 'toy-text', '--output', str(SCRATCH / 'generated')])
assert prep['returncode'] == 0, prep
result['prepare'] = prep
from scripts import train
instances = []
base = train.TinyLM
class ObservedTinyLM(base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.before = {k: v.detach().clone() for k, v in self.state_dict().items()}
        instances.append(self)
train.TinyLM = ObservedTinyLM
original_step = torch.optim.AdamW.step
original_save = train.save_checkpoint
def forbidden(*args, **kwargs):
    raise AssertionError('dry-run must not update or save weights')
torch.optim.AdamW.step = forbidden
train.save_checkpoint = forbidden
argv = [str(ROOT / 'scripts/train.py'), '--task', 'text', '--data', str(SCRATCH / 'generated/toy-text/train.jsonl'), '--device', 'cpu', '--output', str(SCRATCH / 'dry-run.pt')]
before_argv = sys.argv
stream = io.StringIO()
try:
    sys.argv = argv
    with contextlib.redirect_stdout(stream):
        train.main()
finally:
    sys.argv = before_argv
    torch.optim.AdamW.step = original_step
    train.save_checkpoint = original_save
lines = stream.getvalue().splitlines()
report = json.loads(lines[-1])
assert report['mode'] == 'dry-run-no-weight-update'
assert len(report['history']) == 1
assert math.isfinite(report['history'][0]['loss'])
assert math.isfinite(report['history'][0]['grad_norm']) and report['history'][0]['grad_norm'] > 0
unchanged = all(torch.equal(v, model.before[k]) for model in instances for k, v in model.state_dict().items())
assert unchanged and not (SCRATCH / 'dry-run.pt').exists()
result['dry_run'] = {'argv': argv, 'instrumentation': 'TinyLM captures initial tensor states; AdamW.step and save_checkpoint raise if reached; CPU/destination explicitly selected', 'stdout': lines, 'all_state_tensors_unchanged': unchanged, 'checkpoint_absent': True}

from tiny_perceptron.selftrained.model import SelftrainedConfig, LimitedAssistant
v2 = load('docs/selftrained/results/v2-final-public-results.json')
result['v2'] = {'hf_revision': v2['hf_revision'], 'data_scope': v2['data_scope'], 'architectures': {}}
for name, item in v2['architectures'].items():
    model = LimitedAssistant(SelftrainedConfig(**item['capacity']['config']))
    parameters = sum(x.numel() for x in model.parameters())
    assert parameters == item['capacity']['parameters']
    test = item['final_test']
    metrics_path = ROOT / test['source_metrics_path']
    result['v2']['architectures'][name] = {'counted_parameters': parameters, 'final_test_count': test['count'], 'evaluation_complete': test['evaluation_complete'], 'capability_acceptance_passed': item['capability_acceptance_passed'], 'task_counts': {k:v['count'] for k,v in test['per_task_final_reply'].items()}, 'metrics_path': test['source_metrics_path'], 'raw_metrics_available': metrics_path.exists()}
    del model
manifest = load('docs/selftrained/v2-manifest.json')
result['v2_manifest'] = {'package': manifest['package'], 'record_containers': {k:len(v) for k,v in manifest['records'].items()} if isinstance(manifest['records'],dict) else len(manifest['records']), 'asset_containers': {k:len(v) for k,v in manifest['assets'].items()} if isinstance(manifest['assets'],dict) else len(manifest['assets'])}
result['v2_stages'] = {}
for name in ['docs/selftrained/results/v2-moe-native-balanced-4000-training-stage-index.json', 'docs/selftrained/results/v2-dense-weighted-10000-training-stage-index.json']:
    d=load(name)
    result['v2_stages'][name] = {k:d[k] for k in ['completed_steps','selected_step','run_id']}
public = load('docs/course-experiments/public-models.json')
weights = [f for m in public['models'] for f in m['files'] if Path(f['path']).suffix in {'.pt','.pth','.safetensors','.ckpt'}]
result['public_counts'] = {'experiments': len(public['models']), 'weights': len(weights), 'capstone_models': len(load('docs/course-experiments/capstone-public.json')['models'])}
products = load('docs/lesson-products/results/final-report-v1.json')
freeze = load('docs/lesson-products/results/settings-freeze-v1.json')
started = load('docs/lesson-products/results/settings-freeze-v1.test-started.json')
assert sha(ROOT / 'docs/lesson-products/results/settings-freeze-v1.json') == products['settings_freeze_sha256'] == started['settings_freeze_sha256']
result['products'] = {'b_high5_boxes': products['b']['5']['boxes'], 'b_high5_sum': products['b']['5']['sum'], 'c': products['c'], 'c_wording': products['shifts']['c_wording'], 'freeze_time':freeze['recorded_at'], 'test_started':started['recorded_at'], 'report_time':products['recorded_at'], 'test_observed_at_freeze':freeze['test_outcomes_observed'], 'scope': products['scope']}
assert freeze['recorded_at'] < started['recorded_at'] < products['recorded_at']
assert all(x['exists_and_numbered_anchor'] for x in local_links)
(OUT / 'audit-output.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'counts':result['counts'], 'dry_run': json.loads(result['dry_run']['stdout'][-1]), 'unchanged':unchanged, 'parameters':{k:v['counted_parameters'] for k,v in result['v2']['architectures'].items()}, 'public_counts': result['public_counts'], 'command_returncodes':[x['returncode'] for x in result['commands']]}, ensure_ascii=False, indent=2))
