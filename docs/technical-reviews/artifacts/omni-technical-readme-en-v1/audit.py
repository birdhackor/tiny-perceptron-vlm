"""Bounded CPU/source audit for the independently assigned auxiliary README_en.

Full command output and third-party originals stay in /workspace/work, outside Git.
No GPU, optimizer update, publishing, or editing of authored sources is performed.
"""
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path('/workspace/tiny-perceptron-vlm')
HERE = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/omni-technical-readme-en-v1/cpu-run2')
CACHE.mkdir(parents=True, exist_ok=True)
SANDBOX = CACHE / 'command-cwd'
SANDBOX.mkdir(exist_ok=True)
if not (SANDBOX / 'scripts').exists():
    (SANDBOX / 'scripts').symlink_to(ROOT / 'scripts', target_is_directory=True)

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    return json.loads((ROOT / p).read_bytes())

def receipt(p):
    return {'path': p, 'sha256': digest(ROOT / p)}

env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONPATH=str(ROOT))
torch.set_num_threads(2)
summary = {
    'created_at': datetime.now(UTC).isoformat(),
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'torch_git': str(torch.version.git_version), 'device': 'cpu', 'threads': '2', 'platform': platform.platform()},
    'cache': str(CACHE), 'commands': [],
}

def run(name, args, cwd=ROOT, timeout=45):
    command = ['uv', 'run', '--extra', 'cpu', 'python', *args]
    if cwd != ROOT:
        command = ['uv', 'run', '--project', str(ROOT), '--extra', 'cpu', 'python', *args]
    started = datetime.now(UTC).isoformat()
    try:
        p = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
        text = p.stdout + '\n[stderr]\n' + p.stderr
        code = p.returncode
    except subprocess.TimeoutExpired as e:
        text = 'TIMEOUT\n' + str(e.stdout) + '\n' + str(e.stderr)
        code = 'timeout'
    raw = CACHE / (name + '.txt')
    raw.write_text(text)
    entry = {'name': name, 'command': command, 'cwd': str(cwd), 'timeout_seconds': timeout, 'started_at': started, 'returncode': code, 'raw_path': str(raw), 'raw_sha256': digest(raw)}
    lines = text.splitlines()
    jsons = []
    for line in lines:
        try: jsons.append(json.loads(line))
        except (ValueError, TypeError): pass
    entry['key_output'] = jsons[-1] if name in {'dry-run'} and jsons else '\n'.join(lines[:12])[:1800]
    if name.startswith('list-'):
        entry['key_output'] = {'json_line_count': len(jsons), 'first': jsons[0] if jsons else None, 'last': jsons[-1] if jsons else None}
    summary['commands'].append(entry)
    return entry, jsons

run('check-env', ['scripts/check_env.py'])
run('prepare-toy', ['scripts/prepare_data.py', '--kind', 'toy-text'], SANDBOX)
checkpoint = SANDBOX / 'checkpoints/text.pt'
before = checkpoint.exists()
dry, _ = run('dry-run', ['scripts/train.py', '--task', 'text', '--data', 'data/generated/toy-text/train.jsonl'], SANDBOX)
summary['dry_run_checkpoint_exists'] = {'before': before, 'after': checkpoint.exists()}
run('list-assets', ['scripts/fetch_training_assets.py', '--list'])
run('list-models', ['scripts/fetch_course_models.py', '--list'])
run('public-text-model', ['scripts/check_course_models.py', '--model', 'text_foundation', '--output', str(CACHE / 'models'), '--evidence', str(CACHE / 'model-check')], timeout=55)

chapters = sorted((ROOT / 'course/chapters').glob('*.md'))
headings = {}
for p in chapters:
    headings[str(p.relative_to(ROOT))] = re.findall(r'^## ([0-9ABC]+\.[0-9]+)\s', p.read_text(), re.M)
guide_counts = {}
for p in ['course/README.md', 'course/first-steps.md', 'course/training.md', 'course/glossary.md']:
    guide_counts[p] = len(re.findall(r'^## [RWTG]\.[0-9]+\s', (ROOT / p).read_text(), re.M))
summary['inventory'] = {'chapters': len([p for p in chapters if re.fullmatch(r'[0-9]{2}', p.stem) and p.stem != '00']), 'extension_files': [p.name for p in chapters if p.stem in ['0A','0B','0C']], 'lesson_sections': sum(map(len, headings.values())), 'numbered_guides': guide_counts, 'total_numbered': sum(map(len, headings.values())) + sum(guide_counts.values()), 'notebook_count': len(list((ROOT/'notebooks').rglob('*.ipynb')))}
additions = ['7.26','9.17','9.18','10.12','12.17','12.18','14.19','16.15','16.16','17.16','17.17','19.21','19.22','19.23','19.24','20.14','20.15','20.16']
summary['inventory']['18_additions_found'] = {x: sum(x in values for values in headings.values()) for x in additions}
summary['inventory']['lesson_source_inputs'] = [receipt(str(p.relative_to(ROOT))) for p in chapters]

models = load('docs/course-experiments/public-models.json')['models']
cap = load('docs/course-experiments/capstone-public.json')
summary['public_models'] = {'groups': len(models), 'weight_files': sum(Path(f['path']).suffix in {'.pt','.pth','.ckpt','.safetensors'} for m in models for f in m['files']), 'text_foundation_bytes': sum(f['bytes'] for m in models if m['id']=='text_foundation' for f in m['files']), 'capstone_keys': list(cap), 'manifest': receipt('docs/course-experiments/public-models.json'), 'capstone_manifest': receipt('docs/course-experiments/capstone-public.json')}
if 'files' in cap: summary['public_models']['capstone_files'] = len(cap['files'])
if 'models' in cap: summary['public_models']['capstone_weight_files'] = sum(Path(f['path']).suffix in {'.pt','.pth','.ckpt','.safetensors'} for m in cap['models'] for f in m['files'])

v2 = load('docs/selftrained/results/v2-final-public-results.json')
summary['historical_v2'] = {'source': receipt('docs/selftrained/results/v2-final-public-results.json'), 'hf_revision': v2['hf_revision'], 'architectures': {}}
for arch, v in v2['architectures'].items():
    test = v['final_test']
    summary['historical_v2']['architectures'][arch] = {'parameters': v['capacity']['parameters'], 'initialization': v['capacity']['initialization'], 'test_count': test['count'], 'evaluation_complete': test['evaluation_complete'], 'all_criteria_passed': v['capability_acceptance_passed'], 'source_metrics_path': test['source_metrics_path'], 'source_metrics_sha256': test['metrics_sha256'], 'per_task_counts': {k: x['count'] for k,x in test['per_task_final_reply'].items()}}
    raw = ROOT / test['source_metrics_path']
    summary['historical_v2']['architectures'][arch]['raw_metrics_available'] = raw.is_file()
    if raw.is_file():
        summary['historical_v2']['architectures'][arch]['raw_metrics_hash_matches'] = digest(raw) == test['metrics_sha256']
selection = load('docs/selftrained/results/v2-final-training-source-selection.json')
summary['historical_v2']['selection_keys'] = list(selection)
for arch in ['moe','dense']:
    item = selection[arch+'_joint']
    summary['historical_v2'][arch+'_selection'] = {k:v for k,v in item.items() if k in ['selected_step','completed_requested_steps','completed_steps','criterion','reason','selection_rule']}
    raw_path = f'docs/selftrained/results/training-raw/{"moe-native" if arch=="moe" else "dense-weighted"}/raw/train-receipt.json'
    raw = load(raw_path)
    summary['historical_v2'][arch+'_train_receipt'] = {'source': receipt(raw_path), **{k:v for k,v in raw.items() if k in ['steps','train_records','validation_records','test_used_for_selection','seed','environment','best_step','selected_step','device','torch']}}
manifest = load('docs/selftrained/v2-manifest.json')
summary['v2_data'] = {'source': receipt('docs/selftrained/v2-manifest.json'), 'archive_bytes': manifest['package']['bytes'], 'listed_records': len(manifest['records']), 'listed_assets': len(manifest['assets']), 'listed_files': len(manifest['records'])+len(manifest['assets']), 'scope': manifest['scope']}

prod = load('docs/lesson-products/results/final-report-v1.json')
summary['lesson_products'] = {'source': receipt('docs/lesson-products/results/final-report-v1.json'), 'recorded_at': prod['recorded_at'], 'freeze_sha256': prod['settings_freeze_sha256'], 'b_depth5': {k:prod['b']['5'][k] for k in ['boxes','sum']}, 'c': prod['c'], 'top_keys': list(prod)}
for p in ['docs/lesson-products/results/pretest-freeze-record-v1.json','docs/lesson-products/results/settings-freeze-v1.test-started.json']:
    summary['lesson_products'][Path(p).stem] = {'source': receipt(p), 'record': load(p)}

torch_sources = []
for name, obj in [('cross_entropy', torch.nn.CrossEntropyLoss), ('backward', torch.autograd.backward), ('clip_grad_norm_', torch.nn.utils.clip_grad_norm_)]:
    path = Path(inspect.getsourcefile(obj))
    lines, start = inspect.getsourcelines(obj)
    (CACHE / ('torch-'+name+'.txt')).write_text(''.join(lines))
    torch_sources.append({'api': name, 'distribution_file': str(path), 'distribution_file_sha256': digest(path), 'start_line': start, 'end_line': start+len(lines)-1, 'original_excerpt_cache': str(CACHE / ('torch-'+name+'.txt')), 'original_excerpt_sha256': digest(CACHE / ('torch-'+name+'.txt'))})
summary['torch_original_sources'] = torch_sources
probs = torch.tensor([0.1,0.5,0.9])
summary['transparent_loss_example'] = {'correct_class_probabilities': probs.tolist(), 'negative_log_probabilities': (-probs.log()).tolist(), 'explanation': 'For a single hard target, unweighted cross entropy is -log p(target); same batch mean is compared under the same mask/reduction.'}

target = HERE / 'audit-summary.json'
with target.open('x') as f: json.dump(summary, f, indent=2, ensure_ascii=False); f.write('\n')
print(json.dumps({'artifact': str(target), 'sha256': digest(target), 'command_results': [(x['name'], x['returncode']) for x in summary['commands']], 'inventory': {k:v for k,v in summary['inventory'].items() if k!='lesson_source_inputs'}}, ensure_ascii=False))
