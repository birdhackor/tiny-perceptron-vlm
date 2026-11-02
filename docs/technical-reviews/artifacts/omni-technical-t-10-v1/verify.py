"""T.10 independent bounded CPU audit; full/raw outputs remain outside Git.

Run from repo root with uv run --extra cpu python <this file>.
Training CLI calls are explicitly intercepted before execute; only a two-update
mechanism fixture is trained. No production weights, CUDA, Modal, or publishing.
"""

import contextlib
import copy
import hashlib
import inspect
import io
import json
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
CACHE = Path('/workspace/work/reference/omni-technical-t-10-v1')
CACHE.mkdir(parents=True, exist_ok=True)
RUN_CACHE = CACHE / datetime.now(timezone.utc).strftime('run-%Y%m%dT%H%M%S%fZ')
RUN_CACHE.mkdir()
OUT = Path(__file__).with_name('execution-summary.json')

import torch
from scripts.course_experiments import compression as comp
from scripts.course_experiments import run
from scripts.course_experiments.common import Context
from scripts.prepare_data import generate_records
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import save_checkpoint, seed_everything

torch.set_num_threads(2)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def receipt(path):
    return {'path': str(path), 'bytes': Path(path).stat().st_size, 'sha256': sha(path)}

def process(argv, name, timeout=45):
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    raw = RUN_CACHE / (name + '.log')
    raw.write_text(json.dumps({'argv': argv, 'returncode': p.returncode}, ensure_ascii=False) +
                   '\nSTDOUT\n' + p.stdout + '\nSTDERR\n' + p.stderr)
    return p, receipt(raw)

summary = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-t-10-v1/verify.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'torch_git_version': torch.version.git_version, 'device': 'cpu',
                    'cuda_available': str(torch.cuda.is_available()), 'torch_threads': '2'},
    'scope': 'Bounded CPU mechanics, route dry-runs and extraction of existing historical CUDA reports. No new formal recipe or quality reproduction.',
}
source = ROOT / 'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.10.md'
text = source.read_text()
summary['source'] = receipt(source)
assert sha(source) == '62ba0ca08b006245e9bd896b673ce900f877877a2ad8e0fcf29652238dc6139d'
live = (ROOT / 'course/training.md').read_text()
live_section = re.search(r'^## T\.10 .*?(?=^## |\Z)', live, re.M | re.S).group(0)
assert live_section == text
summary['code_sha256'] = {p: sha(ROOT / p) for p in [
    'scripts/course_experiments/run.py', 'scripts/course_experiments/compression.py',
    'scripts/course_experiments/modalities.py', 'scripts/course_experiments/common.py',
    'scripts/course_experiments/text.py', 'scripts/course_experiments/applications.py',
    'scripts/infer.py', 'scripts/fetch_training_assets.py', 'scripts/prepare_data.py',
    'tiny_perceptron/alignment.py', 'tiny_perceptron/model.py', 'tiny_perceptron/data.py',
    'tiny_perceptron/training.py', 'tiny_perceptron/multimodal.py', 'tiny_perceptron/assets.py',
    'assets/training/manifest.json', 'pyproject.toml', 'uv.lock']}
summary['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()

# Read only each executable recipe's runtime config, never author plans/reviews.
summary['recipe_configs'] = {
    name: {k: v for k, v in run.experiment_spec(name).items()
           if k in ('id', 'module', 'function', 'assets', 'dependencies', 'kind')}
    for name in ('sft', 'style', 'moe', 'distillation', 'encoders', 'projector', 'vqa', 'joint', 'multimodal_distillation')
}
summary['help'] = []
for name, argv in [('runner', ['.venv/bin/python', '-m', 'scripts.course_experiments.run', '--help']),
                   ('inference', ['.venv/bin/python', 'scripts/infer.py', '--help']),
                   ('fetch', ['.venv/bin/python', 'scripts/fetch_training_assets.py', '--help'])]:
    p, raw = process(argv, name + '-help')
    assert p.returncode == 0
    summary['help'].append({'name': name, 'exit_code': p.returncode, 'raw': raw})

# Parse each literal training line through the real CLI. execute is replaced
# solely at the dispatch boundary, so this cannot start the long recipes.
summary['literal_training_route_dry_runs'] = []
for line in text.splitlines():
    if not line.startswith('.venv/bin/python -m scripts.course_experiments.run '):
        continue
    tokens = shlex.split(line)
    captured = {}
    def intercept(experiment_id, device, output, dependencies, assets, **kwargs):
        captured.update(experiment_id=experiment_id, device=device, output=str(output),
                        dependencies=str(dependencies), assets=str(assets), **kwargs)
        return {'experiment_id': experiment_id, 'elapsed_seconds': 0,
                'evidence_status': 'intercepted_before_training', 'artifacts': []}
    with patch.object(sys, 'argv', [tokens[2]] + tokens[3:]), patch.object(run, 'execute', intercept), contextlib.redirect_stdout(io.StringIO()):
        run.main()
    assert captured['output'] == str(ROOT / 'outputs/course-experiments/course-v1' / captured['experiment_id'])
    assert captured['dependencies'] == str(ROOT / 'outputs/course-experiments/course-v1')
    summary['literal_training_route_dry_runs'].append({'literal_command': line, 'dispatch': captured,
                                                     'scope': 'Actual argparse/main, execute intercepted; no training.'})

if not torch.cuda.is_available():
    try:
        run.execute('sft', 'cuda', CACHE / 'must-not-exist', CACHE, ROOT / 'assets/training')
    except RuntimeError as e:
        summary['cuda_without_gpu'] = str(e)
    else:
        raise AssertionError('CUDA silently fell back')
p, raw = process(['.venv/bin/python', '-m', 'scripts.course_experiments.run', '--experiment', 'distillation', '--device', 'mps'], 'mps-rejection')
assert p.returncode == 2 and 'invalid choice' in p.stderr
summary['mps_rejection'] = {'exit_code': 2, 'raw': raw}

# Actual fetch CLI; only --output is relocated to this review's external cache.
fetch_line = next(line for line in text.splitlines() if line.startswith('.venv/bin/python scripts/fetch_training_assets.py '))
p, raw = process(shlex.split(fetch_line) + ['--output', str(CACHE / 'fetched')], 'gsm8k-fetch')
summary['fetch'] = {'literal_command': fetch_line, 'output_override': str(CACHE / 'fetched'),
                    'exit_code': p.returncode, 'raw': raw}
assert p.returncode == 0, p.stderr
gsm = CACHE / 'fetched/reasoning-initial/gsm8k-train-first200.jsonl'
assert sha(gsm) == '03a8e89683ff5335dd79e1d8468ede692f263b4de113e59025833eeadc0f05a9'
summary['fetch']['records'] = len(gsm.read_text().splitlines())
summary['fetch']['data'] = receipt(gsm)

fixture_ctx = Context('cpu', CACHE / 'fixture', CACHE / 'fixture-deps', ROOT / 'assets/training', seed=42)
fixture_ctx.output.mkdir(exist_ok=True)
try:
    fixture_ctx.dependency('sft')
except FileNotFoundError as e:
    summary['missing_dependency_guard'] = str(e)
else:
    raise AssertionError('missing dependency unexpectedly exists')
parts = {'train': [{'family': 'a'}], 'validation': [{'family': 'b'}], 'test': [{'family': 'c'}]}
dataset_path = fixture_ctx.dependencies / 'fixture/dataset.json'
dataset_path.parent.mkdir(parents=True, exist_ok=True)
dataset_path.write_text(json.dumps(parts))
got, manifest = comp._dataset(fixture_ctx, 'fixture')
assert got == parts
summary['dataset_fixture'] = manifest
parts['test'][0]['family'] = 'a'
dataset_path.write_text(json.dumps(parts))
try:
    comp._dataset(fixture_ctx, 'fixture')
except ValueError as e:
    summary['family_leakage_rejection'] = str(e)
else:
    raise AssertionError('family leakage accepted')

rows = generate_records('attributes-sft')
prompt = 'color=red;shape=square;pitch=high;joint?'
row = next(r for r in rows if r['messages'][0]['content'] == prompt)
assert row['messages'][1]['content'] == 'square,high'
summary['prompt_target'] = row
challenge, selected = comp._challenge(fixture_ctx, 128, 24)
raw_selection = CACHE / 'challenge-selection-full.json'
raw_selection.write_text(json.dumps({'records': challenge, 'selection': selected}, ensure_ascii=False, indent=2))
summary['challenge'] = {k: v for k, v in selected.items() if k != 'skipped'}
summary['challenge']['raw'] = receipt(raw_selection)
assert selected['source_rows'] == 200 and selected['selected_source_rows'] == [14, 94]

# Run the actual necessary prerequisite snippets, including inline redefinitions.
summary['prerequisite_inline_executions'] = []
chapter = (ROOT / 'course/chapters/18.md').read_text()
for number in ('18.8', '18.9', '18.13'):
    section = re.search(r'^## ' + re.escape(number) + r' .*?(?=^## |\Z)', chapter, re.M | re.S).group(0)
    blocks = re.findall(r'```python\n(.*?)```', section, re.S)
    namespace, output = {}, io.StringIO()
    for block in blocks:
        with contextlib.redirect_stdout(output):
            exec(compile(block, 'course/chapters/18.md#' + number, 'exec'), namespace)
    summary['prerequisite_inline_executions'].append({'section': number, 'blocks': len(blocks),
        'block_sha256': [hashlib.sha256(b.encode()).hexdigest() for b in blocks], 'stdout': output.getvalue()})

# One tiny mechanism fixture, 2 AdamW updates per branch, not any full recipe.
records = [{'question': 'Q?', 'answer': 'A', 'family': 'q'}, {'question': 'R?', 'answer': 'B', 'family': 'r'}]
seed_everything(42)
teacher = TinyLM(ModelConfig(width=16, layers=1, heads=2, max_length=128))
initial = TinyLM(ModelConfig(width=8, layers=1, heads=2, max_length=128))
teacher_hash = comp._parameter_hash(teacher)
cache, _ = comp._cache_text(teacher, records, 'cpu')
summary['two_update_fixture'] = {'records': 2, 'seed': 42, 'teacher_width': 16, 'student_width': 8,
                               'layers': 1, 'heads': 2, 'scope': 'Mechanics only, random untrained teacher, no quality claim.'}
for method in ('ce', 'ce_kl'):
    _, _, result = comp._fit_text(fixture_ctx, copy.deepcopy(initial), records, method, steps=2,
                                 teacher_cache=cache if method == 'ce_kl' else None)
    summary['two_update_fixture'][method] = {k: result[k] for k in (
        'steps', 'batch_size', 'learning_rate', 'initialization_sha256', 'batch_plan_sha256',
        'effective_supervised_tokens', 'weights_changed', 'first_output_weight_gradient_norm', 'objective', 'alpha', 'temperature', 'loss_trace')}
ce, kd = (summary['two_update_fixture'][m] for m in ('ce', 'ce_kl'))
assert ce['initialization_sha256'] == kd['initialization_sha256']
assert ce['batch_plan_sha256'] == kd['batch_plan_sha256']
assert ce['effective_supervised_tokens'] == kd['effective_supervised_tokens'] == 64
assert comp._parameter_hash(teacher) == teacher_hash
summary['two_update_fixture']['teacher_unchanged'] = True

# Literal inference CLI with an explicitly synthetic, equal-logit checkpoint;
# only the checkpoint operand is relocated. Greedy argmax must choose ID 0.
seed_everything(42)
model = TinyLM(ModelConfig(width=32, layers=1, heads=2, max_length=128))
with torch.no_grad():
    model.output.weight.zero_()
summary['inference_fixture'] = []
for i, line in enumerate(line for line in text.splitlines() if line.startswith('.venv/bin/python scripts/infer.py ')):
    args = shlex.split(line)
    checkpoint = CACHE / Path(args[2]).name
    save_checkpoint(checkpoint, model, metadata={'fixture': 'zero output logits, no training, no learned-answer claim'})
    args[2] = str(checkpoint)
    p, raw = process(args, f'infer-fixture-{i}')
    assert p.returncode == 0
    report = json.loads(p.stdout)
    p_explicit, _ = process(args + ['--temperature', '0'], f'infer-fixture-{i}-explicit-zero')
    assert p_explicit.returncode == 0 and json.loads(p_explicit.stdout) == report
    ids = report['raw_ids'] if 'raw_ids' in report else report['generated_ids']
    assert len(ids) == 24 and set(ids) == {0}
    summary['inference_fixture'].append({'literal_command': line, 'checkpoint_override': str(checkpoint),
        'exit_code': 0, 'width': model.config.width, 'prompt': prompt, 'new_tokens': len(ids),
        'argmax_id': 0, 'implicit_equals_explicit_temperature_zero': True, 'raw': raw,
        'scope': 'Equal-logit fixture checks CLI/greedy/budget only; expected square,high is gold target, not fixture output.'})

summary['historical_reports'] = {}
for experiment in ('sft', 'style', 'moe', 'distillation', 'joint', 'multimodal_distillation'):
    path = ROOT / f'docs/course-experiments/results/{experiment}.json'
    report = json.loads(path.read_text())
    entry = {k: report.get(k) for k in ('revision', 'device', 'seed', 'torch_version', 'python_version',
        'gpu', 'elapsed_seconds', 'timing_scope', 'step_scale', 'evidence_status')}
    entry['source'] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
    entry['rounded_seconds'] = round(entry['elapsed_seconds'], 1)
    entry['independently_retimed'] = False
    assert entry['device'] == 'cuda' and entry['gpu'] == 'NVIDIA L4' and entry['step_scale'] == 1
    if experiment in ('distillation', 'multimodal_distillation'):
        entry['tasks'] = {}
        for name, task in report['results']['tasks'].items():
            item = {'data': task['data'], 'teacher_config': task['teacher_provenance']['config'],
                    'teacher_parameters': task['teacher_storage']['parameter_count'], 'runs': {}}
            for method, value in task['runs'].items():
                training = value.get('training', {})
                item['runs'][method] = {'parameters': value['storage']['parameter_count'],
                    'training': {k: training.get(k) for k in ('steps', 'batch_size', 'training_examples',
                        'training_sequence_chunks', 'effective_supervised_tokens', 'effective_answer_tokens',
                        'initialization_sha256', 'batch_plan_sha256')},
                    'test': {k: value['test'].get(k) for k in ('examples', 'supervised_tokens', 'max_new_tokens', 'effective_tokens', 'correct', 'exact_match')}}
            widths = ('w16_', 'w32_') if name == 'attributes' else ('w32_',) if experiment == 'distillation' else ('',)
            for prefix in widths:
                left, right = (item['runs'][prefix + m]['training'] for m in ('ce', 'ce_kl'))
                for field in ('initialization_sha256', 'batch_plan_sha256', 'steps', 'batch_size', 'effective_supervised_tokens', 'effective_answer_tokens'):
                    assert left[field] == right[field], (name, prefix, field)
            item['ce_kl_control_checks'] = 'Same init, batch plan, steps, batch size and effective-token counts for each size.'
            entry['tasks'][name] = item
    summary['historical_reports'][experiment] = entry
assert [summary['historical_reports'][e]['rounded_seconds'] for e in ('sft', 'style', 'moe', 'distillation', 'joint', 'multimodal_distillation')] == [31.7, 29.5, 65.7, 90.1, 33.4, 53.0]

summary['official_package_source_match'] = {}
for module, file in [(torch.nn.modules.loss, 'torch-loss-5c488690.py'), (torch.nn.functional, 'torch-functional-5c488690.py')]:
    remote = CACHE / file
    local = Path(inspect.getfile(module))
    summary['official_package_source_match'][module.__name__] = {'remote': receipt(remote),
        'installed_sha256': sha(local), 'identical': remote.read_bytes() == local.read_bytes()}
    assert remote.read_bytes() == local.read_bytes()
summary['completed_at'] = datetime.now(timezone.utc).isoformat()
assert not OUT.exists(), 'Never overwrite an original execution summary'
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT),
                  'scope': summary['scope']}, ensure_ascii=False))
