"""Read-only independent curriculum checks; no trained/teacher model forward.

Existing raw measurements are counted again. Two freshly initialized tiny course
constructors only count parameters; the artificial gradient example uses 12
logits and is not a training-quality experiment.
"""
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent
inputs = []

def read(path):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    raw = p.read_bytes()
    inputs.append({'path': str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)

freeze = read('docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18.json')
page = next(x for x in freeze['pages'] if x['page_id'] == 'curriculum')
source_check = {}
for f in [page['snapshot'], page['source'], *page['figures_sha256']]:
    raw = (ROOT / f).read_bytes()
    source_check[f] = hashlib.sha256(raw).hexdigest()
assert source_check[page['snapshot']] == source_check[page['source']] == page['source_sha256']
assert all(source_check[p] == h for p, h in page['figures_sha256'].items())
lesson_ids = {}
for p in sorted((ROOT / 'course/chapters').glob('*.md')):
    for sid, title in re.findall(r'^## ([A-Z\d]+\.\d+) (.+)$', p.read_text(), re.M):
        assert sid not in lesson_ids
        lesson_ids[sid] = title
front = {p: len(re.findall(r'^## ([A-Z\d]+\.\d+) (.+)$', (ROOT / p).read_text(), re.M)) for p in ['course/README.md', 'course/first-steps.md', 'course/training.md', 'course/glossary.md']}
new_ids = ['7.26','9.17','9.18','10.12','12.17','12.18','14.19','16.15','16.16','17.16','17.17','19.21','19.22','19.23','19.24','20.14','20.15','20.16']
sync = subprocess.run([sys.executable, str(ROOT / 'scripts/build_course.py'), '--check'], cwd=ROOT, text=True, capture_output=True)
assert sync.returncode == 0, sync.stdout + sync.stderr
notebooks = {}
bundle = ROOT / 'outputs/omni-P5-current-kernel-bundle-v10'
for sid in new_ids:
    paths = list(bundle.rglob(sid + '.ipynb'))
    assert len(paths) == 1, sid
    j = read(paths[0])
    code = [c for c in j['cells'] if c['cell_type'] == 'code']
    notebooks[sid] = {'code_cells': len(code), 'executed': all(c.get('execution_count') is not None for c in code), 'error_outputs': sum(o.get('output_type') == 'error' for c in code for o in c.get('outputs', []))}
    assert notebooks[sid]['executed'] and not notebooks[sid]['error_outputs']

manifest = read('docs/selftrained/v2-manifest.json')
archive = ROOT / 'assets/training/selftrained-v2.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['package']['sha256']
record_counts = Counter()
tasks = defaultdict(Counter)
ocr_glyphs = defaultdict(set)
ocr_lengths = defaultdict(set)
ocr_fonts = defaultdict(set)
voice_intents = defaultdict(set)
vision_classes = defaultdict(set)
data_member_hashes = {}
with tarfile.open(archive) as tf:
    for member in manifest['records']:
        raw = tf.extractfile(member['path']).read()
        assert hashlib.sha256(raw).hexdigest() == member['sha256']
        data_member_hashes[member['path']] = member['sha256']
        for line in raw.splitlines():
            row = json.loads(line)
            split = row['split']
            sup = row['supervision']
            record_counts[split] += 1
            tasks[split][row['task']] += 1
            if row['task'] == 'ocr':
                ocr_glyphs[split].update(sup['ocr_text'])
                ocr_lengths[split].add(len(sup['ocr_text']))
                ocr_fonts[split].add(sup.get('font_family', 'not_in_row'))
            if row['task'].startswith('voice'):
                voice_intents[split].add(sup['intent'])
            for x in sup.get('vision_labels', []):
                if x >= 0: vision_classes[split].add(x)

import torch
import torch.nn.functional as F
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
torch.set_num_threads(1)
torch.manual_seed(0)
v2 = {}
for architecture, stage in [('moe', 'moe-native'), ('dense', 'dense-weighted')]:
    train = read(f'docs/selftrained/results/training-raw/{stage}/raw/train-receipt.json')
    config = train['config']
    model = LimitedAssistant(SelftrainedConfig(**config))
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == train['total_parameters']
    del model
    metrics = read(f'docs/selftrained/results/public-raw/{architecture}/test/metrics.json')
    receipt = read(f'docs/selftrained/results/public-raw/{architecture}/test/evaluation-receipt.json')
    protocol = read(f'docs/selftrained/results/public-raw/{architecture}/freeze/frozen.json')
    assert protocol['safe_weights_sha256'] == metrics['safe_weights_sha256'] == receipt['checkpoint_sha256']
    total = sum(t['count'] for t in metrics['per_task_final_reply'].values())
    assert total == metrics['count'] == receipt['completed_count'] == record_counts['test'] == 3734
    # A concrete failure suffices to establish 'not all original criteria passed'.
    fail = metrics['per_task_final_reply']['tool_call']['tool_roundtrip']
    assert fail['rate'] < metrics['thresholds']['tool_roundtrip']
    v2[architecture] = {'parameters_recounted_without_forward': parameters, 'completed_steps': train['steps'], 'train_records': train['train_records'], 'validation_records': train['validation_records'], 'seed': train['seed'], 'test_used_for_selection': train['test_used_for_selection'], 'full_config': config, 'selected_checkpoint_sha256': metrics['selected_checkpoint_sha256'], 'safe_weights_sha256': metrics['safe_weights_sha256'], 'test_count_recounted': total, 'resumed_from': receipt['resumed_from'], 'freeze_created_at': protocol['created_at'], 'failed_tool_roundtrip': fail, 'required_tool_roundtrip': metrics['thresholds']['tool_roundtrip']}

deployment = read('docs/course-experiments/results/capstone_deployment.json')['results']['stages']
student = read('docs/course-experiments/results/capstone_student.json')['results']['evaluations']
historical = {'joint': {k: deployment['joint'][k] for k in ['count', 'end_to_end_correct']}, 'dpo': {k: deployment['dpo'][k] for k in ['count', 'end_to_end_correct']}, 'student_ce': {k: student['test_ce'][k] for k in ['count', 'end_to_end_correct']}, 'student_kd': {k: student['test_kd'][k] for k in ['count', 'end_to_end_correct']}}
models = read('docs/course-experiments/public-models.json')['models']
historical['formal_experiments'] = len(models)
historical['formal_weight_files'] = sum(f['path'].endswith(('.pt', '.safetensors')) for m in models for f in m['files'])
old_models = read('docs/course-experiments/capstone-public.json')['models']
historical['old_capstone_model_ids'] = [m['id'] for m in old_models]

core_root = Path('/workspace/work/omni-core-completed-40988311100/verified')
recipe = read(core_root / 'frozen/formal-recipes.json')
jobs = recipe['jobs']
job_ids = list(jobs) if isinstance(jobs, dict) else [x.get('id', x.get('job_id')) for x in jobs]
assert len(job_ids) == 10
new_training = []
for job in job_ids:
    result = read(core_root / 'core' / job / 'training/training-result.json')
    assert result['complete'] and result['optimizer_updates_observed']
    new_training.append({'job': job, 'final_update': result.get('final_update', result.get('updates')), 'complete': result['complete'], 'optimizer_updates_observed': result['optimizer_updates_observed'], 'keys': list(result)})
final = Path('/workspace/work/omni-final-completed-41004229262/verified')
selection = read(final / 'frozen/model-selection.json')['selected']
thresholds = read(final / 'frozen/metrics.json')
av_raw = read(final / 'measurements/av/results.json')['models']
av = {}
for name, obj in av_raw.items():
    rows = obj['records']
    n = len(rows)
    correct = sum(r['answers'] == r['gold'] for r in rows)
    swap_correct = sum(r['waveform_swap_answers'] == r['waveform_swap_gold'] for r in rows)
    assert correct == obj['primary']['correct'] and n == obj['primary']['families']
    assert swap_correct == obj['same_metadata_waveform_swap']['paired']['correct']
    av[name] = {'families': n, 'correct_recounted': correct, 'waveform_swap_pair_correct_recounted': swap_correct, 'waveform_swap_failures': n - swap_correct, 'panel_correct': {str(panel): sum(r['answers'] == r['gold'] for r in rows if r['panel'] == panel) for panel in range(4)}, 'remove_image_correct': sum(r['remove_image_answer'] == 2 for r in rows), 'remove_audio_correct': sum(r['remove_audio_answer'] == 2 for r in rows)}
selected_av = av[selection['av']]
assert selected_av['correct_recounted'] >= thresholds['AV']['paired_minimum_correct']
assert all(c / 50 >= thresholds['AV']['panel_minimum_accuracy'] for c in selected_av['panel_correct'].values())
assert selected_av['remove_image_correct'] / 200 >= thresholds['AV']['remove_image_minimum_accuracy']
assert selected_av['remove_audio_correct'] / 200 >= thresholds['AV']['remove_audio_minimum_accuracy']
sound_raw = read(final / 'measurements/sound/results.json')
sound_rows = sound_raw['codec_and_gold_program']['rows']
chain_rows = sound_raw['actual_A_chain']['rows']
sound = {'families': len(sound_rows), 'gold_program_correct_recounted': sum(r['generation']['program_success'] for r in sound_rows), 'legal_recounted': sum(r['generation']['legal'] for r in sound_rows), 'snr_mean_recounted': sum(r['active_reference_snr_db'] for r in sound_rows) / len(sound_rows), 'actual_chain_families': len(chain_rows), 'actual_chain_correct_recounted': sum(r['chain_correct'] for r in chain_rows), 'distinct_sound_programs_in_chain': sound_raw['actual_A_chain']['distinct_semantic_sound_programs']}
assert sound['gold_program_correct_recounted'] / sound['families'] >= thresholds['sound']['gold_program_minimum_accuracy']
assert sound['snr_mean_recounted'] >= thresholds['sound']['active_reference_snr_minimum_db']
assert sound['legal_recounted'] / sound['families'] >= thresholds['sound']['legal_output_minimum_accuracy']
assert sound['actual_chain_correct_recounted'] / sound['actual_chain_families'] >= thresholds['sound']['actual_AV_chain_minimum_accuracy']
choice_raw = read(final / 'measurements/choice/results.json')['primary']
choice_rows = choice_raw['rows']
choice = {kind: sum(r['tasks'][kind]['answer'] == r['tasks'][kind]['gold'] for r in choice_rows) for kind in ['choice', 'info', 'rubric']}
assert all(choice[kind] == choice_raw['types'][kind]['primary']['correct'] for kind in choice)
choice['families'] = len(choice_rows)
choice['accepted_recounted'] = sum(r['accepted'] for r in choice_raw['policy']['rows'])
assert choice['choice'] < thresholds['C']['choice_minimum_correct']
assert choice['rubric'] < thresholds['C']['rubric_minimum_correct']

kev_raw_path = Path('/workspace/work/omni-kev-recovery-completed-41011677439/verified/kev-raw-predictions.jsonl')
kev_bytes = kev_raw_path.read_bytes()
kev_rows = [json.loads(line) for line in kev_bytes.splitlines()]
inputs.append({'path': str(kev_raw_path), 'bytes': len(kev_bytes), 'sha256': hashlib.sha256(kev_bytes).hexdigest()})
kev = {'raw_rows': len(kev_rows), 'row_keys': list(kev_rows[0]), 'not_inferred_locally': True}
qwen = read('/workspace/work/omni-qwen25-original-attempt-41015365689/artifact/research-receipt.json')
qwen_runtime = read('/workspace/work/omni-qwen25-original-attempt-41015365689/artifact/runtime-receipt.json')
qwen_result = {k: qwen[k] for k in ['status','error_type','error','inference_rows','model_load_completed'] if k in qwen}
qwen_result['runtime'] = {k: qwen_runtime[k] for k in ['status','error_type','error','returncode'] if k in qwen_runtime}

from tiny_perceptron.lesson_products.data import service_worlds, split_of, c_rows
duplicates = defaultdict(dict)
for world in service_worlds():
    split = split_of(world['family_id'], 'finite-worlds-v1-coverage-1')
    for row in c_rows(world, split):
        if row['state'] == '卡片進度;申請編號=?' and row['question'] == 'choice':
            duplicates[row['gold']].setdefault(split, {'family': row['family_id'], 'state': row['state'], 'gold': row['gold']})
assert 'train' in duplicates['card'] and 'test' in duplicates['card']
old_products = read('docs/lesson-products/results/final-report-v1.json')
old_training = {}
for kind in ['b', 'c']:
    raw_training = read(f'docs/lesson-products/results/initial-{kind}-execution.json')
    old_training[kind] = {'run_id': raw_training['run_id'], 'gpu_name': raw_training['gpu_name'], 'status': raw_training['status'], 'random_initialization': raw_training['training']['random_initialization'], 'steps_this_attempt': raw_training['training']['steps_this_attempt'], 'last_step': raw_training['training']['last_step'], 'selected_step': raw_training['training']['selected_step'], 'weights_changed': raw_training['training']['weights_changed']}
old_product_summary = {'B_sum_depth3': old_products['b']['3']['sum'], 'C_wording_errors': {k: {s: {'count': v[s]['count'], 'correct': v[s]['correct']} for s in ['before','after']} for k, v in old_products['shifts']['c_wording'].items()}, 'same_observation_across_families': duplicates, 'raw_training': old_training}

# Artificial example of the precise figure mechanics, using CPU logits only.
logits = torch.nn.Parameter(torch.zeros(3, 4))
optimizer = torch.optim.SGD([logits], lr=0.1)
X = torch.tensor([0, 1, 2])
Y = torch.tensor([1, 2, 3])
visibility = torch.ones(3, 3, dtype=torch.bool).tril()
scored = torch.tensor([False, False, True])
loss = F.cross_entropy(logits[scored], Y[scored])
logits.grad = torch.ones_like(logits)  # deliberate stale gradient
optimizer.zero_grad(set_to_none=True)
assert logits.grad is None
loss.backward()
grad = logits.grad.detach().clone()
before = logits.detach().clone()
optimizer.step()
assert grad[:2].count_nonzero() == 0 and (logits.detach() - before).abs().sum() > 0
mechanism = {'X': X.tolist(), 'Y': Y.tolist(), 'visibility': visibility.tolist(), 'scored_positions': scored.tolist(), 'loss': loss.item(), 'gradient': grad.tolist(), 'weight_delta_L1': (logits.detach() - before).abs().sum().item(), 'scope': '12 artificial CPU logits, no trained model forward and no capability conclusion'}

output = {'reviewer_task': '/root/omni_p6_technical_curriculum_final', 'created_at': datetime.now(timezone.utc).isoformat(), 'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'device': 'CPU', 'threads': str(torch.get_num_threads())}, 'source_check': source_check, 'inventory': {'chapters': len([p for p in (ROOT / 'course/chapters').glob('*.md') if p.stem.isdigit()]), 'lesson_sections': len(lesson_ids), 'front_sections': front, 'all_numbered_sections': len(lesson_ids) + sum(front.values()), 'new_sections': {sid: lesson_ids[sid] for sid in new_ids}, 'notebook_sync_result': sync.stdout.strip(), 'preserved_new_notebook_execution': notebooks}, 'data': {'archive_sha256': manifest['package']['sha256'], 'counts': dict(record_counts), 'tasks': tasks, 'member_hashes': data_member_hashes, 'ocr_glyphs': {k: ''.join(sorted(v)) for k, v in ocr_glyphs.items()}, 'ocr_lengths': {k: sorted(v) for k, v in ocr_lengths.items()}, 'ocr_fonts': {k: sorted(v) for k, v in ocr_fonts.items()}, 'voice_intents': {k: sorted(v) for k, v in voice_intents.items()}, 'vision_classes': {k: sorted(v) for k, v in vision_classes.items()}}, 'v2': v2, 'historical': historical, 'new_training_jobs': new_training, 'selected_new_modules': selection, 'new_AV': av, 'new_sound': sound, 'new_C': choice, 'Kev': kev, 'Qwen25': qwen_result, 'old_B_C': old_product_summary, 'synthetic_mechanism': mechanism, 'inputs': inputs}
(OUT / 'raw-audit-output.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'inventory': [len(lesson_ids), len(lesson_ids) + sum(front.values())], 'data': dict(record_counts), 'v2_parameters': {k: v['parameters_recounted_without_forward'] for k,v in v2.items()}, 'historical': historical, 'selected_AV': selected_av, 'sound': sound, 'C': choice, 'Kev_raw_rows': len(kev_rows), 'Qwen25': qwen_result}, ensure_ascii=False))
