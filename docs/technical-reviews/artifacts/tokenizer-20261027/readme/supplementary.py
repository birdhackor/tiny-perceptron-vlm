from pathlib import Path
import hashlib
import json

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/tokenizer-20261027/readme'
def load(name):
    return json.loads((ROOT / name).read_bytes())
summary = {}
index = load('course/lesson-index.json')
missing = []
withcode = 0
for item in index:
    nb = load(item['notebook'])
    has_code = any(c['cell_type'] == 'code' and not c.get('metadata', {}).get('course_setup') and 'course_figure' not in c.get('metadata', {}) for c in nb['cells'])
    if has_code:
        withcode += 1
        html = (ROOT / 'outputs/tokenizer-20261027/site' / (item['id'] + '.html')).read_text()
        if '<div class="output">' not in html:
            missing.append(item['id'])
summary['site'] = {'authored_code_pages': withcode, 'pages_missing_output_block': missing, 'scope': 'Output presence and checked_notebook exporter contract, not a fresh execution of every lesson.'}
summary['v2_raw_metrics'] = {}
for arch in ['moe', 'dense']:
    name = f'docs/selftrained/results/public-raw/{arch}/test/metrics.json'
    d = load(name)
    final = load('docs/selftrained/results/v2-final-public-results.json')['architectures'][arch]
    digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert digest == final['final_test']['metrics_sha256']
    assert d['per_task_final_reply'] == final['final_test']['per_task_final_reply']
    summary['v2_raw_metrics'][arch] = {'path': name, 'sha256': digest, 'split': d['split'], 'count': d['count'], 'evaluation_complete': d['evaluation_complete'], 'teacher_forcing_used_for_generation': d['teacher_forcing_used_for_generation'], 'safe_weights_sha256': d['safe_weights_sha256'], 'sum_task_counts': sum(v['count'] for v in d['per_task_final_reply'].values()), 'voice_qa_semantic': d['per_task_final_reply']['voice_qa']['semantic'], 'tool_roundtrip': d['per_task_final_reply']['tool_call']['tool_roundtrip']}
summary['training'] = {}
for name in ['docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json', 'docs/selftrained/results/training-raw/dense-weighted/raw/train-receipt.json']:
    d = load(name)
    summary['training'][name] = {k: d[k] for k in ['architecture', 'steps', 'train_records', 'validation_records', 'test_used_for_selection', 'selection', 'total_parameters', 'origin', 'completed_requested_steps', 'inference_exported']}
for name in ['b', 'c']:
    path = f'docs/lesson-products/results/initial-{name}-execution.json'
    d = load(path)
    summary['training'][path] = {'status': d['status'], 'gpu_name': d['gpu_name'], 'training_environment': d['training']['environment'], 'completed_steps': d['training']['last_step'], 'selected_step': d['training']['selected_step'], 'weights_changed': d['training']['weights_changed']}
manifest = load('docs/selftrained/v2-manifest.json')
summary['v2_package'] = {'files': len(manifest['records']) + len(manifest['assets']), 'bytes': manifest['package']['bytes']}
(OUT / 'supplementary-output.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
