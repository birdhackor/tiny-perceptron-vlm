"""Inspect only named raw timing/allocator/provenance pointers. No training or author result notes."""
import hashlib
import json
import statistics
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
path = OUT/'inputs/efficiency-original.json'
v = json.loads(path.read_text())
pointers, selected, checks = [], {}, []

def get(pointer):
    x = v
    for k in pointer.strip('/').split('/'):
        x = x[k.replace('~1', '/').replace('~0', '~')]
    pointers.append(pointer)
    selected[pointer] = x
    return x

for pointer in ['/device','/torch_version','/python_version','/gpu','/revision','/results/runtime']:
    get(pointer)
groups = ['/results/models/'+m+'/cache/'+r for m in ['mha','gqa']
          for r in ['prefill','full_recompute_decode','cached_decode']]
groups += ['/results/manual_vs_sdpa/timings/'+r for r in ['manual','sdpa']]
groups += ['/results/compile/'+r for r in ['eager','compiled']]
for pointer in groups:
    x = get(pointer)
    assert set(x) == {'median_seconds','samples_seconds','warmup_calls','measured_calls','synchronized'}
    assert len(x['samples_seconds']) == x['measured_calls'] == 9
    assert x['warmup_calls'] == 3 and x['synchronized']
    median = statistics.median(x['samples_seconds'])
    assert median == x['median_seconds']
    checks.append(dict(pointer=pointer, sample_count=9, warmup=3,
                       recomputed_median_seconds=median, median_matches=True))
training_groups = ['/results/models/'+m+'/training' for m in ['mha','gqa']]
training_groups += ['/results/update_variants/'+m+'/training'
                    for m in ['ordinary','accumulated','activation_checkpoint','sdpa']]
for pointer in training_groups:
    for field in ['steps','optimizer_updates','skipped_updates','warm_step_median_seconds',
                  'memory_allocated_before_bytes','peak_memory_allocated_bytes',
                  'peak_additional_allocated_bytes']:
        get(pointer+'/'+field)
    assert selected[pointer+'/steps'] > 3
    assert selected[pointer+'/steps'] == selected[pointer+'/optimizer_updates']
    assert selected[pointer+'/skipped_updates'] == 0
    assert (selected[pointer+'/peak_memory_allocated_bytes']
            - selected[pointer+'/memory_allocated_before_bytes']
            == selected[pointer+'/peak_additional_allocated_bytes'])
for code in ['tiny_perceptron/model.py','tiny_perceptron/attention.py','tiny_perceptron/modern.py',
             'scripts/course_experiments/architecture.py']:
    recorded = get('/code_sha256/'+code.replace('/', '~1'))
    actual = OUT/'inputs/architecture-measured-version.py' if code.endswith('architecture.py') else ROOT/code
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == recorded
extra_reports = []
for name in ['precision', 'flash_probe']:
    original = ROOT/'docs/course-experiments/results'/(name+'.json')
    snapshot = OUT/'inputs'/(name+'-original.json')
    shutil.copyfile(original, snapshot)
    extra = json.loads(snapshot.read_text())
    extra_selected, extra_checks = {}, []
    def extra_get(pointer):
        x = extra
        for k in pointer.strip('/').split('/'):
            x = x[k.replace('~1','/').replace('~0','~')]
        extra_selected[pointer] = x
        return x
    for pointer in ['/device','/torch_version','/python_version','/gpu','/revision']:
        extra_get(pointer)
    code_sha = extra_get('/code_sha256/scripts~1course_experiments~1architecture.py')
    measured_code = OUT/'inputs'/('architecture-measured-version.py' if name=='precision' else 'architecture.py')
    assert hashlib.sha256(measured_code.read_bytes()).hexdigest() == code_sha
    routes = ['/results/variants/'+d+'/inference' for d in ['fp32','bf16','fp16']] if name=='precision' else [
        '/results/routes/'+d+'/measurements/'+mode+'/'+route
        for d in ['fp16','bf16'] for mode in ['forward','forward_backward']
        for route in ['manual','forced_flash']]
    for pointer in routes:
        fields = ['samples_seconds','median_seconds','warmup_calls','measured_calls','synchronized']
        values = {f:extra_get(pointer+'/'+f) for f in fields}
        assert len(values['samples_seconds']) == values['measured_calls'] == 9
        assert values['warmup_calls'] == 3 and values['synchronized']
        median = statistics.median(values['samples_seconds'])
        assert median == values['median_seconds']
        extra_checks.append(dict(pointer=pointer, sample_count=9, warmup=3, recomputed_median_seconds=median, median_matches=True))
    if name=='precision':
        for dtype in ['fp32','bf16','fp16']:
            pointer = '/results/variants/'+dtype+'/training'
            for field in ['steps','optimizer_updates','skipped_updates','warm_step_median_seconds']:
                extra_get(pointer+'/'+field)
            assert extra_selected[pointer+'/steps'] > 3
            assert extra_selected[pointer+'/steps'] == extra_selected[pointer+'/optimizer_updates']
            assert extra_selected[pointer+'/skipped_updates'] == 0
    extra_reports.append(dict(original_path=str(original.relative_to(ROOT)), original_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(), read_pointers=list(extra_selected), raw_selected_values=extra_selected, recomputed_timing_checks=extra_checks))
record = dict(original_path='docs/course-experiments/results/efficiency.json',
    original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    read_pointers=pointers, raw_selected_values=selected, recomputed_timing_checks=checks,
    measured_code_version=dict(revision=v['revision'],
        architecture_sha256=v['code_sha256']['scripts/course_experiments/architecture.py'],
        restored_by='git show REVISION:scripts/course_experiments/architecture.py; digest matches raw report'),
    additional_reports=extra_reports, limitations=[
        'Training per-step latencies are not stored. Read immutable measured-version _train formula latencies[3:] and timing boundaries; no missing latency fabricated and no median recomputation claimed for training.',
        'Existing raw GPU measurements inspected only; no training, score reevaluation, or GPU work rerun.',
        'Initial current-code digest assertion failed because architecture.py differs from the raw report. Resolved by reading exact git-revision source with matching recorded SHA, not by substituting current code.'
    ])
(OUT/'raw-evidence-inspection.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(runtime=selected['/results/runtime'],
    timing_groups_checked=len(checks), samples_per_group=9,
    training_steps={k:x for k,x in selected.items() if k.endswith('/steps')},
    training_optimizer_updates={k:x for k,x in selected.items() if k.endswith('/optimizer_updates')},
    code_provenance_matched=True, allocator_arithmetic_matched=True,
    additional_timing_groups_checked=sum(len(r['recomputed_timing_checks']) for r in extra_reports),
    training_latency_limitation=record['limitations'][0]), ensure_ascii=False, indent=2))
