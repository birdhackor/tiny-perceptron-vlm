"""Independent bounded R.4 checks. Never train or run a final evaluation.

Raw historical observations are re-aggregated; only the three public C-v3
demonstrations are inferred anew. Large archives/weights stay outside Git.
"""
import ast
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import load as load_tensors

from scripts.course_product import dispatch_request, install_archive
from tiny_perceptron.decision_product.service import DecisionEngine
from tiny_perceptron.lesson_products.inference import auto_solve, rule_depth
from tiny_perceptron.omni_products.choice import Candidate, ChoiceConfig, ChoiceModel, Question, collate_questions

ROOT = Path('/workspace/work/omni-book-integrated-v1')
OUT = Path(__file__).parent
CZIP = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')
OZIP = ROOT / 'outputs/omni-P5-public-raw-results-v1/final-results.zip'
TZIP = Path('/workspace/work/omni-core-completed-40988311100/core-checkpoints.zip')
torch.set_num_threads(2)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text())


def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path.read_bytes())}


def argmax(row):
    return max(range(len(row['logits'])), key=lambda i: -math.inf if row['logits'][i] is None else row['logits'][i])


def independent_gold(task, permission, ready, kind):
    if kind == 'removed':
        return 'none'
    if kind == 'info':
        needed = (permission, ready) if task == 'record' else (permission,) if task == 'play' else (ready,)
        return 'yes' if all(v is not None for v in needed) else 'no'
    if permission is None or ready is None:
        return 'none'
    if kind == 'choice':
        return 'stop' if not permission else 'wait' if not ready else task
    return str(int(permission) + int(ready) if task == 'record' else (2 if permission else int(ready)) if task == 'play' else (2 if ready else int(permission)))


receipt = {
    'created_at': datetime.now(UTC).isoformat(),
    'reviewer_task': '/root/c_v3_technical_coord/numbered_r4',
    'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'CPU',
                    'platform': platform.platform(), 'threads': str(torch.get_num_threads())},
    'command': 'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-numbered-r4/verify.py',
    'scope': 'Fresh CPU verification of three examples, pure Auto control-flow probes, archive identity checks, and retained raw-observation reaggregation. No training, no final NN re-evaluation, no mature-model inference, no publication.',
    'archives': [identity(path) for path in (CZIP, OZIP, TZIP)],
}
source = ROOT / 'docs/course-revision-20261101-c-v3/numbered-source-freeze-v21/sections/R.4.md'
assert digest(source.read_bytes()) == 'a0adc0f0ed9e1d7d2f7bcdf3b0036eb0398f1e477eec0b9f33cf0394a70527df'
release = load('docs/decision-product/v3/public-release.json')
assert identity(CZIP)['sha256'] == release['sha256'] and CZIP.stat().st_size == release['bytes']
metadata = load('docs/decision-product/v3/metadata/acquisition.json')
assert metadata['archive']['sha256'] == release['sha256']
prefix = 'tiny-perceptron-c-v3/'
with zipfile.ZipFile(CZIP) as z:
    read = lambda name: z.read(prefix + name)
    doc = lambda name: json.loads(read(name))
    manifest = doc('MANIFEST.json')
    checked = []
    # Review files are neither read nor used as evidence of correctness.
    for name, expected in manifest['files'].items():
        if any(token in name for token in ('review', 'readability', 'coherence', 'correctness')):
            continue
        raw = read(name)
        assert len(raw) == expected['bytes'] and digest(raw) == expected['sha256'], name
        checked.append(name)
    for row in metadata['files']:
        assert digest(z.read(row['archive_member'])) == row['sha256']
        assert (ROOT / row['path']).read_bytes() == z.read(row['archive_member'])
    for path, expected in release['implementation'].items():
        assert digest((ROOT / path).read_bytes()) == expected
        assert (ROOT / path).read_bytes() == read(path)
    opening = doc('evidence/final-opening.json')
    selection = doc('evidence/selection.json')
    final = doc('evidence/final-results.json')
    rows = doc('evidence/final-observations.json')
    examples = doc('evidence/final-examples.json')
    families = doc('evidence/final-families.json')
    for name in ['observations', 'examples', 'families']:
        assert digest(read('evidence/final-' + name + '.json')) == final[name + '_sha256']
    assert digest(read('evidence/final-results.json')) == release['formal_final_sha256']
    assert digest(read('evidence/selection.json')) == opening['selection_sha256']
    assert digest(read('evidence/completion.json')) == selection['completion_sha256']
    excluded = {key for entry in doc('evidence/registry-before-final.json')['entries'] for key in entry['family_keys']}
    final_keys = {f['key'] for f in families}
    assert len(final_keys) == len(families) == 1500
    assert not final_keys.intersection(excluded)
    assert not final_keys.intersection(selection['families']['calibration'])
    assert not final_keys.intersection(selection['families']['policy'])
    assert len(rows) == len(examples) == 4500
    by_family = defaultdict(set)
    facts_by_family = {family['key']: family['facts'][0] for family in families}
    counts = defaultdict(lambda: [0, 0])
    accepted = errors = 0
    for row, example in zip(rows, examples):
        assert row['family'] == example['family'] and row['kind'] == example['kind']
        assert row['gold_id'] == example['gold']
        assert row['ids'][row['label']] == row['gold_id']
        task_index, permission, ready = facts_by_family[row['family']]
        expected = independent_gold(('record','play','save')[task_index], None if permission == -1 else bool(permission), None if ready == -1 else bool(ready), row['kind'])
        if row['kind'] == 'choice' and expected != 'none':
            target = re.findall(r'(?<![A-Za-z0-9])([A-H])(?![A-Za-z0-9])', example['question'])[0]
            matches = [key for key, description in example['candidates'] if re.findall(r'(?<![A-Za-z0-9])([A-H])(?![A-Za-z0-9])', description) == [target] and description.rsplit(':',1)[-1].rsplit('：',1)[-1].strip() == expected]
            expected = matches[0] if matches else 'none'
        assert expected == row['gold_id']
        by_family[row['family']].add(row['kind'])
        pred = argmax(row)
        counts[row['kind']][0] += pred == row['label']
        counts[row['kind']][1] += 1
        if row['kind'] == 'choice':
            logits = torch.tensor([v if v is not None else -math.inf for v in row['logits']])
            confidence = float((logits / selection['temperatures']['choice']).softmax(-1)[pred])
            accept = row['ids'][pred] != 'none' and confidence >= selection['selected']['threshold'] and row['eligible'][pred]
            accepted += int(accept)
            errors += int(accept and pred != row['label'])
    assert all(kinds == {'choice', 'info', 'rubric'} for kinds in by_family.values())
    assert dict(counts) == {'choice': [1500, 1500], 'info': [1500, 1500], 'rubric': [1500, 1500]}
    assert (accepted, errors) == (1025, 0)
    upper95 = 1.6448536269514722**2 / (accepted + 1.6448536269514722**2)
    assert abs(upper95 - final['metrics']['selective_choice']['risk_one_sided_wilson95_upper']) < 1e-15
    manual = doc('evidence/product-checks.json')
    manual_counts = Counter()
    for row in manual['observations']:
        prediction = row['prediction']
        actual = prediction['selected'] if prediction['selected'] is not None else 'none'
        assert actual == row['expected']
        assert row['expected'] == independent_gold(row['task'], row['permission'], row['ready'], row['kind'])
        manual_counts[row['kind']] += 1
    assert not manual['accepted_errors'] and not manual['invariance_failures']
    completion = doc('evidence/completion.json')
    parent = doc('evidence/lineage-parent.json')['receipts']
    assert completion['completed_updates'] == 1500 and completion['process_exit_code'] == 0
    assert parent['completion.json']['completed_updates'] == 4000
    assert parent['plan.json']['training'].startswith('fresh random initialization')
    assert doc('evidence/lineage.json')['parent_weights_sha256'] == parent['completion.json']['artifacts']['best.safetensors']
    receipt['c_v3_retained_measurement'] = {
        'metadata_members_byte_identical': len(metadata['files']), 'archive_members_checked_without_reviews': len(checked),
        'source_files_byte_identical': len(release['implementation']),
        'weights_sha256': release['bundle']['weights.safetensors'],
        'final_families': len(families), 'observations': len(rows), 'per_kind_correct_total': dict(counts),
        'independently_checked_final_labels': len(rows), 'independently_checked_manual_labels': len(manual['observations']),
        'device_counts': dict(Counter(len(f['facts']) for f in families)),
        'previous_exposed_families': len(excluded), 'final_overlap_previous_exposed': 0,
        'accepted': accepted, 'accepted_errors': errors, 'overall_coverage': accepted / 1500,
        'answerable_choice_coverage': 1.0, 'one_sided_wilson95_upper': upper95,
        'temperatures': selection['temperatures'], 'threshold': selection['selected']['threshold'],
        'manual_observation_counts': dict(manual_counts),
        'manual_local_states': len({(r['task'], r['permission'], r['ready']) for r in manual['observations'] if 'task' in r}),
        'training_lineage': {'own_parent_updates': 4000, 'hardening_updates': 1500, 'external_pretrained': False},
        'identity_note': 'Raw 2026-11-01 final/manual observations retained under the fixed public ZIP; only arithmetic/identity rechecked, not final model predictions regenerated.',
    }

with tempfile.TemporaryDirectory(prefix='numbered-r4-inference-') as temporary:
    bundle = Path(temporary) / 'bundle'
    install_archive(CZIP, bundle, release)
    cases = []
    for filename in ['record-ready.json', 'record-unknown.json', 'answer-absent.json']:
        command = [sys.executable, '-m', 'scripts.course_product', 'judge', '--bundle', str(bundle), '--request', 'examples/decision_product/' + filename]
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=45, check=True)
        result = json.loads(proc.stdout)
        assert result['product'] == 'C-v3' and result['dispatch'] == 'typed-devices'
        if filename == 'record-ready.json':
            assert result['results']['choice']['status'] == 'accepted' and result['results']['choice']['selected'] == 'record'
            assert result['results']['info']['selected'] == 'yes' and result['results']['rubric']['selected'] == '2'
        else:
            assert result['results']['choice']['status'] == 'abstained' and result['results']['choice']['selected'] is None
        if filename == 'record-unknown.json':
            assert result['results']['info']['selected'] == 'no' and result['results']['rubric']['status'] == 'abstained'
        cases.append({'request': filename, 'command': command, 'returncode': proc.returncode, 'result': result, 'stderr': proc.stderr})
    receipt['fresh_cpu_examples'] = cases
    engine = DecisionEngine(bundle)
    request = load('examples/decision_product/record-ready.json')
    calls = []
    hook = engine._model.state_encoder.register_forward_hook(lambda module, args, result: calls.append({'input': list(args[0].shape), 'output': list(result.shape)}))
    first = dispatch_request(engine, request)
    second = dispatch_request(engine, request)
    hook.remove()
    assert first == second and len(calls) == 1
    rejected = []
    for bad in [{'state': 'old text Auto'}, {**request, 'gold': 'record'}]:
        try:
            dispatch_request(engine, bad)
        except ValueError as error:
            rejected.append(str(error))
        else:
            raise AssertionError('unsupported schema accepted')
    receipt['fresh_interface_and_cache_probe'] = {'encoder_calls_for_two_all_kind_requests': calls, 'predictions_equal': first == second, 'schema_rejections': rejected, 'scope': 'The same fixed four-device example, three questions per request, two requests. One state-only encoding reused; no general accuracy inference.'}

omni = load('docs/omni-products/v1/results-summary.json')
assert identity(OZIP)['sha256'] == omni['final']['raw_archive']['sha256']
with zipfile.ZipFile(OZIP) as z:
    records = {}
    for name, expected in omni['final']['raw_results'].items():
        raw = z.read(expected['archive_path'])
        assert digest(raw) == expected['sha256']
        records[name] = json.loads(raw)
    av_counts = {}
    for job, data in records['av']['models'].items():
        rr = data['records']
        primary = sum(r['answers'] == r['gold'] for r in rr)
        swapped = sum(r['waveform_swap_answers'] == r['waveform_swap_gold'] for r in rr)
        assert primary == data['primary']['correct'] and len(rr) == data['primary']['families'] == 200
        assert swapped == data['same_metadata_waveform_swap']['paired']['correct']
        assert sum(r['remove_image_answer'] == 2 for r in rr) == data['remove_image']['correct']
        assert sum(r['remove_audio_answer'] == 2 for r in rr) == data['remove_audio']['correct']
        av_counts[job] = {'primary': primary, 'waveform_swap': swapped, 'families': len(rr)}
    sound = records['sound']['codec_and_gold_program']
    ss = sound['rows']
    gold_correct = sum(r['generation']['program_success'] for r in ss)
    snr = sum(r['active_reference_snr_db'] for r in ss) / len(ss)
    assert gold_correct == sound['gold_program']['correct'] == 200
    assert abs(snr - sound['mean_active_reference_snr_db']) < 1e-11
    chain_rows = records['sound']['actual_A_chain']['rows']
    chain_correct = sum(r['chain_correct'] for r in chain_rows)
    assert chain_correct == records['sound']['actual_A_chain']['primary']['correct'] == len(chain_rows) == 200
    choice = records['choice']['primary']
    cc = {kind: sum(r['tasks'][kind]['answer'] == r['tasks'][kind]['gold'] for r in choice['rows']) for kind in ['choice','info','rubric']}
    assert cc == {'choice': 39, 'info': 55, 'rubric': 35} and len(choice['rows']) == 60
    pp = choice['policy']['rows']
    accepts = sum(r['accepted'] for r in pp)
    wrong = sum(r['error'] for r in pp)
    assert (accepts, wrong) == (53, 19)
    retention = records['integration']
    assert len(retention['old_A_retention']['rows']) == 258
    assert all(row['old_correct'] == row['wrapper_correct'] for row in retention['old_A_retention']['rows'])
    b = retention['B_retention']
    assert b['population_families'] == 84 and b['old_correct'] == b['wrapper_correct'] == 60
    receipt['omni_retained_measurement'] = {
        'training_run_id': omni['training']['run_id'], 'final_run_id': omni['final']['run_id'],
        'selected_av': load('docs/omni-products/v1/model-selection.json')['selected']['av'],
        'av_recount': av_counts, 'sound_families': len(ss), 'sound_gold_correct': gold_correct,
        'codec_mean_active_reference_snr_db': snr, 'c_v2_families': len(choice['rows']),
        'sound_actual_av_chain_correct': chain_correct, 'sound_actual_av_chain_families': len(chain_rows),
        'c_v2_correct': cc, 'c_v2_accepted': accepts, 'c_v2_accepted_errors': wrong,
        'c_v2_accepted_error_risk': wrong / accepts,
        'legacy_A_observations': 258, 'legacy_A_parent_families': 180,
        'legacy_B_population': b['population_families'], 'legacy_B_correct': b['wrapper_correct'],
        'legacy_B_depth': b['frozen_depth'],
        'legacy_binding': retention['legacy_binding'],
        'identity_note': '2026-10-30 original fixed raw results retained; counts recomputed from records, no historical training or NN inference rerun. Old A/B failures retained as recorded.',
    }

selection = load('docs/omni-products/v1/model-selection.json')
jobs = load('docs/omni-products/v1/formal-recipes.json')['job_order']
assert len(jobs) == len(set(jobs)) == 10
with zipfile.ZipFile(TZIP) as z:
    formal_jobs = []
    for name in jobs:
        raw = z.read('core/' + name + '/job-result.json')
        assert digest(raw) == selection['all_formal_jobs'][name]['job_result_sha256']
        job_result = json.loads(raw)
        training_raw = z.read('core/' + name + '/training/training-result.json')
        assert digest(training_raw) == job_result['training_result_sha256']
        training = json.loads(training_raw)
        assert job_result['success'] and training['complete']
        assert training['updates'] == load('docs/omni-products/v1/formal-recipes.json')['jobs'][name]['training']['updates']
        assert training['initial_parameter_sha256'] != training['latest_parameter_sha256']
        weights = z.read('core/' + name + '/training/weights.safetensors')
        assert digest(weights) == selection['all_formal_jobs'][name]['files']['weights.safetensors']['sha256']
        formal_jobs.append({'job': name, 'job_result_sha256': digest(raw), 'weights_sha256': digest(weights), 'complete': training['complete'], 'updates': training['updates'], 'initial_parameter_sha256': training['initial_parameter_sha256']})
        if name == 'c-augmented':
            c2_config, c2_weights = training['config']['model'], weights
    receipt['formal_training_retention'] = {'completed_jobs': len(formal_jobs), 'jobs': formal_jobs, 'note': 'Original ten result/weight identities checked; no training repeated.'}

c2 = ChoiceModel(ChoiceConfig(**c2_config)).eval().requires_grad_(False)
c2.load_state_dict(load_tensors(c2_weights))
question = Question('A: task=record; permission=yes; ready=yes.', 'choice', "Choose A's action.", (Candidate('record','Device A: record'), Candidate('play','Device A: play')))
with torch.inference_mode():
    batch = collate_questions([question])
    shared = c2(batch)['logits']
    recomputed = c2(batch, recompute=True)['logits']
    memory = c2.state_memory(question.state)
    cached = c2(batch, memory=memory)['logits']
    torch.testing.assert_close(shared, recomputed, atol=1e-5, rtol=0)
    torch.testing.assert_close(shared, cached, atol=1e-5, rtol=0)
    reverse = Question(question.state, question.kind, question.text, tuple(reversed(question.candidates)))
    permuted = c2(collate_questions([reverse]))['logits'][:, [1,0,2]]
    torch.testing.assert_close(shared, permuted, atol=1e-5, rtol=0)
receipt['fresh_c_v2_shared_mechanism_probe'] = {'weights_sha256':digest(c2_weights), 'input_state':question.state, 'candidates':[c.id for c in question.candidates], 'max_abs_shared_vs_recompute':float((shared-recomputed).abs().max()), 'max_abs_shared_vs_cache':float((shared-cached).abs().max()), 'max_abs_candidate_reorder':float((shared-permuted).abs().max()), 'atol':1e-5, 'rtol':0.0, 'scope':'One bounded fresh mechanism probe on the retained selected C-v2 weights, no training/final measurement or quality score.'}

class SolverProbe:
    auto_policy = {'version': 1, 'low_max_events': 0, 'medium_max_events': 4, 'sum_depth': 1}
    def __init__(self): self.calls = []
    def synchronize(self): pass
    def solve(self, state, loops): self.calls.append(loops); return {'loops': loops, 'answer': 'probe'}


class JudgeProbe:
    def __init__(self, task, info): self.task, self.info = task, info
    def synchronize(self): pass
    def judge(self, state, kind): return {'selected': self.task if kind == 'choice' else self.info, 'seconds': 0.0}


probes = []
for task, info, expected in [('none','yes','unsupported'),('boxes','no','clarification_needed'),('boxes','yes','answered')]:
    solver = SolverProbe()
    result = auto_solve(solver, JudgeProbe(task,info), '兩盒追蹤;A=2;B=6;事件=交換;查詢=B')
    assert result['status'] == expected
    assert solver.calls == ([3] if expected == 'answered' else [])
    probes.append({'task': task, 'info': info, 'status': result['status'], 'solver_calls': solver.calls})
receipt['auto_control_flow'] = {'probes': probes, 'rule_depths_events1through6': [rule_depth(SolverProbe.auto_policy,'boxes',i) for i in range(1,7)], 'scope': 'Pure service-control probes with explicit fake model selections; no accuracy claim.'}

source_paths = [
    'LICENSE', 'assets/training/README.md', 'pyproject.toml', 'tiny_perceptron/natural_assistant.py',
    'tiny_perceptron/selftrained/model.py', 'tiny_perceptron/lesson_products/model.py',
    'scripts/selftrained/train.py', 'docs/selftrained/v2-manifest.json',
    'tiny_perceptron/lesson_products/train.py', 'tiny_perceptron/lesson_products/inference.py',
    'docs/lesson-products/results/initial-b-execution.json', 'docs/lesson-products/results/initial-c-execution.json',
    'docs/lesson-products/results/b-artifact-manifest.json', 'docs/lesson-products/results/c-artifact-manifest.json',
    'docs/omni-products/v1/results-summary.json', 'docs/omni-products/v1/formal-recipes.json',
    'docs/omni-products/v1/model-selection.json', 'docs/omni-products/v1/metrics.json',
    'docs/decision-product/README.md', 'docs/decision-product/v3/public-release.json',
    'docs/decision-product/v3/qualification-summary.json', 'scripts/course_product.py',
    'tiny_perceptron/omni_products/sound.py', 'tiny_perceptron/omni_products/choice.py', 'course/first-steps.md',
    *['docs/decision-product/v3/metadata/' + name for name in ['acquisition.json','plan.json','completion.json','lineage.json','final-opening.json','selection.json']],
    *['tiny_perceptron/decision_product/' + name for name in ['cli.py','service.py','runtime.py','indexed_model.py','record_model.py','model.py','data.py','hardening_experiment.py','indexed_experiment.py','qualification.py']],
]
receipt['repository_source_identities'] = {p: digest((ROOT / p).read_bytes()) for p in source_paths}
assert (ROOT/'LICENSE').read_text().startswith('MIT License')
assert 'license = "MIT"' in (ROOT/'pyproject.toml').read_text()
assert '資料依各自授權，不能統一當成此repo的MIT內容' in (ROOT/'assets/training/README.md').read_text()
receipt['license_declarations'] = {'project':'MIT', 'third_party_materials':'source-specific licenses; training README explicitly separates them from project MIT'}
for p in ['tiny_perceptron/natural_assistant.py','scripts/selftrained/train.py','tiny_perceptron/lesson_products/train.py']:
    ast.parse((ROOT / p).read_text())
receipt['legacy_initialization'] = {
    'A_manifest_initialization': load('docs/selftrained/v2-manifest.json')['initialization'],
    'B_initial': {k:load('docs/lesson-products/results/initial-b-execution.json')['training'][k] for k in ['random_initialization','resume_source','initial_weights_sha256','selected_weights_sha256']},
    'C_initial': {k:load('docs/lesson-products/results/initial-c-execution.json')['training'][k] for k in ['random_initialization','resume_source','initial_weights_sha256','selected_weights_sha256']},
}
receipt['result'] = 'All bounded assertions passed'
(OUT / 'execution.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'result': receipt['result'], 'fresh_cpu_examples': len(receipt['fresh_cpu_examples']), 'retained_formal_jobs': len(jobs), 'final_predictions_regenerated': False}, ensure_ascii=False))
