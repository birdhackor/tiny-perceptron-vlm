"""Independent bounded audit of existing 20.8 artifacts. No model loading/generation."""
import ast
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path
from types import SimpleNamespace
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INPUT = HERE / 'inputs'
VAL = INPUT / 'outputs/natural-v4/modal-runs/validation-37219466611/natural-natural-v4-validation-37219466611-1/review'
TRAIN = INPUT / 'outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1/review'
TEST = INPUT / 'outputs/natural-v4/modal-runs/evaluate-37221188153/natural-natural-v4-evaluate-37221188153-1/review'
BLIND = INPUT / 'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review'
V4 = INPUT / 'docs/natural-assistant/v4'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def original_functions(path, names, namespace):
    raw = Path(path).read_text()
    tree = ast.parse(raw)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    for n in selected:
        exec(compile(ast.Module(body=[n], type_ignores=[]), str(path), 'exec'), namespace)
    return [{'name': n.name, 'start': n.lineno, 'end': n.end_lineno} for n in selected]

core_ns = {'hashlib': hashlib, 'Path': Path, 'unicodedata': unicodedata, 're': re}
core_locators = original_functions(INPUT / 'tiny_perceptron/natural_assistant.py',
    ['sha256', 'normalized', 'edit_distance', 'contains_fact', 'score_output'], core_ns)
core = SimpleNamespace(**core_ns)
score_ns = {'Counter': Counter, 'Path': Path, 'json': json, 'hashlib': hashlib, 'core': core}
tree = ast.parse((INPUT / 'scripts/score_natural_v4_validation.py').read_text())
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'WEIGHTS', 'DENOMINATOR', 'MANUAL'} for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<original constants>', 'exec'), score_ns)
scoring_locators = original_functions(INPUT / 'scripts/score_natural_v4_validation.py',
    ['require', 'no_duplicate_keys', 'read_json', 'case_id', 'mapping_commitment', 'expected_cases', 'eos_complete', 'score_blind'], score_ns)
manifest = read(V4 / 'manifest.json')
protocol = read(V4 / 'validation-protocol-lower-lr.json')
cases, audio = score_ns['expected_cases'](manifest, protocol)
result = read(VAL / 'result.json')
execution = read(VAL / 'execution.json')
private = read(BLIND / 'private-map.json')
binding = private['artifact_binding']
assert binding['manifest_sha256'] == digest(V4 / 'manifest.json') == result['manifest_sha256']
assert binding['protocol_sha256'] == digest(V4 / 'validation-protocol-lower-lr.json')
assert binding['validation_result_sha256'] == digest(VAL / 'result.json')
assert binding['transcripts_sha256'] == digest(VAL / 'transcripts.json')
assert binding['asr_selection_sha256'] == digest(V4 / 'asr-selection.json')
assert binding['model'] == result['model'] == 'Qwen/Qwen3-VL-2B-Instruct'
assert binding['model_revision'] == result['model_revision'] == '89644892e4d85e24eaac8bacfd4f463576704203'
assert binding['adapters'] == execution['adapters']
assert result['split'] == 'validation' and result['status'] == 'completed'
assert result['max_pixels'] == protocol['generation']['max_pixels']
assert result['min_pixels'] == protocol['generation']['min_pixels']
assert result['max_tokens'] == 2048 and result['seed'] == 42
assert result['asr_model'] == 'openai/whisper-large-v3-turbo'
assert result['asr_revision'] == '41f01f3fe87f28c78e2fbf8b568835947dd65ed9'
assert '--max-new-tokens' in execution['runner_arguments']
assert execution['runner_arguments'][execution['runner_arguments'].index('--max-new-tokens') + 1] == '384'
transcripts = {r['id']: r for r in read(VAL / 'transcripts.json')}
assert set(transcripts) == {r['id'] for r in audio}
file_hashes = {r['path']: r['sha256'] for r in manifest['files']}
for row in audio:
    assert transcripts[row['id']]['reference_transcript'] == row['user']
    assert transcripts[row['id']]['audio_sha256'] == file_hashes[row['audio']]
records, source_pointers, bad = {}, {}, {}
for item in binding['generation_files']:
    variant, name = item['variant'], item['name']
    assert digest(VAL / name) == item['sha256']
    raw = read(VAL / name)
    records[variant] = {}
    bad[variant] = []
    source_pointers[variant] = {}
    for i, r in enumerate(raw):
        key = score_ns['case_id'](r['id'], r['task'])
        assert key in cases and key not in records[variant]
        row = cases[key]['row']
        assert r['variant'] == variant and r['split'] == 'validation'
        assert r['image'] == row.get('image')
        expected_user = transcripts[row['id']]['transcript'] if cases[key]['group'] == 'voice_actual_asr_chat' else row['user']
        assert r['user'] == expected_user
        if cases[key]['group'] == 'voice_actual_asr_chat':
            assert r['reference_user'] == row['user'] and r['transcript'] == expected_user
        records[variant][key] = r
        source_pointers[variant][key] = f'/{i}'
        if not score_ns['eos_complete'](r, 384):
            bad[variant].append({'case_id': key, 'pointer': f'/{i}', 'generated_tokens': r['generated_tokens'],
                'raw_token_count': len(r['generated_token_ids']), 'final_raw_token': r['generated_token_ids'][-1],
                'eos_token_ids': r['eos_token_ids'], 'stop_reason': r['stop_reason'],
                'ended_with_eos': r['ended_with_eos'], 'truncated': r['truncated']})
    assert set(records[variant]) == set(cases) and len(raw) == 132
scores, selection = score_ns['score_blind']((manifest, protocol, cases, records, binding, None), BLIND, BLIND / 'combined/grades.json')
packet = read(BLIND / 'blind-packet.json')
grades = read(BLIND / 'combined/grades.json')['grades']
assert len(packet['cases']) == len(grades) == 303
for c in packet['cases']:
    case = cases[c['case_id']]
    row = case['row']
    raw = records[private['aliases'][c['candidate']]][c['case_id']]
    assert c['prediction'] == raw['prediction'] and c['model_user'] == raw['user'] and c['user'] == row['user']
    assert c['system'] == row.get('system') and c['history'] == row.get('history', [])
    assert c['rubric'] == (row.get('predeclared_semantic_rubric') if c['group']=='text_chat' else row.get('references', {}).get('rubric'))
    assert c['decoder_complete'] == score_ns['eos_complete'](raw, 384)
saved_scores = read(BLIND / 'scored/scores.json')
assert scores['variants'] == saved_scores['variants']
assert selection['selected_variant'] == read(V4 / 'selection.json')['selected_variant'] == 'base'
assert selection['criterion'] == read(V4 / 'selection.json')['criterion']
assert scores['denominators'] == saved_scores['denominators']
for variant in ['adapter-step-001039', 'adapter-step-002077']:
    assert len(bad[variant]) == 3
    assert all(r['generated_tokens'] == r['raw_token_count'] == 384 and r['stop_reason'] == 'max_new_tokens' and r['truncated'] for r in bad[variant])
assert scores['variants']['base']['correct_counts']['photo_summary'] == 16
assert scores['variants']['base']['correct_counts']['photo_fact'] == 31
assert scores['variants']['adapter-step-001039']['correct_counts']['photo_summary'] == 23
assert scores['variants']['adapter-step-001039']['correct_counts']['photo_fact'] == 53
assert scores['variants']['base']['correct_counts']['voice_typed_reference_chat'] == 4
assert scores['variants']['base']['correct_counts']['voice_actual_asr_chat'] == 3
assert scores['variants']['adapter-step-001039']['correct_counts']['voice_typed_reference_chat'] == 2
assert scores['variants']['adapter-step-001039']['correct_counts']['voice_actual_asr_chat'] == 2
# Exercise the original EOS contract on small CPU values, including exact-cap EOS.
eos = score_ns['eos_complete']
fixture = {'generated_token_ids': [3, 9], 'eos_token_ids': [9], 'generated_tokens': 2,
    'ended_with_eos': True, 'stop_reason': 'eos', 'truncated': False, 'completion_unknown': False}
eos_checks = {'at_cap_eos': eos(fixture, 2), 'over_cap': eos(fixture, 1),
    'false_claimed_eos': eos(dict(fixture, generated_token_ids=[3, 8]), 2),
    'raw_count_mismatch': eos(dict(fixture, generated_tokens=1), 2),
    'marked_truncated': eos(dict(fixture, truncated=True), 2)}
assert eos_checks == {'at_cap_eos': True, 'over_cap': False, 'false_claimed_eos': False, 'raw_count_mismatch': False, 'marked_truncated': False}
train = read(TRAIN / 'training.json')
assert train['status'] == 'completed' and train['completed_steps'] == 2077
assert train['checkpoint_steps'] == [1039, 2077] and train['optimizer_only_lora'] is True
assert all('lora_' in n for n in train['optimizer_parameter_names'])
assert train['model_revision'] == result['model_revision'] and train['manifest_sha256'] == result['manifest_sha256']
assert train['frozen_parameter_samples_initial'] == train['frozen_parameter_samples_final']
assert train['changed_adapter_tensor_count'] == 112
assert [r['completed_steps'] for r in train['archived_checkpoints']] == [1039, 2077]
adapter_configs = []
for archive in train['archived_checkpoints']:
    config_path = TRAIN / archive['path'] / 'adapter_config.json'
    config = read(config_path)
    assert digest(config_path) == next(f['sha256'] for f in archive['files'] if f['path']=='adapter_config.json')
    assert config['bias']=='none' and config['lora_bias'] is False and config['modules_to_save'] is None
    assert config['base_model_name_or_path']=='Qwen/Qwen3-VL-2B-Instruct' and config['peft_type']=='LORA'
    adapter_configs.append({'checkpoint':archive['completed_steps'], 'sha256':digest(config_path),
        'bias':config['bias'], 'lora_bias':config['lora_bias'], 'modules_to_save':config['modules_to_save']})
loss_first = sum(r['answer_token_loss_before_update'] for r in train['history'][:100]) / 100
loss_last = sum(r['answer_token_loss_before_update'] for r in train['history'][-100:]) / 100
assert loss_last < loss_first
test = read(TEST / 'result.json')
test_exec = read(TEST / 'execution.json')
closure = read(BLIND / 'pretest-closure.json')
assert test_exec['selection_sha256'] == digest(V4 / 'selection.json') == closure['selection_sha256']
assert test_exec['pretest_selection']['selected_variant'] == 'base'
assert test['split'] == 'test' and test['selected_only'] is True and set(test['variants']) == {'base'}
assert test['model_revision'] == result['model_revision'] and test['asr_revision'] == result['asr_revision']
assert (test['max_pixels'], test['min_pixels'], test['max_tokens']) == (result['max_pixels'], result['min_pixels'], result['max_tokens'])
assert '--selected-only' in test_exec['runner_arguments']
assert test_exec['runner_arguments'][test_exec['runner_arguments'].index('--max-new-tokens')+1] == '384'
times = {}
for stage, run in [('train', '37217452291'), ('validation', '37219466611'), ('evaluate', '37221188153')]:
    gh = read(INPUT / f'docs/natural-assistant/evidence/v4-runtime/{stage}-{run}/github-run.json')
    times[stage] = {k: gh[k] for k in ['created_at', 'run_started_at', 'updated_at', 'head_sha']}
iso = lambda s: datetime.fromisoformat(s.replace('Z', '+00:00'))
assert iso(protocol['declared_utc']) < iso(times['train']['created_at'])
assert iso(times['train']['updated_at']) < iso(times['validation']['created_at'])
assert iso(times['validation']['updated_at']) < iso(closure['created_utc']) < iso(times['evaluate']['created_at'])
# Verify selection existed as exact bytes in the test's historical Git revision.
import subprocess
historic_selection = subprocess.run(['git', 'show', test_exec['revision'] + ':docs/natural-assistant/v4/selection.json'],
    cwd=ROOT, check=True, capture_output=True).stdout
assert hashlib.sha256(historic_selection).hexdigest() == test_exec['selection_sha256']
photo_rows = [c['row'] for c in cases.values() if c['group'] == 'photo_summary']
photo_sharing = Counter(c['row']['image'] for c in cases.values() if c['group'] in {'photo_summary','photo_fact'})
assert len(photo_rows) == 28 and len(photo_sharing) == 28 and set(photo_sharing.values()) == {3}
image_root = ROOT / 'outputs/natural-v4/anonymous-data-download/data'
image_receipt = []
for row in photo_rows:
    original = image_root / row['image']
    assert digest(original) == file_hashes[row['image']]
    retained = HERE / 'source-photos' / Path(row['image']).name
    retained.parent.mkdir(exist_ok=True)
    retained.write_bytes(original.read_bytes())
    assert digest(retained) == digest(original)
    image_receipt.append({'original': str(original.relative_to(ROOT)), 'retained': str(retained.relative_to(ROOT)), 'sha256': digest(retained)})
def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not installed; not used by the bounded AST/JSON audit'

summary = {'environment': {'python': platform.python_version(), 'device': 'CPU; JSON/AST audit only',
        'torch': installed_version('torch'), 'transformers': installed_version('transformers'),
        'peft': installed_version('peft')},
    'original_function_locators': {'core': core_locators, 'scoring': scoring_locators},
    'denominators': scores['denominators'], 'variants': scores['variants'], 'recomputed_selected_variant': selection['selected_variant'],
    'incomplete_raw_leaves': bad, 'raw_case_pointers': source_pointers,
    'eos_fixture_results': eos_checks,
    'adapter_config_controls': adapter_configs,
    'training': {'completed_steps': train['completed_steps'], 'checkpoints': train['checkpoint_steps'],
        'changed_adapter_tensors': train['changed_adapter_tensor_count'], 'frozen_parameter_samples_equal': True,
        'loss_first_100_mean': loss_first, 'loss_last_100_mean': loss_last,
        'scope': 'Existing training history only, not fresh training; sampled frozen tensors are not a whole-weight equality proof'},
    'timing': {'protocol_declared': protocol['declared_utc'], **times, 'selection_closed': closure['created_utc'],
        'selection_sha256_before_test': test_exec['selection_sha256'], 'historical_test_git_selection_verified': True},
    'source_photo_group_counts': {'photos': len(photo_sharing), 'questions_per_photo': 3},
    'source_photo_copy_receipt': image_receipt,
    'scope': 'Existing raw measurements and original scorer; all photo numerator counts retain original blinded semantic grades. No new trained-model scores.'}
(HERE / 'verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ['environment','denominators','variants','incomplete_raw_leaves','eos_fixture_results','training','timing','source_photo_group_counts']}, ensure_ascii=False, indent=2))
