"""19.23 reviewer: acquire small primary extracts, never run a trained network."""

import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CORE = Path('/workspace/work/omni-core-completed-40988311100')
FINAL = Path('/workspace/work/omni-final-completed-41004229262')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def record(row):
    return {key: row[key] for key in ('ids', 'logits', 'answer', 'gold') if key in row}


def family(row):
    return {
        **{key: row[key] for key in ('family', 'source', 'panel', 'world_supported', 'choice_information_sufficient')},
        'tasks': {kind: record(value) for kind, value in row['tasks'].items()},
        'choice_info': record(row['choice_info']),
    }


acquired = []


def original(base, filename, archive):
    raw = (base / 'verified' / filename).read_bytes()
    with zipfile.ZipFile(base / archive) as zipped:
        assert zipped.read(filename) == raw, filename
    acquired.append({'archive': archive, 'entry': filename, 'bytes': len(raw), 'sha256': sha(raw),
                     'archive_entry_equals_local_verified': True})
    return json.loads(raw)


results = original(FINAL, 'measurements/choice/results.json', 'final-results.zip')
origin = original(FINAL, 'measurements/choice/part-origin.json', 'final-results.zip')
completed = original(FINAL, 'measurements/choice/part-completion.json', 'final-results.zip')
definition = original(FINAL, 'frozen/data-definition.json', 'final-results.zip')
selected = original(FINAL, 'frozen/model-selection.json', 'final-results.zip')
opening = original(FINAL, 'frozen/final-opening.json', 'final-results.zip')
metrics = original(FINAL, 'frozen/metrics.json', 'final-results.zip')
calibration = original(CORE, 'selection-evidence/calibration.json', 'core-checkpoints.zip')
policy = original(CORE, 'selection-evidence/policy-predictions.json', 'core-checkpoints.zip')
temperature = original(CORE, 'selection-evidence/temperature.json', 'core-checkpoints.zip')
chosen_policy = original(CORE, 'selection-evidence/policy.json', 'core-checkpoints.zip')
training = {}
for mode in ('augmented', 'compact'):
    prefix = f'core/c-{mode}/training/'
    training[mode] = {
        'result': original(CORE, prefix + 'training-result.json', 'core-checkpoints.zip'),
        'log': original(CORE, prefix + 'training-log.json', 'core-checkpoints.zip'),
    }
assert completed['results_sha256'] == sha((FINAL / 'verified/measurements/choice/results.json').read_bytes())
assert origin['model_selection_sha256'] == sha((FINAL / 'verified/frozen/model-selection.json').read_bytes())
assert opening['model_selection_sha256'] == origin['model_selection_sha256']
assert selected['created_at'] < opening['created_at'] < origin['created_at'] < completed['created_at']

snapshot_revision = 'ee799d24fd1ea26108af709ec41e49db63d58e31'
precommit_bytes = subprocess.check_output(['git', 'show', snapshot_revision + ':docs/omni-products/v1/model-selection.json'], cwd=ROOT)
assert precommit_bytes == (FINAL / 'verified/frozen/model-selection.json').read_bytes()
precommit_log = subprocess.check_output(['git', 'log', '-1', '--format=%H%n%aI%n%cI', snapshot_revision,
                                       '--', 'docs/omni-products/v1/model-selection.json'], cwd=ROOT).decode().strip()
assert datetime.fromisoformat(precommit_log.splitlines()[2]) < datetime.fromisoformat(opening['created_at'])

write('fixed-inputs.json', {
    'primary': [family(row) for row in results['primary']['rows']],
    'compact': [family(row) for row in results['data_control_raw_only']['rows']],
    'diagnostic': [{key: row[key] for key in ('family', 'total_choices_including_none', 'gold', 'ids', 'logits', 'answer')}
                   for row in results['candidate_count_secondary']['rows']],
    'calibration': [family(row) for row in calibration],
    'policy': [family(row) for row in policy],
    'development_invariants': results['development_invariants'],
    'temperature': selected['temperature'], 'threshold': selected['threshold'],
    'first_final_input': definition['C']['panels']['final'][0],
    'original_final_labels': [
        {'family': row['family'],
         'golds': {kind: q['gold'] for kind, q in row['questions'].items()},
         'choice_ids': [c['id'] for c in row['questions']['choice']['candidates']]}
        for row in definition['C']['panels']['final']
    ],
    'original_frozen_C_family_counts': {key: len(value) for key, value in definition['C']['families'].items()},
    'original_C_metrics': metrics.get('C'),
    'training': training,
    'recorded_selection': {'created_at': selected['created_at'], 'selected': selected['selected'],
                           'temperature': temperature['temperature'], 'policy_feasible': chosen_policy['feasible'],
                           'threshold': chosen_policy['threshold'], 'final_opened': selected['final_opened']},
    'recorded_opening_created_at': opening['created_at'],
    'choice_part_origin': {key: origin[key] for key in ('created_at', 'final_opened', 'model_selection_sha256',
                                                       'final_opening_sha256', 'data_definition_sha256')},
    'choice_part_completion': completed,
})

torch_commit = '5c4886908584029761b579af026dcfb627c84070'
cache = Path('/workspace/work/reference/omni-P5-19-23-independent')
third_party = []
for basename, url, ranges in [
    ('model.py', 'https://raw.githubusercontent.com/jaredpalmer/kev/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py', [(205, 217), (407, 411)]),
    ('metrics.py', 'https://raw.githubusercontent.com/jaredpalmer/kev/2074cc44256200bb37540103fa4572901faa10f3/kev/metrics.py', [(273, 281), (290, 295)]),
    ('torch-modules-activation.py', f'https://raw.githubusercontent.com/pytorch/pytorch/{torch_commit}/torch/nn/modules/activation.py', [(1273, 1294)]),
    ('torch-modules-transformer.py', f'https://raw.githubusercontent.com/pytorch/pytorch/{torch_commit}/torch/nn/modules/transformer.py', [(946, 957)]),
    ('torch-functional.py', f'https://raw.githubusercontent.com/pytorch/pytorch/{torch_commit}/torch/nn/functional.py', [(2182, 2190), (2300, 2305), (3489, 3501)]),
]:
    raw = (cache / basename).read_bytes()
    lines = raw.decode().splitlines()
    third_party.append({'url': url, 'accessed_on': '2026-10-30', 'bytes': len(raw), 'sha256': sha(raw),
                        'full_text_outside_git': str(cache / basename),
                        'small_supporting_excerpts': [{'lines': f'{start}-{end}', 'text': '\n'.join(lines[start-1:end])}
                                                     for start, end in ranges]})

archives = []
for base, filename, asset_id, expected in [
    (CORE, 'core-checkpoints.zip', 690797670, '107c418e69f917294c7e2fd1a17b820285e8712d5ea4e1f1d62332285de17aee'),
    (FINAL, 'final-results.zip', 690952957, '50b11a952e9ceb57eafa00891b710f44157a17adb742b69abad61655365c8de3'),
]:
    raw = (base / filename).read_bytes()
    assert sha(raw) == expected
    archives.append({'asset_api_url': f'https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/releases/assets/{asset_id}',
                     'name': filename, 'bytes': len(raw), 'sha256': sha(raw)})

write('primary-provenance.json', {
    'reviewer': '/root/omni_p5_technical_19_23', 'created_at': datetime.now(timezone.utc).isoformat(),
    'scope': 'Own archive-entry comparisons and small primary extraction; no peer conclusions used; no neural forward/training.',
    'archives': archives, 'entries_actually_compared': acquired,
    'pre_final_git_evidence': {'final_execution_source_revision': snapshot_revision,
                               'model_selection_at_revision_bytes_equal_original_archive': True,
                               'last_selection_commit_hash_author_time_committer_time': precommit_log,
                               'selection_created_at': selected['created_at'], 'final_opening_created_at': opening['created_at']},
    'third_party_originals_actually_read': third_party,
    'raw_packages_not_required_by_portable_recompute': True,
})
print(json.dumps({'acquired_entries': len(acquired), 'archive_hashes_verified': len(archives),
                  'selection_pre_final_committed_bytes_equal': True,
                  'fixed_inputs_bytes': (OUT / 'fixed-inputs.json').stat().st_size,
                  'no_trained_network_executed': True}, ensure_ascii=False))
