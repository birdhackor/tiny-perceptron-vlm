"""19.25 source21 independent probes; no training or final inference."""

import contextlib
import hashlib
import io
import itertools
import json
import math
import platform
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from zipfile import ZipFile

import torch

ROOT = Path('/workspace/work/omni-book-integrated-v1')
OUT = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-canonical-19-25'
ARCHIVE = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')
PREFIX = 'tiny-perceptron-c-v3/'
SHARED = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-numbered-r2/execution.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


started = datetime.now(UTC).isoformat()
torch.set_num_threads(1)
freeze_path = ROOT / 'docs/course-revision-20261101-c-v3/source-freeze-v21.json'
assert sha(freeze_path.read_bytes()) == 'd9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea'
freeze = read_json(freeze_path)
entry = next(row for row in freeze['pages'] if row['page_id'] == '19.25')
body = (ROOT / entry['snapshot']).read_text(encoding='utf-8')
live = re.search(r'^## 19\.25 .+?(?=^## |\Z)', (ROOT / 'course/chapters/19.md').read_text(), re.M | re.S).group()
assert body == live and sha(body.encode()) == entry['source_sha256'] == '3a68a27b5e0cccc451a2c7c2c8afcfd49f009e9981b09ce6f450bcdca62bfd2b'
assert entry['figures_sha256'] == {}

# Execute both literal Python fences, without changing them.
fences = re.findall(r'```python\n(.*?)```', body, re.S)
assert len(fences) == 2
namespace, captured = {}, io.StringIO()
with contextlib.redirect_stdout(captured):
    for index, fence in enumerate(fences):
        exec(compile(fence, f'course/chapters/19.md#19.25/python-{index + 1}', 'exec'), namespace)
assert tuple(namespace['memory'].shape) == (1, 12, 8)
assert tuple(namespace['scores'].shape) == (1, 4)
assert all(torch.allclose(a, b, atol=1e-5) for a, b in (
    (namespace['reordered'][:, :3], namespace['scores'][:, namespace['permutation']]),
    (namespace['reordered'][:, -1], namespace['scores'][:, -1]),
    (namespace['moved'], namespace['scores']),
))
policy_fn = namespace['choose_with_policy']
toy_policy = [policy_fn(torch.tensor(values), namespace['ids'], namespace['owners'], 'A', 1.0, 0.99)
              for values in ([8., 0., -2.], [0., 8., -2.])]
assert toy_policy == [('record-main', 'record-main', True), ('record-other', None, False)]
tie_probabilities = torch.tensor([8., 8., -2.]).softmax(-1)
assert float(tie_probabilities.max()) <= 0.5 < 0.99

descriptor = read_json(ROOT / 'docs/decision-product/v3/public-release.json')
assert ARCHIVE.stat().st_size == descriptor['bytes'] == 6168350
assert sha(ARCHIVE.read_bytes()) == descriptor['sha256'] == '552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88'
members, metadata_checks = {}, []
with ZipFile(ARCHIVE) as archive:
    def read(name):
        raw = archive.read(PREFIX + name)
        members[name] = {'bytes': len(raw), 'sha256': sha(raw)}
        return raw

    def load(name):
        return json.loads(read(name))

    acquisition = read_json(ROOT / 'docs/decision-product/v3/metadata/acquisition.json')
    for row in acquisition['files']:
        local = (ROOT / row['path']).read_bytes()
        assert local == archive.read(row['archive_member'])
        assert len(local) == row['bytes'] and sha(local) == row['sha256']
        metadata_checks.append({'path': row['path'], 'sha256': sha(local), 'byte_identical': True})
    opening, selection = load('evidence/final-opening.json'), load('evidence/selection.json')
    final_results = load('evidence/final-results.json')
    families, observations = load('evidence/final-families.json'), load('evidence/final-observations.json')
    examples = load('evidence/final-examples.json')
    manual, legacy = load('evidence/product-checks.json'), load('evidence/legacy-regression.json')
    registry, bundle = load('evidence/registry-before-final.json'), load('bundle/bundle.json')
    policy, vocabulary = load('bundle/policy.json'), load('bundle/tokenizer.json')
    plan, completion, lineage = load('evidence/plan.json'), load('evidence/completion.json'), load('evidence/lineage.json')
    parent = load('evidence/lineage-parent.json')
    training_events = [json.loads(line) for line in read('evidence/training.jsonl').decode().splitlines()]
    initial_digest = sha(read('evidence/initial.safetensors'))
    assert initial_digest == lineage['parent_weights_sha256'] == parent['receipts']['best.json']['weights_sha256']
    assert parent['receipts']['completion.json']['completed_updates'] == 4000
    assert completion['completed_updates'] == 1500 == max(row.get('step', 0) for row in training_events)
    assert plan['families'] == parent['receipts']['plan.json']['families'] == 8000
    assert parent['receipts']['plan.json']['training'].startswith('fresh random initialization;')
    assert opening['selection_sha256'] == sha(read('evidence/selection.json'))
    assert final_results['final_opening_sha256'] == sha(read('evidence/final-opening.json'))
    assert final_results['observations_sha256'] == sha(read('evidence/final-observations.json'))
    assert final_results['examples_sha256'] == sha(read('evidence/final-examples.json'))
    assert final_results['families_sha256'] == sha(read('evidence/final-families.json'))
    assert selection['created_at'] < opening['created_at'] < final_results['created_at']
    for name, digest in opening['implementation'].items():
        assert sha((ROOT / 'tiny_perceptron/decision_product' / name).read_bytes()) == digest
    for name, digest in descriptor['bundle'].items():
        assert sha(read('bundle/' + name)) == digest
    assert len(read('bundle/weights.safetensors')) == 2818540

# Independently parse visible rows and derive labels, then recompute saved scores.
def parsed_state(text):
    result = {}
    values = {'no': 0, 'yes': 1, 'unknown': -1, '否': 0, '是': 1, '未知': -1}
    for row in re.split(r'[。.]', text):
        if not row.strip():
            continue
        name = re.findall(r'(?<![A-Za-z0-9])([A-H])(?![A-Za-z0-9])', row)
        task = re.search(r'(?<![A-Za-z])(record|play|save)(?![A-Za-z])', row).group(1)
        permission = re.search(r'許可(未知|是|否)|permission(?:\s+is\s+|\s+|=)(unknown|yes|no)', row)
        ready = re.search(r'就緒(未知|是|否)|(?:ready|readiness)(?:\s+is\s+|\s+|=)(unknown|yes|no)', row)
        assert len(name) == 1 and name[0] not in result
        result[name[0]] = (('record', 'play', 'save').index(task),
                           values[permission.group(1) or permission.group(2)], values[ready.group(1) or ready.group(2)])
    return result


def expected_value(kind, fact):
    task, permission, ready = fact
    if kind == 'info':
        required = (permission, ready) if task == 0 else (permission,) if task == 1 else (ready,)
        return 'no' if -1 in required else 'yes'
    if -1 in (permission, ready):
        return 'none'
    if kind == 'choice':
        return 'stop' if permission == 0 else 'wait' if ready == 0 else ('record', 'play', 'save')[task]
    return str(permission + ready if task == 0 else (2 if permission else ready) if task == 1 else (2 if ready else permission))


family_by_key = {row['key']: row for row in families}
examples_by_key = {(row['family'], row['kind']): row for row in examples}
assert len(family_by_key) == 1500 and len(observations) == len(examples_by_key) == 4500
counts, panels, family_kinds = defaultdict(lambda: [0, 0]), Counter(), defaultdict(Counter)
accepted = accepted_errors = answerable_accepted = 0
for row in observations:
    example = examples_by_key[row['family'], row['kind']]
    visible = parsed_state(example['state'])
    assert sorted(visible.values()) == sorted(tuple(fact) for fact in family_by_key[row['family']]['facts'])
    target = re.findall(r'(?<![A-Za-z0-9])([A-H])(?![A-Za-z0-9])', example['question'])[0]
    fact = visible[target]
    assert fact == tuple(family_by_key[row['family']]['facts'][0])
    expected = expected_value(row['kind'], fact)
    if row['kind'] == 'choice' and expected != 'none':
        matches = [identifier for identifier, description in example['candidates']
                   if re.findall(r'(?<![A-Za-z0-9])([A-H])(?![A-Za-z0-9])', description) == [target]
                   and description.rsplit(':', 1)[-1].rsplit('：', 1)[-1].strip() == expected]
        expected = matches[0] if matches else 'none'
    ids = [identifier for identifier, _ in example['candidates']] + ['none']
    assert expected == example['gold'] == row['gold_id']
    assert row['ids'] == ids and row['label'] == ids.index(expected)
    logits = torch.tensor([-math.inf if value is None else value for value in row['logits']])
    index = int(logits.argmax())
    counts[row['kind']][0] += index == row['label']
    counts[row['kind']][1] += 1
    family_kinds[row['family']][row['kind']] += 1
    if row['kind'] == 'choice':
        panels[row['panel']] += 1
        eligible = [re.findall(r'(?<![A-Za-z0-9])([A-H])(?![A-Za-z0-9])', description) == [target]
                    for _, description in example['candidates']] + [False]
        assert eligible == row['eligible']
        probabilities = (logits / policy['temperatures']['choice']).softmax(-1)
        selected = int(probabilities.argmax())
        accept = ids[selected] != 'none' and eligible[selected] and float(probabilities[selected]) >= policy['threshold']
        accepted += accept
        accepted_errors += accept and selected != row['label']
        answerable_accepted += accept and expected != 'none'
assert dict(counts) == {kind: [1500, 1500] for kind in ('choice', 'info', 'rubric')}
assert all(value == Counter(choice=1, info=1, rubric=1) for value in family_kinds.values())
assert panels == Counter(complete=1025, missing=362, removed=113)
assert (accepted, accepted_errors, answerable_accepted) == (1025, 0, 1025)

from tiny_perceptron.decision_product.data import family_key, make_families, partition
from tiny_perceptron.decision_product.runtime import create_model, encode_for_model, collate_for_model, candidate_eligibility
from tiny_perceptron.decision_product.model import Tokenizer
from tiny_perceptron.decision_product.service import Device, DeviceSession, DecisionEngine

train = make_families('train', 8000, 826019)
dev = make_families('dev', 600, 826021)
splits = {'train': {row['key'] for row in train}, 'dev': {row['key'] for row in dev},
          'calibration': set(selection['families']['calibration']), 'policy': set(selection['families']['policy']),
          'final': set(family_by_key)}
for name, keys in splits.items():
    assert all(partition(key) == name for key in keys)
assert all(not a & b for a, b in itertools.combinations(splits.values(), 2))
assert len({tuple(fact) for family in train for fact in family['facts']}) == 27
assert all(row['key'] == family_key(row['facts']) == family_key(list(reversed(row['facts']))) for row in families)
excluded = {key for row in registry['entries'] for key in row['family_keys']}
assert not excluded & set(family_by_key)

manual_counts = defaultdict(lambda: [0, 0])
manual_coverage = defaultdict(set)
for row in manual['observations']:
    fact = (('record', 'play', 'save').index(row['task']),
            -1 if row['permission'] is None else int(row['permission']),
            -1 if row['ready'] is None else int(row['ready']))
    expected = 'none' if row['kind'] == 'removed' else expected_value(row['kind'], fact)
    assert row['expected'] == expected
    manual_counts[row['kind']][0] += row['prediction']['model_choice'] == expected
    manual_counts[row['kind']][1] += 1
    if row['kind'] != 'removed':
        manual_coverage[row['kind']].add((fact, row['devices'], row['target_position'], row['language']))
assert dict(manual_counts) == {'choice': [972, 972], 'info': [972, 972], 'rubric': [972, 972], 'removed': [432, 432]}
assert all(len(value) == 27 * (3 + 4 + 5 + 6) * 2 == 972 for value in manual_coverage.values())
legacy_counts = {kind: [sum(row['prediction']['model_choice'] == row['gold'] for row in legacy['observations'] if row['kind'] == kind),
                        sum(row['kind'] == kind for row in legacy['observations'])] for kind in ('choice', 'info', 'rubric')}
assert all(value == [60, 60] for value in legacy_counts.values())

# Small current-code mechanics probes use random parameters, not qualified-model inference.
model = create_model(bundle['config']).eval()
assert sum(value.numel() for value in model.parameters()) == 630443
tokenizer = Tokenizer(vocabulary)
assert tokenizer.encode('😀') == [3, *[4 + value for value in '😀'.encode('utf-8')]]
recorded = []
class Recorder:
    def _predict(self, example, use_cache=True):
        recorded.append(example)
        return example

devices = [Device('主錄音室', 'record', True, True), Device('播放台', 'play', True, False),
           Device('封存NAS', 'save', False, True), Device('備援錄音室', 'record', False, False)]
session = DeviceSession(Recorder(), devices, 'zh')
choice = session.choose('主錄音室')
info = DeviceSession(Recorder(), devices[:3], 'zh').information('播放台')
rubric = session.grade('主錄音室')
encoded = [encode_for_model(example, tokenizer, model) for example in (choice, info, rubric)]
batch = collate_for_model(encoded, model)
with torch.inference_mode():
    memory = model.state_encoder(batch['state'])
    logits = model(batch, memory=memory)
assert batch['state'].dtype == batch['branches'].dtype == torch.int64
assert batch['target_rows'].tolist() == [0, 1, 0]
assert batch['counts'].tolist() == [5, 2, 3]
assert tuple(memory.shape) == (3, 192, 96)
assert torch.isfinite(memory).all() and torch.equal(memory[1, 144:], torch.zeros_like(memory[1, 144:]))
assert torch.isneginf(logits[1, 2:]).all() and torch.isfinite(logits[2, :4]).all()
engine = DecisionEngine._for_development(model, tokenizer, policy['temperatures'], policy['threshold'])
key = engine._memory_key(choice.state)
with torch.no_grad():
    model.embedding.weight[1, 0] += 1
assert key != engine._memory_key(choice.state)
assert candidate_eligibility(choice) == [True] * 5 + [False]
rejections = []
for invalid in (devices[:2], [*devices[:-1], Device('主錄音室', 'record', False, False)],
                [Device('bad', 'paint', True, True), *devices[1:]],
                [Device('bad', 'record', 1, True), *devices[1:]]):
    try:
        DeviceSession(Recorder(), invalid, 'zh')
    except ValueError as error:
        rejections.append(str(error))
    else:
        raise AssertionError('invalid schema accepted')

assert sha(SHARED.read_bytes()) == '375deeebc7d2107171fd38ed80ffbe127823b7865640ca3118a8fa479c6b1959'
shared = read_json(SHARED)
assert shared['archive']['sha256'] == descriptor['sha256']
assert [[row['response']['results'][kind]['selected'] for kind in ('choice', 'info', 'rubric')]
        for row in shared['new_cpu_examples'][:2]] == [['record', 'yes', '2'], [None, 'no', None]]
assert shared['new_cpu_examples'][2]['response']['results']['choice']['selected'] is None
z = NormalDist().inv_cdf(0.95)
wilson_upper = z*z / (accepted + z*z)
assert abs(wilson_upper * 100 - 0.2632605682948971) < 1e-12

receipt = {
    'reviewer_task': '/root/c_v3_technical_coord/canonical_19_25', 'started_at': started,
    'completed_at': datetime.now(UTC).isoformat(),
    'command': 'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-19-25/audit.py',
    'environment': {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'platform': platform.platform(), 'device': 'cpu', 'threads': '1'},
    'scope': 'Exact source21 fences and small random-weight current-code mechanics executed. Saved final/manual logits, labels, metadata and training receipts are retained observations recomputed without new training, final opening or final inference. Three trained CPU public examples reuse the explicitly identified raw shared receipt; no public transport, GPU/MPS or performance claim.',
    'source_identity': {'path': entry['snapshot'], 'sha256': entry['source_sha256'], 'live_exact': True, 'figures': {}},
    'literal_fences': {'count': len(fences), 'stdout': captured.getvalue(), 'memory_shape': [1, 12, 8], 'scores_shape': [1, 4], 'atol': 1e-5, 'toy_policy': toy_policy, 'tie_max_probability': float(tie_probabilities.max()), 'zero_based_B_indices': {'before': [4,5,6,7], 'after': [8,9,10,11]}},
    'archive': {'path': str(ARCHIVE), 'bytes': descriptor['bytes'], 'sha256': descriptor['sha256'], 'members_read': members, 'metadata': metadata_checks, 'implementation_matches_opening': len(opening['implementation']), 'public_transport_executed': False},
    'retained_training': {'parent_updates': 4000, 'own_updates': 1500, 'training_families': 8000, 'seed': plan['seed'], 'batch_size': plan['batch_size'], 'initial_weights_sha256': initial_digest, 'final_saved_training_step': max(row.get('step', 0) for row in training_events)},
    'recomputed': {'final_per_kind': dict(counts), 'choice_panels': dict(panels), 'accepted': accepted, 'accepted_errors': accepted_errors, 'answerable_accepted': answerable_accepted, 'coverage_all': accepted / 1500, 'family_count': len(family_by_key), 'observation_count': len(observations), 'one_per_kind_per_family': True, 'saved_labels_independently_derived': len(observations), 'device_counts': sorted({len(row['facts']) for row in families}), 'split_counts': {name: len(keys) for name, keys in splits.items()}, 'pairwise_split_intersections': 0, 'previous_exposed_overlap': 0, 'train_local_facts': 27, 'manual_counts': dict(manual_counts), 'manual_combinations_per_kind': {kind: len(value) for kind,value in manual_coverage.items()}, 'legacy_per_kind': legacy_counts, 'policy': policy, 'z95_one_sided': z, 'wilson_upper': wilson_upper},
    'current_code_random_probes': {'distinct_parameters': 630443, 'state_dtype': str(batch['state'].dtype), 'state_shape': list(batch['state'].shape), 'memory_shape': list(memory.shape), 'target_rows': batch['target_rows'].tolist(), 'candidate_counts': batch['counts'].tolist(), 'empty_record_zero': True, 'info_none_and_candidate_padding_masked': True, 'rubric_none_finite': True, 'cache_changes_when_shared_embedding_changes': True, 'unknown_unicode_byte_fallback': True, 'schema_rejections': rejections},
    'retained_cpu_examples': {'path': SHARED.relative_to(ROOT).as_posix(), 'sha256': sha(SHARED.read_bytes()), 'json_pointer': '/new_cpu_examples', 'environment': shared['environment'], 'result': 'record-ready: record/yes/2; record-unknown: null/no/null; answer-absent: abstained/null', 'new_execution_by_this_actor': False},
    'derivations': ['3 tasks * 3 permission values * 3 ready values = 27 local facts.', '27 * (3+4+5+6 target positions) * 2 languages = 972 per kind; 12 fully known facts * 18 positions * 2 languages = 432 removed-answer checks.', '1025 / 1500 = 0.6833333333333333; 362 + 113 = 475; 1025 + 475 = 1500.', 'For zero errors among n=1025 independent identical Bernoulli trials, Wilson score inversion n*p^2 <= z^2*p*(1-p), z=Phi^-1(.95), gives p <= z^2/(n+z^2) = 0.002632605682948971; percent=0.2632605682948971. This is an approximate model-dependent bound, not proof of zero true risk.'],
}
with (OUT / 'execution.json').open('x', encoding='utf-8') as handle:
    json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write('\n')
print(json.dumps({'artifact': str(OUT/'execution.json'), 'final_per_kind': dict(counts), 'accepted': accepted, 'accepted_errors': accepted_errors, 'manual': dict(manual_counts), 'fences_executed': len(fences), 'random_parameter_count': 630443}, ensure_ascii=False))
