"""R.2 independent audit: navigation, CPU examples and retained measurements only."""

import contextlib
import copy
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import torch

ROOT = Path('/workspace/work/omni-book-integrated-v1')
OUT = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-numbered-r2'
ARCHIVE = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')
PREFIX = 'tiny-perceptron-c-v3/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def section(text, identifier):
    return re.search(r'^## ' + re.escape(identifier) + r' .+?(?=^## |\Z)', text, re.M | re.S).group(0)


def write_new(name, value):
    with (OUT / name).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


started = datetime.now(UTC).isoformat()
torch.set_num_threads(1)
source_path = ROOT / 'docs/course-revision-20261101-c-v3/numbered-source-freeze-v21/sections/R.2.md'
body = source_path.read_text()
assert sha(body.encode()) == '96bed90b23afbe71e853f4638f0243ab0d8550fdec016ea49fbb88b470ef22e3'
assert section((ROOT / 'course/README.md').read_text(), 'R.2') == body

links = []
for label, href in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', body):
    path, _, anchor = href.partition('#')
    target = ((ROOT / 'course' / path) if path else ROOT / 'course/README.md').resolve()
    assert target.is_file(), href
    text = target.read_text()
    heading = None
    if anchor:
        match = re.search(r'^## ' + re.escape(anchor) + r' (.+)$', text, re.M)
        assert match, href
        heading = match.group(0)
    else:
        heading = re.search(r'^# .+$', text, re.M).group(0)
    links.append({'label': label, 'href': href, 'target': target.relative_to(ROOT).as_posix(),
                  'anchor_heading': heading, 'sha256': sha(target.read_bytes())})
linked_heading_inventory = {}
for path in sorted({item['target'] for item in links} | {'course/chapters/02.md', 'course/chapters/03.md', 'course/chapters/04.md'}):
    linked_heading_inventory[path] = re.findall(r'^## .+$', (ROOT / path).read_text(), re.M)

# Execute the actual 1.5 table-building fence, then query it with the two R.2 contexts.
intro = section((ROOT / 'course/chapters/01.md').read_text(), '1.5')
fence = re.search(r'```python\n(.*?)```', intro, re.S).group(1)
namespace = {}
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile(fence, 'course/chapters/01.md#1.5', 'exec'), namespace)
contexts = ['貓看', '狗看']
rows = [namespace['probability'][namespace['chars'].index(context[-1])] for context in contexts]
assert torch.equal(rows[0], rows[1])
assert torch.equal(namespace['counts'][namespace['chars'].index('看')], torch.tensor([1., 1., 2., 1., 1., 1., 2.]))
bigram = {'contexts': contexts, 'last_character': [text[-1] for text in contexts],
          'lookup_indices': [namespace['chars'].index(text[-1]) for text in contexts],
          'characters': namespace['chars'], 'probability_rows': [row.tolist() for row in rows],
          'same_row_exactly': True, 'counts_for_看': [1, 1, 2, 1, 1, 1, 2], 'row_denominator': 9,
          'original_fence_stdout': capture.getvalue(), 'training_or_update_performed': False}

# Check the actual chat data construction, without constructing/updating a model.
from tiny_perceptron.data import ByteTokenizer, pad_batch, render_chat, toy_conversations

chats = toy_conversations()[:2]
rendered = [render_chat(chat) for chat in chats]
x, y, valid = pad_batch(rendered)
tok = ByteTokenizer()
assert x.shape == (2, 10)
assert y[y != -100].tolist() == [56, 2, 57, 2]
assert tok.user_id in x[0] and tok.assistant_id in x[0]
chat = {'inputs': chats, 'X': x.tolist(), 'Y': y.tolist(), 'valid': valid.tolist(),
        'supervised_targets': y[y != -100].tolist(), 'role_ids': {'user': tok.user_id, 'assistant': tok.assistant_id},
        'eos_id': tok.eos_id, 'effective_answer_targets': 4, 'model_or_training_executed': False}

descriptor = json.loads((ROOT / 'docs/decision-product/v3/public-release.json').read_text())
archive_bytes = ARCHIVE.read_bytes()
assert len(archive_bytes) == descriptor['bytes'] == 6168350
assert sha(archive_bytes) == descriptor['sha256'] == '552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88'
member_receipts = {}
with ZipFile(ARCHIVE) as archive:
    def read(name):
        raw = archive.read(PREFIX + name)
        member_receipts[name] = {'bytes': len(raw), 'sha256': sha(raw)}
        return raw

    def load(name):
        return json.loads(read(name))

    acquisition = json.loads((ROOT / 'docs/decision-product/v3/metadata/acquisition.json').read_text())
    for file in acquisition['files']:
        local = (ROOT / file['path']).read_bytes()
        archived = archive.read(file['archive_member'])
        assert local == archived
        assert sha(local) == file['sha256']

    opening = load('evidence/final-opening.json')
    observations = load('evidence/final-observations.json')
    families = load('evidence/final-families.json')
    results = load('evidence/final-results.json')
    manual = load('evidence/product-checks.json')
    legacy = load('evidence/legacy-regression.json')
    registry = load('evidence/registry-before-final.json')
    selection = load('evidence/selection.json')
    policy = load('bundle/policy.json')
    for name in descriptor['bundle']:
        assert sha(read('bundle/' + name)) == descriptor['bundle'][name]
    assert sha(archive.read(PREFIX + 'evidence/selection.json')) == opening['selection_sha256']
    assert sha(archive.read(PREFIX + 'evidence/product-checks.json')) == opening['product_checks_sha256']
    assert sha(archive.read(PREFIX + 'evidence/legacy-regression.json')) == opening['legacy_regression_sha256']
    assert sha(archive.read(PREFIX + 'evidence/registry-before-final.json')) == opening['previous_registry_sha256']
    for name, digest in opening['implementation'].items():
        assert sha((ROOT / 'tiny_perceptron/decision_product' / name).read_bytes()) == digest
    assert sha(archive.read(PREFIX + 'evidence/final-observations.json')) == results['observations_sha256']
    assert sha(archive.read(PREFIX + 'evidence/final-families.json')) == results['families_sha256']

# Independently recompute measured counts from saved logits, not summary pass flags.
grouped = defaultdict(lambda: [0, 0])
accepted = errors = answerable_accepted = 0
for row in observations:
    logits = [-math.inf if value is None else value for value in row['logits']]
    index = max(range(len(logits)), key=logits.__getitem__)
    correct = index == row['label']
    for key in [row['kind'], f"{row['kind']}/source/{row['source']}", f"{row['kind']}/panel/{row['panel']}",
                f"{row['kind']}/gold/{'none' if row['gold_id'] == 'none' else 'answer' if row['kind'] == 'choice' else row['gold_id']}"]:
        grouped[key][0] += correct
        grouped[key][1] += 1
    if row['kind'] == 'choice':
        temperature = policy['temperatures']['choice']
        weights = [math.exp((value - logits[index]) / temperature) for value in logits]
        confidence = 1 / sum(weights)
        accept = row['ids'][index] != 'none' and row['eligible'][index] and confidence >= policy['threshold']
        accepted += accept
        errors += accept and not correct
        answerable_accepted += accept and row['answerable']
measured = {key: {'correct': c, 'observations': n, 'accuracy': c / n} for key, (c, n) in sorted(grouped.items())}
assert all(measured[kind] == {'correct': 1500, 'observations': 1500, 'accuracy': 1.0} for kind in ('choice', 'info', 'rubric'))
assert (accepted, errors, answerable_accepted) == (1025, 0, 1025)
family_keys = {row['key'] for row in families}
observed_keys = {row['family'] for row in observations}
assert len(family_keys) == 1500 and family_keys == observed_keys
assert all(Counter(row['kind'] for row in observations if row['family'] == key) == Counter({'choice': 1, 'info': 1, 'rubric': 1}) for key in family_keys)
excluded = {key for entry in registry['entries'] for key in entry['family_keys']}
assert not family_keys & excluded
assert set(len(row['facts']) for row in families) == {4, 5, 6}
z = 1.6448536269514722
upper = z * z / (accepted + z * z)  # Wilson one-sided upper for zero errors.
assert abs(upper - results['metrics']['selective_choice']['risk_one_sided_wilson95_upper']) < 1e-14

def rule(row):
    task, permission, ready = row['task'], row['permission'], row['ready']
    if row['kind'] == 'removed':
        return 'none'
    if row['kind'] == 'choice':
        return 'none' if permission is None or ready is None else 'stop' if not permission else 'wait' if not ready else task
    if row['kind'] == 'info':
        required = [permission, ready] if task == 'record' else [permission] if task == 'play' else [ready]
        return 'yes' if all(value is not None for value in required) else 'no'
    if permission is None or ready is None:
        return 'none'
    return str(int(permission) + int(ready) if task == 'record' else 2 if (permission if task == 'play' else ready) else int(ready if task == 'play' else permission))

manual_counts = defaultdict(lambda: [0, 0])
manual_labels_checked = 0
for row in manual['observations']:
    expected = rule(row)
    assert row['expected'] == expected
    manual_labels_checked += 1
    manual_counts[row['kind']][0] += row['prediction']['model_choice'] == expected
    manual_counts[row['kind']][1] += 1
assert dict(manual_counts) == {'choice': [972, 972], 'info': [972, 972], 'rubric': [972, 972], 'removed': [432, 432]}
assert not any(row['prediction']['status'] == 'accepted' and row['prediction']['model_choice'] != rule(row) for row in manual['observations'])
legacy_counts = {kind: [sum(row['prediction']['model_choice'] == row['gold'] for row in legacy['observations'] if row['kind'] == kind),
                        sum(row['kind'] == kind for row in legacy['observations'])] for kind in ('choice', 'info', 'rubric')}
assert all(value == [60, 60] for value in legacy_counts.values())

# Real CPU public-entry smoke examples from the fixed local archive. No download or final run.
from scripts.course_product import dispatch_request, install_archive
from tiny_perceptron.decision_product.service import DecisionEngine

cpu_examples = []
with tempfile.TemporaryDirectory(prefix='c-v3-v21-r2-inference-') as temporary:
    bundle = Path(temporary) / 'bundle'
    install_archive(ARCHIVE, bundle, descriptor)
    for name in ('record-ready.json', 'record-unknown.json', 'answer-absent.json'):
        cmd = [sys.executable, '-m', 'scripts.course_product', 'judge', '--bundle', str(bundle), '--request', str(ROOT / 'examples/decision_product' / name)]
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
        response = json.loads(result.stdout)
        assert response['product'] == 'C-v3' and response['dispatch'] == 'typed-devices'
        cpu_examples.append({'request': name, 'command': cmd, 'returncode': result.returncode, 'response': response, 'stderr': result.stderr})
    first, unknown, absent = [row['response']['results'] for row in cpu_examples]
    assert [first[kind]['selected'] for kind in ('choice', 'info', 'rubric')] == ['record', 'yes', '2']
    assert [unknown[kind]['selected'] for kind in ('choice', 'info', 'rubric')] == [None, 'no', None]
    assert absent['choice']['status'] == 'abstained'
    engine = DecisionEngine(bundle, 'cpu')
    invalid_task = json.loads((ROOT / 'examples/decision_product/record-ready.json').read_text())
    invalid_task['devices'][0]['task'] = 'paint'
    rejection_checks = []
    for request in ({'text': 'Please choose a service'}, invalid_task):
        try:
            dispatch_request(engine, request)
        except ValueError as error:
            rejection_checks.append(str(error))
        else:
            raise AssertionError('unsupported input was accepted')

authority_inputs = [
    ('s_bigram', 'https://web.stanford.edu/~jurafsky/slp3/3.pdf', 'Speech and Language Processing, draft 2026-08-19, Chapter 3 §3.1.1 Eq.3.7', 'ngrams', [(130, 152)]),
    ('s_attention', 'https://arxiv.org/pdf/1706.03762v7', 'arXiv:1706.03762v7, 2023-08-02, §3.2/3.4 and Table 1', 'attention', [(183, 199), (253, 259), (263, 272)]),
    ('s_chat', 'https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/chat_templating.md', 'Transformers v4.57.1 chat templates, lines 23–26, 132–133, 194–196', 'chat-templating', [(23, 26), (132, 133), (194, 196)]),
    ('s_omni', 'https://arxiv.org/pdf/2503.20215v1', 'arXiv:2503.20215v1, 2025-03-26, §2.2 Video and TMRoPE / §2.4 Designs for Streaming', 'omni', [(180, 186), (231, 237)]),
    ('s_reuse', 'https://arxiv.org/pdf/2311.04934v2', 'arXiv:2311.04934v2, 2024-04-25, §1 Introduction', 'prompt-cache', [(42, 55)]),
]
source_excerpts = []
for identifier, url, version, stem, ranges in authority_inputs:
    is_markdown = stem == 'chat-templating'
    original = Path('/tmp/c-v3-v21-r2-' + stem + ('.md' if is_markdown else '.pdf'))
    text_path = original if is_markdown else original.with_suffix('.txt')
    # pdftotext form-feeds are page separators, not sed/rg line separators.
    lines = text_path.read_text().split('\n')
    source_excerpts.append({'source_id': identifier, 'url': url, 'version': version,
                            'accessed_at': started, 'original_bytes': original.stat().st_size,
                            'original_sha256': sha(original.read_bytes()),
                            'excerpts': [{'text_line_range': [start, end], 'text': '\n'.join(lines[start - 1:end])} for start, end in ranges],
                            'preservation': 'Short supporting excerpts only; full third-party document remains outside Git.'})
write_new('source-excerpts.json', source_excerpts)

receipt = {
    'schema': 'c-v3-v21-numbered-r2-audit-v1', 'reviewer_task': '/root/c_v3_technical_coord/numbered_r2',
    'started_at': started, 'completed_at': datetime.now(UTC).isoformat(),
    'environment': {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'platform': platform.platform(), 'device': 'cpu', 'threads': '1 (CLI child uses 2)'},
    'scope': 'No training, optimizer update, new final opening, final inference, Modal, upload, publication or commit. Existing final/manual observations are retained historical evidence, recomputed here; three CPU public examples are newly executed smoke checks only.',
    'source_sha256': sha(body.encode()), 'canonical_section_matches_freeze': True,
    'navigation': {'link_count': len(links), 'all_targets_and_anchors_exist': True, 'links': links, 'heading_inventory': linked_heading_inventory},
    'bigram': bigram, 'chat': chat,
    'archive': {'path': str(ARCHIVE), 'bytes': len(archive_bytes), 'sha256': sha(archive_bytes), 'members_read': member_receipts,
                'metadata_byte_identical_to_members': True, 'implementation_files_match_opening': len(opening['implementation']),
                'public_transport_retested': False},
    'retained_measurement_recalculation': {'evidence_mode': 'retained saved logits and manual observations; not rerun training or final inference',
        'final_metrics': measured, 'choice_accepted': accepted, 'choice_accepted_errors': errors,
        'choice_answerable_accepted': answerable_accepted, 'choice_coverage_all': accepted / 1500,
        'one_sided_wilson95_upper': upper, 'unique_final_families': len(family_keys),
        'observations': len(observations), 'one_per_kind_per_family': True,
        'final_device_counts': sorted({len(row['facts']) for row in families}), 'overlap_with_previous_exposed': len(family_keys & excluded),
        'manual_labels_independently_checked': manual_labels_checked, 'manual_counts': dict(manual_counts),
        'manual_invariance_failures_retained_only': manual['invariance_failures'], 'legacy_counts': legacy_counts,
        'training_seed': 59, 'parent_updates_retained': 4000, 'own_updates_retained': 1500,
        'final_seed_retained': opening['protocol'], 'policy': policy},
    'new_cpu_examples': cpu_examples, 'unsupported_rejections': rejection_checks,
    'limits': ['Final types share the same 1500 visible-fact families; do not pool 4500 as independent worlds.',
               'Three-device evidence is manual engineering and already-exposed 60-world regression, not fresh final.',
               'Typed bilingual record/play/save fields only; no general natural-language or novel-task guarantee.',
               'CPU examples are not performance, GPU/MPS, or broad reliability evidence; suggestions do not execute devices.',
               'This audit does not re-establish original training history or public-release network transport.'],
}
write_new('execution.json', receipt)
print(json.dumps({'links': len(links), 'bigram_equal': True, 'retained_final_families': len(family_keys),
                  'retained_accepted': accepted, 'retained_errors': errors, 'manual': dict(manual_counts),
                  'cpu_examples': len(cpu_examples), 'unsupported_rejections': len(rejection_checks), 'artifact': str(OUT / 'execution.json')}, ensure_ascii=False))
