"""Bounded source-first probes for course/README.md; no training or final evaluation."""

import hashlib
import json
import platform
import re
import subprocess
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path('/tmp/tpb').resolve()
OUT = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-canonical-course/probe-course-v1.json'
ZIP = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')
PREFIX = 'tiny-perceptron-c-v3/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path.read_bytes())}


record = {
    'created_at': datetime.now(UTC).isoformat(),
    'actor': '/root/c_v3_technical_coord/canonical_course',
    'scope': 'Source-first whole-page navigation, fixed release identity, and three CPU examples only; no training, final evaluation, scientific requalification, installation, or copied ZIP/weights.',
    'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform(), 'device': 'cpu'},
    'source': ref(ROOT / 'course/README.md'),
    'snapshot': ref(ROOT / 'docs/course-revision-20261101-c-v3/source-freeze-v19/sources/course.md'),
}
body = (ROOT / 'course/README.md').read_text()
links = []
for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)', body):
    relative, _, fragment = link.partition('#')
    target = ((ROOT / 'course') / relative).resolve() if relative else ROOT / 'course/README.md'
    row = {'link': link, 'path': str(target.relative_to(ROOT)), 'exists': target.is_file()}
    if target.is_file():
        text = target.read_text()
        row['sha256'] = sha(target.read_bytes())
        headings = re.findall(r'^#{1,6} .+$', text, re.M)
        row['first_heading'] = headings[0] if headings else None
        if fragment:
            row['matched_heading'] = next((line for line in headings if re.match(r'^#{1,6} ' + re.escape(fragment) + r'(?:\s|$)', line)), None)
    links.append(row)
record['navigation'] = {'count': len(links), 'links': links, 'all_local_files_and_numbered_anchors_exist': all(x['exists'] and ('#' not in x['link'] or x.get('matched_heading')) for x in links)}
text = '貓看狗，狗看貓。'
chars = sorted(set(text))
ids = [chars.index(char) for char in text]
record['reversible_example'] = {'text': text, 'chars': chars, 'ids': ids, 'cat_ids': [ids[i] for i, char in enumerate(text) if char == '貓'], 'restored': ''.join(chars[i] for i in ids), 'assertion': ''.join(chars[i] for i in ids) == text}
record['license'] = {'source': ref(ROOT / 'LICENSE'), 'MIT_header_present': 'MIT License' in (ROOT / 'LICENSE').read_text()}

descriptor = json.loads((ROOT / 'docs/decision-product/v3/public-release.json').read_text())
record['descriptor'] = ref(ROOT / 'docs/decision-product/v3/public-release.json')
record['local_archive'] = {'path': str(ZIP), 'bytes': ZIP.stat().st_size, 'sha256': sha(ZIP.read_bytes()), 'copied': False}
assert record['local_archive']['bytes'] == descriptor['bytes']
assert record['local_archive']['sha256'] == descriptor['sha256']
metadata = json.loads((ROOT / 'docs/decision-product/v3/metadata/acquisition.json').read_text())
with zipfile.ZipFile(ZIP) as archive:
    bundle = {}
    for name, digest in descriptor['bundle'].items():
        member = PREFIX + 'bundle/' + name
        raw = archive.read(member)
        path = ROOT / 'checkpoints/decision-product-v3' / name
        bundle[name] = {'archive_member': member, 'bytes': len(raw), 'sha256': sha(raw), 'installed_sha256': sha(path.read_bytes()), 'same_installed_bytes': raw == path.read_bytes()}
        assert sha(raw) == digest and raw == path.read_bytes()
    record['bundle_identity'] = bundle
    sources = []
    for name, digest in descriptor['implementation'].items():
        raw = (ROOT / name).read_bytes()
        archived = archive.read(PREFIX + name)
        sources.append({'path': name, 'sha256': sha(raw), 'same_archive_bytes': raw == archived})
        assert sha(raw) == digest and raw == archived
    record['implementation_identity'] = sources
    comparisons = []
    for row in metadata['files']:
        raw = (ROOT / row['path']).read_bytes()
        archived = archive.read(row['archive_member'])
        comparisons.append({'path': row['path'], 'archive_member': row['archive_member'], 'bytes': len(raw), 'sha256': sha(raw), 'same_archive_bytes': raw == archived})
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256'] and raw == archived
    record['small_raw_metadata_identity'] = comparisons
    policy = json.loads(archive.read(PREFIX + 'bundle/policy.json'))
    record['policy'] = policy

# One client-side anonymous GET of the fixed public URL. No local archive copy.
request = urllib.request.Request(descriptor['url'], headers={'User-Agent': 'tiny-perceptron-source-first-technical-course-review'})
download = {'started_at': datetime.now(UTC).isoformat(), 'url': descriptor['url'], 'method': 'GET', 'client_authorization_header_present': request.has_header('Authorization'), 'attempts': 1, 'saved_download': False}
try:
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=55) as response:
        download['http_status'] = response.status
        download['redirected'] = response.geturl() != descriptor['url']
        download['content_type'] = response.headers.get('Content-Type')
        while block := response.read(1024 * 1024):
            size += len(block)
            digest.update(block)
            if size > descriptor['bytes']:
                raise ValueError('response exceeds pinned archive size')
    download.update({'bytes': size, 'sha256': digest.hexdigest(), 'matches_fixed_archive': size == descriptor['bytes'] and digest.hexdigest() == descriptor['sha256']})
except Exception as error:
    download.update({'error_type': type(error).__name__, 'error': str(error), 'matches_fixed_archive': False})
download['completed_at'] = datetime.now(UTC).isoformat()
record['public_transport'] = download

examples = []
expected = {
    'record-ready.json': {'choice': ('accepted', 'record'), 'info': ('prediction', 'yes'), 'rubric': ('prediction', '2')},
    'record-unknown.json': {'choice': ('abstained', None), 'info': ('prediction', 'no'), 'rubric': ('abstained', None)},
    'answer-absent.json': {'choice': ('abstained', None)},
}
for name, expectations in expected.items():
    relative = 'examples/decision_product/' + name
    command = [sys.executable, '-m', 'scripts.course_product', 'judge', '--request', relative]
    started = datetime.now(UTC).isoformat()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=40)
    output = json.loads(result.stdout) if result.returncode == 0 else None
    match = result.returncode == 0 and output['product'] == 'C-v3' and output['dispatch'] == 'typed-devices' and all((output['results'][kind]['status'], output['results'][kind]['selected']) == wanted for kind, wanted in expectations.items())
    examples.append({'request': ref(ROOT / relative), 'command': command, 'started_at': started, 'completed_at': datetime.now(UTC).isoformat(), 'exit_code': result.returncode, 'stdout': output, 'stderr': result.stderr, 'matches_expected': match})
    assert match
record['cpu_examples'] = examples

from scripts.course_product import dispatch_request
from tiny_perceptron.decision_product.service import DecisionEngine

torch.set_num_threads(2)
engine = DecisionEngine(ROOT / 'checkpoints/decision-product-v3')
calls = []
hook = engine._model.register_forward_hook(lambda module, inputs, output: calls.append({'model_class': type(module).__name__, 'logits_shape': list(output.shape), 'device': str(output.device), 'finite_required_logits': bool(torch.isfinite(output[0, :-1]).all())}))
request = json.loads((ROOT / 'examples/decision_product/record-ready.json').read_text())
dispatch_request(engine, request)
hook.remove()
record['neural_forward_observed'] = {'calls': calls, 'parameters_device': str(next(engine._model.parameters()).device), 'training': engine._model.training, 'model_class': type(engine._model).__name__, 'state_encoder_class': type(engine._model.state_encoder).__name__, 'cache_entries_after_all_three_questions': len(engine._cache), 'configuration': engine._model.metadata()}
assert len(calls) == 3 and all(row['device'] == 'cpu' for row in calls)
record['legacy_text_boundary'] = []
for request in [{'text': 'add three numbers'}, [], None]:
    try:
        dispatch_request(None, request)
    except ValueError as error:
        record['legacy_text_boundary'].append({'input': request, 'error': str(error)})
    else:
        raise AssertionError('legacy text unexpectedly dispatched')
record['completed_at'] = datetime.now(UTC).isoformat()
with OUT.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({'artifact': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT.read_bytes()), 'navigation_links': len(links), 'navigation_pass': record['navigation']['all_local_files_and_numbered_anchors_exist'], 'public_transport': record['public_transport'], 'cpu_examples': [{'name': row['request']['path'], 'matches_expected': row['matches_expected']} for row in examples], 'neural_forward_observed': record['neural_forward_observed']}, ensure_ascii=False, indent=2))
