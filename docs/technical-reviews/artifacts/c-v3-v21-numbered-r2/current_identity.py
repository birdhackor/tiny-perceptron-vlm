"""Same-owner R.2 current review: identity checks only; no measurement rerun."""

import copy
import hashlib
import importlib.util
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path('/workspace/work/omni-book-integrated-v1')
BASE = 'docs/technical-reviews/artifacts/c-v3-v21-numbered-r2/'
REVIEW = 'docs/course-revision-20261101-c-v3/technical-v21/numbered_r2/'
TASK = '/root/c_v3_technical_coord/numbered_r2'
ORIGINAL_SHA = 'c4a70c8d98214b2a2d244da568ff1b087d6d30a5ee886b40bda4f3e2cbb7fa50'
INTAKE = 'docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/numbered-r2-original-v1.json'
INTAKE_SHA = '770b4e9b7d935203e1edb920e15117ab09480a12a9206d5134993b3ff3858396'


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write_new(path, value):
    with (ROOT / path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


started = datetime.now(UTC).isoformat()
assert sha(REVIEW + 'original-v1.json') == ORIGINAL_SHA
assert sha(INTAKE) == INTAKE_SHA
intake = json.loads((ROOT / INTAKE).read_text())
original = json.loads((ROOT / (REVIEW + 'original-v1.json')).read_text())
assert original['reviewer_task'] == TASK and not original['issues']
frozen_path = 'docs/course-revision-20261101-c-v3/numbered-source-freeze-v21/sections/R.2.md'
frozen = (ROOT / frozen_path).read_text()
canonical = re.search(r'^## R\.2 .+?(?=^## |\Z)', (ROOT / 'course/README.md').read_text(), re.M | re.S).group(0)
assert canonical == frozen
assert hashlib.sha256(canonical.encode()).hexdigest() == original['source_sha256']
assert sha('docs/course-revision-20261101-c-v3/numbered-source-freeze-v21.json') == '0db79be8a75904d464b0dd8245eefe93b0dd617b85583e79cd631d116fb598fd'

dependencies = []
for category, items in [('original_repository_source', original['sources']), ('original_artifact', original['artifacts'])]:
    for item in items:
        if 'path' not in item:
            continue
        actual = sha(item['path'])
        assert actual == item['sha256']
        dependencies.append({'category': category, 'path': item['path'], 'sha256': actual, 'same': True})
execution = json.loads((ROOT / (BASE + 'execution.json')).read_text())
for path, expected in sorted({(item['target'], item['sha256']) for item in execution['navigation']['links']}):
    actual = sha(path)
    assert actual == expected
    dependencies.append({'category': 'navigation_target', 'path': path, 'sha256': actual, 'same': True})
for path, expected in execution['navigation']['heading_inventory'].items():
    assert re.findall(r'^## .+$', (ROOT / path).read_text(), re.M) == expected

acquisition = json.loads((ROOT / 'docs/decision-product/v3/metadata/acquisition.json').read_text())
for item in acquisition['files']:
    actual = sha(item['path'])
    assert actual == item['sha256']
    dependencies.append({'category': 'retained_metadata', 'path': item['path'], 'sha256': actual, 'same': True})
opening = json.loads((ROOT / 'docs/decision-product/v3/metadata/final-opening.json').read_text())
for name, expected in opening['implementation'].items():
    path = 'tiny_perceptron/decision_product/' + name
    actual = sha(path)
    assert actual == expected
    dependencies.append({'category': 'fixed_product_implementation', 'path': path, 'sha256': actual, 'same': True})

environment = {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'platform': platform.platform(), 'device': 'CPU identity checks only'}
command = 'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 uv run --extra cpu --no-sync python ' + BASE + 'current_identity.py'
identity = {
    'created_at': datetime.now(UTC).isoformat(), 'started_at': started, 'reviewer_task': TASK,
    'original_report': {'path': REVIEW + 'original-v1.json', 'sha256': ORIGINAL_SHA},
    'original_intake': {'path': INTAKE, 'sha256': INTAKE_SHA},
    'source_sha256': original['source_sha256'], 'canonical_and_source21_same': True,
    'dependencies': dependencies, 'all_navigation_heading_inventories_same': True,
    'retained_execution': {'path': BASE + 'execution.json', 'sha256': sha(BASE + 'execution.json')},
    'measurement_action': 'No repeat ZIP/raw calculation, CPU inference, research, training, final opening, final inference, Modal, publication or commit. Reuse own unchanged original evidence.',
    'scope_decision': 'R.2 calls C-v3 qualified in the bounded device-service scope linked at19.25; it does not claim anonymous public transport succeeded. Local technical pass remains supported by unchanged retained evidence. Release draft/transport-unverified is a separate whole-round publication blocker supplied in current coordination context, not independently tested here.',
    'original_issue_count': 0, 'new_necessary_issues': [],
    'command': command, 'result': 'All inspected source21, repository dependencies, original artifacts, metadata and17product implementation hashes are unchanged.',
    'environment': environment,
}
write_new(BASE + 'current-identity-recheck.json', identity)

current = copy.deepcopy(original)
current['created_at'] = datetime.now(UTC).isoformat()
current['original_issue_resolutions'] = []
current['original_report'] = {'path': REVIEW + 'original-v1.json', 'sha256': ORIGINAL_SHA}
current['current_review'] = {
    'same_owner': True, 'same_source21_and_dependencies': True,
    'reused_evidence': 'Original execution.json and authoritative source-excerpts.json retain their exact hashes. Current rechecked identity/scope; earlier CPU, ZIP/raw recomputation and original-source reading are retained evidence, not redone in this callback.',
    'publication_context': 'Coordinator reports public Release remains draft and transport is unverified. No anonymous-download success is asserted; this remains a necessary publication blocker for the full round.',
    'scope_decision': 'R.2 contains local bounded qualification and navigation, not a public-download command or public-transport success claim. Original limitations already exclude transport verification and broad reliability; no new R.2 technical issue is introduced.',
}
current['artifacts'] += [
    {'id': 'a_current_code', 'kind': 'code', 'path': BASE + 'current_identity.py', 'sha256': sha(BASE + 'current_identity.py'), 'description': '本owner current的來源／依賴同identity核對程式，不重做原量測。'},
    {'id': 'a_current_identity', 'kind': 'execution', 'path': BASE + 'current-identity-recheck.json', 'sha256': sha(BASE + 'current-identity-recheck.json'), 'description': '本次current真執行的source21、原判、原收件、依賴與保留證據同identity核對收據；含技術scope與發布阻礙的區分。', 'command': command, 'result': identity['result'], 'environment': environment},
]
current['sources'].append({'id': 's_current_identity', 'kind': 'execution', 'title': '同owner current來源及依賴核對', 'verified': True, 'artifact_id': 'a_current_identity'})
for item in current['claims']:
    if item['id'] in {'c09', 'c10'}:
        item['evidence'].append({'source_id': 's_current_identity', 'locator': 'dependencies、canonical_and_source21_same、scope_decision', 'supports': '本次current核對原來源、導航及固定產品依賴未改，沿用本人保留證據；沒有重新宣稱已驗公開transport。'})
        item['artifact_ids'].append('a_current_identity')
current['checks']['source_verification']['details'] += ' 本次current另實核source21、原判/收件、原證據、八份metadata、17產品實作及導覽依賴同identity，沿用本人已讀原文與已實跑證據。'
current['checks']['limitations']['details'] += ' 公開Release仍draft、transport未驗是全輪必要發布阻礙；本節pass只覆蓋原有本機有界技術定位，不聲稱匿名下載成功。'

spec = importlib.util.spec_from_file_location('r2_current_validator', ROOT / 'scripts/check_technical_reviews.py')
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)
errors = validator._validate(ROOT, ROOT / 'course/README.md', 'R.2', canonical, current, {TASK: ['R.2']}, set())
assert errors == [], errors
write_new(REVIEW + 'current-v1.json', current)
assert sha(REVIEW + 'original-v1.json') == ORIGINAL_SHA
print(json.dumps({'path': REVIEW + 'current-v1.json', 'sha256': sha(REVIEW + 'current-v1.json'), 'verdict': current['verdict'], 'native_local_validation': errors, 'identity_receipt': BASE + 'current-identity-recheck.json', 'identity_sha256': sha(BASE + 'current-identity-recheck.json')}, ensure_ascii=False))
