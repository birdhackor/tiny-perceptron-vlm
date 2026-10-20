"""核對凍結頁面、沿用證據與開放的訓練版本紀錄；不讀其他輸出檔。"""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron import natural_assistant as core


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_bytes())


original_path = 'docs/course-review-20261019/reviews/rv1019_technical_natural-v4-training/technical-natural-v4-training-original-01.json'
original_bytes = (ROOT / original_path).read_bytes()
assert digest(original_bytes) == '7618e3bd2ec340f2c0b6517464a6a0ce97e1f5179a3a3b00c84d50a4f5137165'
original = json.loads(original_bytes)
reused = []
for category in ('sources', 'artifacts'):
    for item in original[category]:
        if 'path' not in item:
            continue
        actual = digest((ROOT / item['path']).read_bytes())
        assert actual == item['sha256'], item['path']
        reused.append({'category': category, 'id': item['id'], 'path': item['path'],
                       'recorded_sha256': item['sha256'], 'current_sha256': actual, 'unchanged': True})

source_path = 'docs/natural-assistant/v4/TRAINING.md'
snapshot_path = 'docs/course-review-20261019/freeze-01/sources/natural-v4-training.md'
source = (ROOT / source_path).read_bytes()
snapshot = (ROOT / snapshot_path).read_bytes()
assert source == snapshot and digest(source) == original['source_sha256']
freeze_path = 'docs/course-review-20261019/freeze-05.json'
freeze_raw = (ROOT / freeze_path).read_bytes()
assert digest(freeze_raw) == 'e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba'
page = next(page for page in json.loads(freeze_raw)['pages'] if page['page_id'] == 'natural-v4-training')
assert page['source_sha256'] == digest(source) and page['figures_sha256'] == {}

index_path = 'docs/natural-assistant/evidence/v4-runtime/train-37217452291/actual-artifact-index.json'
index_raw = (ROOT / index_path).read_bytes()
entry = next(item for item in json.loads(index_raw)['files'] if item['path'] == 'review/training.json')
allowed_path = 'outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1/review/training.json'
training_raw = (ROOT / allowed_path).read_bytes()
actual = digest(training_raw)
assert actual == entry['sha256'] and len(training_raw) == entry['bytes']
# 只有完整位元組的指紋核對成功後，才解析允許的版本欄位。
versions = json.loads(training_raw)['versions']
expected_versions = {'torch': '2.8.0+cu128', 'torchvision': '0.23.0+cu128',
                     'transformers': '4.57.6', 'peft': '0.18.1'}
assert all(versions[name] == expected for name, expected in expected_versions.items())

manifest = load('docs/natural-assistant/v4/manifest.json')
rows = [row for row in manifest['rows'] if row['split'] == 'train']
visits = [core.training_row_at(rows, index, 42)['id'] for index in range(4154)]
assert len(rows) == 2077 and len(set(visits)) == 2077 and set(Counter(visits).values()) == {2}
base = sum(map(Fraction, ['1/2', '1', '4/5', '1/3', '1'])) / 5
chat = (Fraction(1) + Fraction(3, 4) + Fraction(1)) / 3
candidate = (Fraction(3, 4) + 1 + Fraction(4, 5) + Fraction(1, 3) + chat) / 5
assert base == Fraction(109, 150) and chat == Fraction(11, 12) and candidate == Fraction(19, 25)

previous_run = load(str(ARTIFACTS.relative_to(ROOT) / 'execution.json'))
raw_validation = []
for variant, record in previous_run['validation_raw_recount'].items():
    path = 'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/raw/generations-' + variant + '.json'
    actual = digest((ROOT / path).read_bytes())
    assert actual == record['file_sha256']
    raw_validation.append({'path': path, 'sha256': actual, 'unchanged': True})

command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
           'tests/test_natural_assistant.py', 'tests/test_natural_selected_only.py',
           'tests/test_natural_v4_scoring.py', 'tests/test_natural_v4_data_download.py']
with tempfile.TemporaryDirectory(prefix='recheck-', dir=ARTIFACTS) as temporary:
    process = subprocess.run(command + ['--basetemp', temporary], cwd=ROOT,
                             env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True)

result = {
    'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-natural-v4-training/verify_recheck_current_01.py',
    'executed_at': datetime.now(UTC).isoformat(),
    'environment': {'python': sys.version, 'platform': platform.platform(),
                    'torch': importlib.metadata.version('torch'), 'device': 'CPU'},
    'original_report': {'path': original_path, 'sha256': digest(original_bytes)},
    'source_comparison': {'source': source_path, 'snapshot': snapshot_path, 'source_sha256': digest(source),
                          'snapshot_sha256': digest(snapshot), 'changed_sentences': [], 'byte_identical': True},
    'source_freeze': {'path': freeze_path, 'sha256': digest(freeze_raw)},
    'reused_evidence_hashes': reused,
    'raw_validation_hashes': raw_validation,
    'training_record_excerpt': {
        'path': allowed_path, 'bytes': len(training_raw), 'sha256': digest(training_raw),
        'index': {'path': index_path, 'sha256': digest(index_raw), 'entry': entry},
        'sha256_checked_before_json_parse': True, 'versions': versions,
        'expected_versions': expected_versions, 'all_expected_versions_match': True,
        'scope': '只摘錄已核對整檔SHA與大小的training.json內versions；未開啟其他outputs檔。',
    },
    'arithmetic': {'training_rows': len(rows), 'row_visits': len(visits), 'visits_per_row': 2,
                   'base_example': str(base), 'dialogue_example': str(chat), 'candidate_example': str(candidate)},
    'pytest': {'command': command + ['--basetemp', '<本artifact目錄內的暫存子目錄；結束後移除>'],
               'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr},
    'scope': '沿用證據先逐檔重新核對SHA，重跑相同既有離線測試與透明算例；未下載模型、未長時間訓練、未連Modal或HuggingFace寫入。',
}
with (ARTIFACTS / 'recheck-execution-current-01.json').open('x') as file:
    json.dump(result, file, ensure_ascii=False, indent=2)
    file.write('\n')
print(json.dumps({'versions': versions, 'training_record_sha256': digest(training_raw),
                  'byte_identical': True, 'unchanged_evidence_count': len(reused),
                  'pytest': result['pytest'], 'arithmetic': result['arithmetic']}, ensure_ascii=False, indent=2))
raise SystemExit(process.returncode)
