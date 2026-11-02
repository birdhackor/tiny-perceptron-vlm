"""Bounded T.5 callback: exact prerequisites, their short code, and preserved evidence hashes."""
from contextlib import redirect_stdout
from datetime import UTC, datetime
import hashlib
from io import StringIO
import json
from pathlib import Path
import platform
import re

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REPORT_DIR = ROOT / 'docs/course-revision-20261029-omni/reviews/technical/technical_t_5'
OLD = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix07.json'
NEW = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix10.json'
NUMBERED = ROOT / 'docs/course-revision-20261029-omni/reviews/numbered-freeze-v8.json'
torch.set_num_threads(2)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def reference(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}

def section_bytes(path, identifier):
    text = path.read_bytes().decode('utf-8')
    headings = list(re.finditer(r'^## (\S+) .+$', text, re.M))
    for index, match in enumerate(headings):
        if match[1] == identifier:
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            return text[match.start():end].encode('utf-8')
    raise ValueError(identifier)

old = json.loads(OLD.read_bytes())
new = json.loads(NEW.read_bytes())
numbered = json.loads(NUMBERED.read_bytes())
original_path = REPORT_DIR / 'technical-original-v1.json'
previous_path = REPORT_DIR / 'technical-current-v1.json'
previous = json.loads(previous_path.read_bytes())
result = {'created_at': datetime.now(UTC).isoformat(),
          'reviewer_task': '/root/omni_technical_coordinator/technical_t_5',
          'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu', 'threads': '2'},
          'original_report': reference(original_path), 'previous_current_report': reference(previous_path),
          'old_canonical_freeze': reference(OLD), 'current_canonical_freeze': reference(NEW),
          'current_numbered_freeze': reference(NUMBERED), 'prerequisites': []}
notes = {
 '9.1': '重讀全部正文與補充：相對偏好和各自安全標籤不同；兩個皆unsafe仍有相對勝者，裁短可能失去上下文。與T.5選safer且safe及片段限制一致。',
 '9.2': '重讀全部正文與補充：目標只看visible_count，缺資訊請補充；碰巧猜中不算有根據，改問法失敗仍保留。支持T.5把澄清與拒絕分開。',
 '9.4': '重讀全部正文與補充：只規定兩個有效場景；permission來自可信管理流程，公開性非布林推出；正式題無owner/公開欄位。支持T.5 toy規約且不外推真實授權。',
 '9.6': '重讀全部正文與補充：應拒絕和正常題有不同分母；正常未拒絕不等於完成；固定拒絕詞偵測不是真實安全裁判。支持T.5評估及限制。',
}
for identifier in ['9.1', '9.2', '9.4', '9.6']:
    prior = next(page for page in old['pages'] if page['page_id'] == identifier)
    current = next(page for page in new['pages'] if page['page_id'] == identifier)
    prior_path, current_path = ROOT / prior['snapshot'], ROOT / current['snapshot']
    old_bytes, current_bytes = prior_path.read_bytes(), current_path.read_bytes()
    live = section_bytes(ROOT / current['source'], identifier)
    assert digest(prior_path) == prior['source_sha256']
    assert digest(current_path) == current['source_sha256']
    assert old_bytes == current_bytes == live
    outputs = []
    for block in re.findall(r'```python\n(.*?)\n```', current_bytes.decode('utf-8'), re.S):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(compile(block, current['snapshot'] + ':' + identifier, 'exec'), {})
        outputs.append(stdout.getvalue())
    result['prerequisites'].append({'lesson_id': identifier, 'live_source': current['source'],
        'selector': current['selector'], 'old_snapshot': reference(prior_path),
        'current_snapshot': reference(current_path), 'live_slice_sha256': hashlib.sha256(live).hexdigest(),
        'old_current_live_bytes_equal': True, 'semantic_inspection': notes[identifier],
        'python_blocks_executed': len(outputs), 'short_stdout': outputs})

t5 = next(page for page in numbered['pages'] if page['page_id'] == 'T.5')
t5_path = ROOT / t5['snapshot']
t5_prior = ROOT / previous['review_scope']['snapshot']
assert t5_path.read_bytes() == t5_prior.read_bytes() == section_bytes(ROOT / 'course/training.md', 'T.5')
assert digest(t5_path) == previous['source_sha256'] == t5['source_sha256']
result['T.5'] = {'current_snapshot': reference(t5_path), 'previous_snapshot': reference(t5_prior),
    'old_current_live_bytes_equal': True, 'figures_sha256': t5['figures_sha256'],
    'semantic_inspection': '本人重讀完整T.5標題、人工示例、資料/訓練/評估區塊、inline推論及fixed-style-recipe全部補充；24項原claims及其界線仍成立，無新issue。'}

checks = []
for source in previous['sources']:
    if source['kind'] != 'repository_code':
        continue
    current_hash = digest(ROOT / source['path'])
    matches = current_hash == source['sha256']
    if source['id'] != 's23':
        assert matches, source['id']
    checks.append({'id': source['id'], 'path': source['path'], 'recorded_sha256': source['sha256'],
                   'current_sha256': current_hash, 'matches': matches})
result['previous_dependency_hash_checks'] = checks
result['previous_artifacts'] = []
for artifact in previous['artifacts']:
    actual = digest(ROOT / artifact['path'])
    assert actual == artifact['sha256']
    result['previous_artifacts'].append({'id': artifact['id'], 'path': artifact['path'], 'sha256': actual, 'matches': True})
assert digest(original_path) == '0b50b389b3ccacdf85e45ddd1574044cf0ce1b75b02f8961897ce3ebe776c098'
assert digest(previous_path) == 'bbbeaf6a8eb8e44018944de9e23b8080db8cb21f31acf60882c3e818d30ea37f'
result['preserved_reports_unchanged'] = True
result['limits'] = '只T.5和必要9.1/9.2/9.4/9.6 callback；未代審T.4/T.9/9.18，未重跑訓練、生成、GPU、Modal、付費或發布。舊實驗只因實作和證據指紋未變而保留，不冒稱新實測。'
result['completed_at'] = datetime.now(UTC).isoformat()
output = HERE / 'callback-fix10-source-check-v1.json'
with output.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'path': reference(output), 'prerequisites_equal': len(result['prerequisites']),
    'T.5_equal': True, 'previous_dependencies_changed': [x['id'] for x in checks if not x['matches']],
    'preserved_reports_unchanged': True}, ensure_ascii=False))
