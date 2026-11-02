"""第9章導讀技術審查：只核本頁導覽承諾及其必要的精確來源切片。"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import UTC, datetime

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/reference/omni-technical-chapter-09-v1')
FREEZE = 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix12.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    with (OUT / name).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def source(path):
    return (ROOT / path).read_text(encoding='utf-8')


def section(path, number):
    text = source(path)
    start = re.search(r'^## ' + re.escape(number) + r'\s', text, re.M).start()
    later = re.search(r'^## ', text[start + 1:], re.M)
    end = start + 1 + later.start() if later else len(text)
    return text[start:end]


slices = []


def record_slice(identifier, path, text, purpose):
    raw = (ROOT / path).read_bytes()
    part = text.encode('utf-8')
    start = raw.index(part)
    assert raw.count(part) == 1, identifier
    slices.append({
        'id': identifier, 'source': path,
        'byte_start': start, 'byte_end_exclusive': start + len(part),
        'line_start': raw[:start].count(b'\n') + 1,
        'line_end': raw[:start + len(part)].count(b'\n') + (not part.endswith(b'\n')),
        'slice_sha256': sha(part), 'purpose': purpose, 'text': text,
    })


def prefix(number, path='course/chapters/09.md'):
    text = section(path, number)
    return text.split('\n```python\n', 1)[0]


def paragraph(path, starts):
    text = source(path)
    start = text.index(starts)
    end = text.find('\n\n', start)
    return text[start:end if end >= 0 else len(text)]


started = datetime.now(UTC).isoformat()
frozen_raw = (ROOT / FREEZE).read_bytes()
page = next(p for p in json.loads(frozen_raw)['pages'] if p['page_id'] == 'chapter-09')
body = (ROOT / page['snapshot']).read_bytes()
assert sha(body) == page['source_sha256']
assert body == (ROOT / page['source']).read_bytes().split(b'## ', 1)[0]
assert page['figures_sha256'] == {} and b'```' not in body and b'![' not in body

links = []
for label, link in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', body.decode()):
    path, _, fragment = link.partition('#')
    target = (ROOT / page['source']).parent / path
    assert target.is_file(), link
    target_text = target.read_text(encoding='utf-8')
    if fragment:
        assert re.search(r'^## ' + re.escape(fragment) + r'(?:\s|$)', target_text, re.M), link
    links.append({'label': label, 'link': link, 'target': target.resolve().relative_to(ROOT).as_posix(),
                  'heading_found': bool(fragment), 'file_exists': True})
assert len(links) == 6
for number in ['9.1', '9.2', '9.3', '9.4', '9.5', '9.8', '9.9', '9.10',
               '9.11', '9.12', '9.13', '9.14', '9.15', '9.16', '9.17', '9.18']:
    record_slice(number + '-opening', 'course/chapters/09.md', prefix(number),
                 '只核導讀對此節例子、工作或機制的承接；不宣稱整節完整技術重審。')

for identifier, starts in [
    ('9.11-typed', '這類介面稱為**有型別的機率判斷**'),
    ('9.11-jev', '外部也有這類服務。**Jev**'),
    ('9.11-own', '這三種型別是任務與輸出的約定'),
]:
    record_slice(identifier, 'course/chapters/09.md', paragraph('course/chapters/09.md', starts),
                 '核有型別介面、外部Jev與本課自製版本的區別。')

record_slice('19.16-old-c', 'course/chapters/19.md',
             section('course/chapters/19.md', '19.16').split('\n主要測試題中，', 1)[0],
             '舊版C的自製有限世界、候選評分及既有訓練事實；未取整節或全章hash。')
record_slice('19.23-new-c', 'course/chapters/19.md', prefix('19.23', 'course/chapters/19.md'),
             '新版C明連9.17/9.18，且明說為隨機初值CPU參考實作。')
text19 = section('course/chapters/19.md', '19.23')
start = text19.index('訓練時同題全部')
end = text19.index('\n核心final每家族', start)
record_slice('19.23-calibration-policy', 'course/chapters/19.md', text19[start:end],
             '新版C的溫度擬合核心及固定接受規則；不把事前目標當成實測結果。')
record_slice('19.23-limits', 'course/chapters/19.md',
             paragraph('course/chapters/19.md', '正式模型預定另實作於'),
             '正式新模型與密封測試仍待後續，不能冒稱已有新版checkpoint。')
glossary = source('course/glossary.md')
for identifier, starts in [('glossary-typed', '| 有型別的機率判斷、Jev'),
                           ('glossary-calibration', '| 校準（calibration）'),
                           ('glossary-confidence', '| 信心（confidence）')]:
    line = next(line for line in glossary.splitlines(keepends=True) if line.startswith(starts))
    record_slice(identifier, 'course/glossary.md', line, '核本頁明連名詞頁具有對應條目。')

snippets = []
for path, number, count in [('course/chapters/09.md', '9.17', 1),
                            ('course/chapters/09.md', '9.18', 1),
                            ('course/chapters/19.md', '19.23', 2)]:
    chunks = re.findall(r'```python\n(.*?)```', section(path, number), re.S)
    assert len(chunks) == count
    for i, code in enumerate(chunks):
        record_slice(number + '-python-' + str(i + 1), path, code,
                     '實際執行的教材內嵌原碼；輸入、dtype、候選軸及輸出已人工核讀。')
    code = '\n'.join(chunks)
    proc = subprocess.run([sys.executable, '-c', code], cwd=ROOT, text=True,
                          capture_output=True, timeout=30)
    assert proc.returncode == 0, (number, proc.stderr)
    if number == '9.17':
        assert len(proc.stdout.splitlines()) == 8
        assert "{'card': 0.7124, 'address': 0.1732, 'none': 0.1145}" in proc.stdout
    elif number == '9.18':
        expected = re.findall(r'```text\n(.*?)```', section(path, number), re.S)[0]
        assert proc.stdout == expected
    else:
        expected = re.findall(r'```text\n(.*?)```', section(path, number), re.S)[0]
        assert proc.stdout.startswith(expected)
        assert proc.stdout.endswith('等分數人工校準T 1.0\n')
    snippets.append({'section': number, 'code_sha256': sha(code.encode()),
                     'command': [sys.executable, '-c', '<exact inline code recorded in evidence-slices.json>'],
                     'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr})


def tensor_literal(number, name):
    code = re.findall(r'```python\n(.*?)```', section('course/chapters/09.md', number), re.S)[0]
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value.args[0])
    raise AssertionError(name)


counts = {'9.9_candidates': len(tensor_literal('9.9', 'scores')[0]),
          '9.10_questions': len(tensor_literal('9.10', 'confidence'))}
assert counts == {'9.9_candidates': 3, '9.10_questions': 4}

sys.path.insert(0, str(ROOT))
from tiny_perceptron.lesson_products import data
import torch

world = {'task': 'card', 'application': 'K17', 'family_id': 'reviewer-synthetic-only'}
judgments = {'state': data.render(world),
             'choice': data.judgment(world, (), 'choice', ['card', 'address', 'none']),
             'info_complete': data.judgment(world, (), 'info', ['yes', 'no']),
             'info_missing': data.judgment(world, ('application',), 'info', ['yes', 'no']),
             'rubric_missing': data.judgment(world, ('application',), 'rubric', ['grade0', 'grade1', 'grade2']),
             'choice_removed': data.judgment(world, (), 'choice', ['address', 'none'])}
assert judgments == {'state': '卡片進度;申請編號=K17', 'choice': 'card', 'info_complete': 'yes',
                     'info_missing': 'no', 'rubric_missing': 'grade0', 'choice_removed': 'none'}
data_path = 'tiny_perceptron/lesson_products/data.py'
data_hash = sha((ROOT / data_path).read_bytes())
record_slice('data-labels', data_path, source(data_path)[source(data_path).index('def judgment('):
             source(data_path).index('\n\ndef build_rows(')], '自製資料的確定性標籤規則與缺候選變體。')

receipt_path = 'docs/lesson-products/results/initial-c-execution.json'
archived = json.loads((ROOT / receipt_path).read_bytes())
t = archived['training']
manifest_path = 'docs/lesson-products/data-v2-manifest.json'
manifest_raw = (ROOT / manifest_path).read_bytes()
manifest = json.loads(manifest_raw)
assert archived['status'] == 'completed' and t['product'] == 'c'
assert t['random_initialization'] is True and t['weights_changed'] is True
assert t['steps_this_attempt'] == t['last_step'] == t['selected_step'] == 800
assert t['initial_weights_sha256'] != t['selected_weights_sha256']
assert t['data_manifest_sha256'] == sha(manifest_raw)
assert archived['source_hashes'][data_path] == data_hash
old_training = {'record_path': receipt_path, 'record_sha256': sha((ROOT / receipt_path).read_bytes()),
                'run_id': archived['run_id'], 'status': archived['status'],
                **{key: t[key] for key in ['product', 'random_initialization', 'steps_this_attempt',
                    'last_step', 'selected_step', 'seed', 'batch_size', 'weights_changed',
                    'data_manifest_sha256', 'initial_weights_sha256', 'selected_weights_sha256', 'environment']},
                'manifest_path': manifest_path, 'manifest_version': manifest['config']['version'],
                'c_dataset_rows': manifest['files']['c.jsonl']['rows'],
                'current_generator_sha256': data_hash,
                'scope': '讀取並交叉核對既有原始訓練收據；本次沒有啟動訓練、載入權重或驗證泛化品質。'}

external = [
    {'id': 'guo2017', 'url': 'https://proceedings.mlr.press/v70/guo17a/guo17a.pdf',
     'version': 'ICML 2017 proceedings, PMLR 70:1321–1330, publisher PDF',
     'accessed_on': datetime.now(UTC).date().isoformat(),
     'local_cache': str(CACHE / 'guo17a.pdf'), 'sha256': sha((CACHE / 'guo17a.pdf').read_bytes()),
     'bytes': (CACHE / 'guo17a.pdf').stat().st_size,
     'retrieval_command': 'curl -fL --max-time 45 https://proceedings.mlr.press/v70/guo17a/guo17a.pdf -o /workspace/work/reference/omni-technical-chapter-09-v1/guo17a.pdf',
     'inspection_locator': 'PDF p.2, §2 Definitions, Eq.(1), Reliability Diagrams paragraph',
     'formula_transcription': 'P(Y_hat = Y | P_hat = p) = p, for all p in [0,1]',
     'short_excerpt': 'To estimate the expected accuracy from finite samples, we group predictions into M interval bins',
     'inspection_note': '讀出版方PDF並用pdftotext -layout抽取本機文字；核分類信心的條件頻率定義與有限樣本分箱近似，不外推成逐題保證。'},
    {'id': 'jev-introduction', 'url': 'https://docs.typesafe.ai/introduction.md',
     'version': 'TypeSafe live official Markdown, accessed 2026-10-29; SHA-bound below',
     'accessed_on': datetime.now(UTC).date().isoformat(),
     'local_cache': str(CACHE / 'jev-introduction.md'), 'sha256': sha((CACHE / 'jev-introduction.md').read_bytes()),
     'bytes': (CACHE / 'jev-introduction.md').stat().st_size,
     'retrieval_command': 'curl -fL --max-time 45 https://docs.typesafe.ai/introduction.md -o /workspace/work/reference/omni-technical-chapter-09-v1/jev-introduction.md',
     'inspection_locator': 'Introduction paragraphs 1–3; TypeSafe primitives table',
     'short_excerpt': 'Jev evaluates typed *questions* against a *state* and returns structured results directly. No text generation, no parsing.',
     'inspection_note': '實讀TypeSafe官方原文；只支持Jev的供應者、狀態／問題介面與型別化結果，不採用廣告性的可靠度、速度或未公開訓練結論。'}]

write('evidence-slices.json', {'created_at': datetime.now(UTC).isoformat(),
      'policy': '僅保存導讀主張所需的精確UTF-8段落與實際執行程式切片；source byte offsets以本次live檔為準。',
      'slices': slices})
write('external-sources.json', {'created_at': datetime.now(UTC).isoformat(), 'sources': external,
      'third_party_fulltext_storage': 'Git外的持久本機cache；repo僅存定位、短摘錄與hash。'})
execution = {'started_at': started, 'completed_at': datetime.now(UTC).isoformat(),
             'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-chapter-09-v1/review-audit.py',
             'environment': {'python': sys.version, 'torch': str(torch.__version__), 'device': 'cpu',
                             'platform': sys.platform},
             'page': {'lesson_id': 'chapter-09', 'snapshot': page['snapshot'], 'source_sha256': sha(body),
                      'live_matches_freeze': True, 'figure_sha256': {}, 'embedded_code_blocks': 0,
                      'freeze': FREEZE, 'freeze_sha256': sha(frozen_raw)},
             'resolved_links': links, 'example_counts': counts, 'linked_inline_programs': snippets,
             'selfmade_label_probe': judgments, 'archived_training_crosscheck': old_training,
             'scope': '本頁完整導讀的主張核對；連結目的段落只查承接所需內容，未冒稱第9章或第19章逐節完整重審。',
             'result': 'all focused assertions passed'}
write('execution.json', execution)
print(json.dumps({'result': execution['result'], 'source_sha256': sha(body), 'links': len(links),
                  'slices': len(slices), 'linked_programs': [s['section'] for s in snippets],
                  'environment': execution['environment']}, ensure_ascii=False))
