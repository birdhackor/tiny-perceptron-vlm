"""Independent read-only source/output audit and tiny character-mapping examples."""

import contextlib
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
BASE = 'docs/course-revision-20261029-omni/reviews/P5-actual/'
PREVIEW = ROOT / 'outputs/omni-P5-preview-v10'


def digest(path):
    raw = Path(path).read_bytes()
    return {'path': str(Path(path).relative_to(ROOT)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def text(value):
    return ''.join(value) if isinstance(value, list) else value


def section(raw, lesson):
    pattern = rb'(?m)^## ' + re.escape(lesson.encode()) + rb' [^\n]*\n'
    match = re.search(pattern, raw)
    end = re.search(rb'(?m)^## ', raw[match.end():])
    return raw[match.start():match.end() + end.start()] if end else raw[match.start():]


numbered_path = ROOT / (BASE + 'technical-numbered-source-freeze-v5.json')
canonical_path = ROOT / (BASE + 'source-freeze-v10.json')
numbered = json.loads(numbered_path.read_bytes())
canonical = json.loads(canonical_path.read_bytes())
entry = next(p for p in numbered['pages'] if p['page_id'] == 'R.1')
parent = next(p for p in canonical['pages'] if p['page_id'] == 'course')
raw = (ROOT / entry['snapshot']).read_bytes()
assert digest(canonical_path)['sha256'] == numbered['canonical_source_freeze']['sha256']
assert digest(ROOT / entry['snapshot'])['sha256'] == entry['source_sha256']
assert digest(ROOT / parent['snapshot'])['sha256'] == parent['source_sha256']
assert raw == section((ROOT / parent['snapshot']).read_bytes(), 'R.1')
assert raw == section((ROOT / 'course/README.md').read_bytes(), 'R.1')
assert entry['figures_sha256'] == {}
assert not re.search(rb'!\[.*?\]\(.*?\.svg\)', raw)
assert b'```' not in raw

first = next(p for p in canonical['pages'] if p['page_id'] == '1.1')
first_raw = (ROOT / first['snapshot']).read_bytes()
assert digest(ROOT / first['snapshot'])['sha256'] == first['source_sha256']
assert first_raw == section((ROOT / first['source']).read_bytes(), '1.1')
code = re.findall(r'```python\n(.*?)\n```', first_raw.decode(), re.S)[0]
notebook = json.loads((ROOT / first['notebook']).read_bytes())
teaching = [c for c in notebook['cells'] if c['cell_type'] == 'code'
            and not c.get('metadata', {}).get('course_setup') and 'course_figure' not in c.get('metadata', {})]
assert len(teaching) == 1 and text(teaching[0]['source']) == code
examples = []
for name, snippet, expected_ids, expected_text in [
    ('exact-frozen-1.1', code, [3, 2, 1, 4, 1, 2, 3, 0], '貓看狗，狗看貓。'),
    ('same-code-reversed-vocabulary', code.replace('sorted(set(text))', 'sorted(set(text), reverse=True)'),
     [1, 2, 3, 0, 3, 2, 1, 4], '貓看狗，狗看貓。'),
    ('W.1-local-edit-example', code.replace('貓看狗，狗看貓。', '鳥看狗，狗看鳥。'),
     [3, 2, 1, 4, 1, 2, 3, 0], '鳥看狗，狗看鳥。'),
]:
    namespace = {}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(snippet, 'frozen_1.1:' + name, 'exec'), namespace)
    assert namespace['ids'] == expected_ids
    assert namespace['restored'] == namespace['text'] == expected_text
    examples.append({'example': name, 'source_sha256': hashlib.sha256(snippet.encode()).hexdigest(),
                     'stdout': stream.getvalue(), 'chars': namespace['chars'], 'ids': namespace['ids'],
                     'restored': namespace['restored'], 'passed': True})
assert examples[0]['ids'][0] == examples[0]['ids'][6] == 3
assert examples[0]['ids'] != examples[1]['ids'] and examples[0]['restored'] == examples[1]['restored']

receipt_path = ROOT / 'outputs/omni-P5-current-kernel-bundle-v9-receipt.json'
receipt = json.loads(receipt_path.read_bytes())
assert receipt['source_freeze']['sha256'] == digest(canonical_path)['sha256']
counts = {'notebooks': 0, 'cells': 0, 'code_cells': 0, 'visible_code_cells': 0,
          'visible_output_blocks': 0, 'errors': 0, 'download_byte_matches': 0}
first_output = None
first_preserved = None
for row in receipt['rows']:
    for field in ['source', 'executed_source', 'joined']:
        assert digest(ROOT / row[field]['path'])['sha256'] == row[field]['sha256']
    source = json.loads((ROOT / row['source']['path']).read_bytes())
    executed = json.loads((ROOT / row['joined']['path']).read_bytes())
    assert len(source['cells']) == len(executed['cells'])
    page = PREVIEW / (row['id'] + '.html')
    page_text = page.read_text()
    rendered_codes = [html.unescape(re.sub('<[^>]*>', '', s)).rstrip('\n')
                      for s in re.findall(r'<div class="language-python highlight"><pre>.*?<code>(.*?)</code></pre></div>', page_text, re.S)]
    expected_code = []
    expected_output_blocks = 0
    for original, result in zip(source['cells'], executed['cells'], strict=True):
        assert original['cell_type'] == result['cell_type']
        assert text(original['source']) == text(result['source'])
        counts['cells'] += 1
        if result['cell_type'] != 'code':
            continue
        counts['code_cells'] += 1
        assert isinstance(result['execution_count'], int)
        assert not any(o['output_type'] == 'error' for o in result.get('outputs', []))
        meta = original.get('metadata', {})
        if meta.get('course_setup') or 'course_figure' in meta:
            continue
        expected_code.append(text(original['source']).rstrip('\n'))
        counts['visible_code_cells'] += 1
        parts = []
        for output in result.get('outputs', []):
            kind = output['output_type']
            if kind == 'stream':
                parts.append('<pre>' + html.escape(text(output['text'])) + '</pre>')
            elif kind in ('display_data', 'execute_result'):
                data = output['data']
                if 'image/png' in data:
                    parts.append('<img src="data:image/png;base64,' + html.escape(text(data['image/png']), quote=True) + '" alt="本段程式的實際輸出圖">')
                elif 'text/plain' in data:
                    parts.append('<pre>' + html.escape(text(data['text/plain'])) + '</pre>')
        if parts:
            output_html = '<div class="output"><p class="output-label">實際執行結果 · CPU</p>' + ''.join(parts) + '</div>'
            # Markdown-in-details may insert whitespace between HTML elements.
            compact = lambda value: re.sub(r'>\s+<', '><', value)
            assert compact(output_html) in compact(page_text), (row['id'], 'output differs')
            expected_output_blocks += 1
        if row['id'] == '1.1':
            first_output = [o for o in result['outputs'] if o.get('name') == 'stdout'][0]['text']
    assert rendered_codes == expected_code, (row['id'], len(rendered_codes), len(expected_code))
    assert page_text.count('class="output-label"') == expected_output_blocks
    counts['visible_output_blocks'] += expected_output_blocks
    download = PREVIEW / row['source']['path']
    assert download.read_bytes() == (ROOT / row['source']['path']).read_bytes()
    counts['download_byte_matches'] += 1
    counts['notebooks'] += 1
    if row['id'] == '1.1':
        first_preserved = {**row, 'stdout': text(first_output), 'source_and_cell_types_equal': True}
assert counts['notebooks'] == receipt['total'] == 331
assert text(first_output) == examples[0]['stdout']

prerequisites = []
first_steps = next(p for p in canonical['pages'] if p['page_id'] == 'first-steps')
steps = (ROOT / first_steps['snapshot']).read_bytes()
assert digest(ROOT / first_steps['snapshot'])['sha256'] == first_steps['source_sha256']
for lesson in ('W.1', 'W.2', 'W.3'):
    e = next(p for p in numbered['pages'] if p['page_id'] == lesson)
    fragment = (ROOT / e['snapshot']).read_bytes()
    assert section(steps, lesson) == fragment == section((ROOT / 'course/first-steps.md').read_bytes(), lesson)
    assert digest(ROOT / e['snapshot'])['sha256'] == e['source_sha256']
    assert f'id="{lesson}"' in (PREVIEW / 'first-steps.html').read_text()
    prerequisites.append({**digest(ROOT / e['snapshot']), 'scope': 'Explicit navigation prerequisite read; no separate whole-section verdict.'})
glossary = next(p for p in canonical['pages'] if p['page_id'] == 'glossary')
assert digest(ROOT / glossary['snapshot'])['sha256'] == glossary['source_sha256']
assert (PREVIEW / 'glossary.html').exists()
course_html = (PREVIEW / 'course.html').read_text()
r1_html = re.search(r'<h2 id="R\.1">.*?(?=<h2 id="R\.2">)', course_html, re.S)[0]
links = re.findall('href="([^"]+)"', r1_html)
for wanted in ('1.1.html', 'first-steps.html#W.1', 'first-steps.html#W.2', 'first-steps.html#W.3', 'glossary.html'):
    assert wanted in links
for path, sha in first['figures_sha256'].items():
    assert digest(ROOT / path)['sha256'] == sha

origin = Path('/workspace/tiny-perceptron-vlm/outputs/omni-notebooks-technical-fix15-current/01/1.1.ipynb')
origin_nb = json.loads(origin.read_bytes())
assert [(c['cell_type'], text(c['source'])) for c in origin_nb['cells']] == [(c['cell_type'], text(c['source'])) for c in notebook['cells']]
assert all(c.get('execution_count') is not None for c in origin_nb['cells'] if c['cell_type'] == 'code')
assert text(origin_nb['cells'][5]['outputs'][0]['text']) == examples[0]['stdout']
raw_cpu_path = ROOT / 'docs/course-revision-20261029-omni/evidence/notebooks-kernel-v1.json'
raw_cpu = json.loads(raw_cpu_path.read_bytes())
raw_cpu_row = next(r for r in raw_cpu['executed_notebooks'] if r['page_id'] == '1.1')
initial_origin = Path('/workspace/tiny-perceptron-vlm') / raw_cpu_row['executed']
initial = json.loads(initial_origin.read_bytes())
assert hashlib.sha256(initial_origin.read_bytes()).hexdigest() == raw_cpu_row['executed_sha256']
assert raw_cpu_row['notebook_sha256'] == digest(ROOT / first['notebook'])['sha256']
assert [(c['cell_type'], text(c['source'])) for c in initial['cells']] == [(c['cell_type'], text(c['source'])) for c in notebook['cells']]
assert initial['cells'][5]['outputs'] == origin_nb['cells'][5]['outputs']
raw_log = Path(raw_cpu['raw_log']['path']).read_bytes()
assert hashlib.sha256(raw_log).hexdigest() == raw_cpu['raw_log']['sha256']
assert json.loads(raw_log[raw_log.rfind(b'\n{') + 1:]) == raw_cpu['result']
lineage = []
for name in ('notebooks-reader-fix04-cumulative.json', 'notebooks-technical-fix15-cumulative.json'):
    p = ROOT / 'docs/course-revision-20261029-omni/evidence' / name
    r = next(r for r in json.loads(p.read_bytes())['records'] if r['lesson'] == '1.1')
    assert r['executed_sha256'] == raw_cpu_row['executed_sha256']
    lineage.append({'receipt': digest(p), 'row': r})
output = {
    'reviewer_task': '/root/omni_p5_technical_r_1', 'created_at': datetime.now(timezone.utc).isoformat(),
    'command': 'python docs/technical-reviews/artifacts/omni-P5-R.1/verify_r1.py',
    'environment': {'python': sys.version.split()[0], 'platform': platform.platform(), 'device': 'CPU; pure Python builtins; no neural execution'},
    'scope': 'R.1 full exact slice; explicit linked prerequisites; whole-site static notebook/code/output inventory only. No final NN forward, training, model load, network runtime, or publication.',
    'freeze_inputs': [digest(numbered_path), digest(canonical_path), digest(ROOT / entry['snapshot'])],
    'source_slice_matches_current_and_canonical': True, 'assigned_figures': entry['figures_sha256'], 'assigned_code_blocks': 0,
    'examples': examples, 'current_preview_static_inventory': counts,
    'receipt': digest(receipt_path), 'first_lesson_preserved_result': first_preserved,
    'original_1.1_kernel_origin': {'path': str(origin), 'sha256': hashlib.sha256(origin.read_bytes()).hexdigest(),
                                   'bytes': origin.stat().st_size, 'source_and_cell_types_equal': True,
                                   'stdout': text(origin_nb['cells'][5]['outputs'][0]['text'])},
    'original_CPU_command_receipt': {'receipt': digest(raw_cpu_path), 'recorded_at': raw_cpu['recorded_at'],
                                     'command': raw_cpu['command'], 'exit_code': raw_cpu['exit_code'],
                                     'environment': raw_cpu['environment'], 'result': raw_cpu['result'],
                                     'raw_log': {**raw_cpu['raw_log'], 'bytes': len(raw_log), 'hash_checked': True, 'result_checked': True},
                                     'row': raw_cpu_row, 'source_cell_types_and_first_lesson_stdout_match_current': True,
                                     'current_presentation_metadata_not_assumed_identical': True, 'later_reuse_lineage': lineage},
    'prerequisites': [digest(ROOT / first['snapshot']), *prerequisites],
    'navigation_links': links, 'preview_page': digest(PREVIEW / 'course.html'),
    'notebook_download': digest(PREVIEW / first['notebook']),
    'not_verified': ['Authenticated Colab execution', 'MPS/CUDA/Windows runtime', 'Live production website deployment',
                     'Independent semantic rereview of every other lesson; inventory checks source/output presentation only']
}
(OUT / 'verification-v1.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(output, ensure_ascii=False, indent=2))
