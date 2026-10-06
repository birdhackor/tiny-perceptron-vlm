"""Recheck the changed 1.1 dependency only; no training, installation, or full-course run."""
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
current_path = ROOT / 'notebooks/01/1.1.ipynb'
old_path = OUT / 'inputs/notebooks/01/1.1.ipynb'
history_path = ROOT / 'docs/course-revision-20261006-phase6/history/before-final-continuity-and-format-repairs/notebooks/01/1.1.ipynb'
current = json.loads(current_path.read_bytes())
old = json.loads(old_path.read_bytes())
assert digest(old_path) == digest(history_path)
assert len(current['cells']) == len(old['cells']) == 7
assert current['metadata'] == old['metadata']
all_code_unchanged = []
for number, (before, after) in enumerate(zip(old['cells'], current['cells'])):
    assert before['cell_type'] == after['cell_type']
    if after['cell_type'] == 'code':
        assert before['source'] == after['source']
        all_code_unchanged.append({'cell': number,
            'source_sha256': hashlib.sha256(''.join(after['source']).encode()).hexdigest(),
            'unchanged': True})
expected = copy.deepcopy(old)
assert expected['cells'][3]['metadata']['course_figure']['alt'] == '原句每字與固定編號的對應，並可查迴文字'
expected['cells'][3]['metadata']['course_figure']['alt'] = '原句每字與固定編號的對應，並可查回文字'
expected['cells'][6]['source'] = [line.replace('透過時沒有額外輸出', '通過時沒有額外輸出')
                                 for line in expected['cells'][6]['source']]
assert expected == current

# This is an existing raw nbclient output, not a new kernel run performed by this reviewer.
executed_path = ROOT / 'outputs/notebooks/01/1.1.ipynb'
executed_saved = OUT / 'dependency-existing-kernel-1.1.ipynb'
shutil.copyfile(executed_path, executed_saved)
assert digest(executed_path) == digest(executed_saved)
executed = json.loads(executed_saved.read_bytes())
assert len(executed['cells']) == len(current['cells'])
executed_summary = []
for number, (source, result) in enumerate(zip(current['cells'], executed['cells'])):
    assert source['cell_type'] == result['cell_type']
    assert ''.join(source['source']) == ''.join(result['source'])
    if result['cell_type'] == 'code':
        assert result.get('execution_count') is not None
        outputs = result.get('outputs', [])
        assert not any(o['output_type'] == 'error' for o in outputs)
        executed_summary.append({'cell': number, 'execution_count': result['execution_count'],
            'stream': ''.join(''.join(o.get('text', '')) for o in outputs if o['output_type'] == 'stream'),
            'output_types': [o['output_type'] for o in outputs],
            'execution_metadata': result.get('metadata', {}).get('execution', {})})
assert [x['execution_count'] for x in executed_summary] == [1, 2, 3]
assert executed_summary[-1]['stream'].splitlines()[-1] == '貓看狗，狗看貓。'

# Independently execute just the unchanged deterministic exercise in fresh namespaces.
main = ''.join(current['cells'][5]['source'])
computations = []
for text in ['貓看狗，狗看貓。', '鳥看狗，狗看鳥。']:
    source = main.replace('text = "貓看狗，狗看貓。"', f'text = "{text}"')
    namespace = {}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(source, str(current_path) + ':cell5', 'exec'), namespace)
    observed = stream.getvalue().splitlines()
    assert len(observed) == 3  # A passing assert adds no output.
    assert observed[-1] == text
    assert namespace['restored'] == namespace['text']
    computations.append({'input': text, 'stdout': stream.getvalue(),
        'assert_passed': True, 'fresh_namespace': True, 'new_jupyter_kernel': False})

report = json.loads((OUT / 'rechecked-report.json').read_bytes())
unchanged_sources = []
for source in report['sources']:
    if source['kind'] == 'repository_code' and source['id'] != 's_notebook':
        actual = digest(ROOT / source['path'])
        assert actual == source['sha256'], source['path']
        unchanged_sources.append({'path': source['path'], 'sha256': actual})
for figure, expected_hash in report['figure_sha256'].items():
    assert digest(ROOT / figure) == expected_hash
raw = (ROOT / 'course/first-steps.md').read_bytes()
start = raw.index(b'## W.1 ')
end = raw.index(b'## W.2 ', start)
assert hashlib.sha256(raw[start:end]).hexdigest() == report['source_sha256']
assert hashlib.sha256(raw[:start]).hexdigest() == report['intro_sha256']
results = {'reviewer_task': '/root/p6_fact_w1', 'current_notebook_sha256': digest(current_path),
    'prior_notebook_sha256': digest(old_path), 'history_input_sha256': digest(history_path),
    'historical_input_equals_reviewer_initial': True, 'exact_changes': [
        {'cell': 3, 'field': 'metadata/course_figure/alt', 'before': '查迴', 'after': '查回'},
        {'cell': 6, 'field': 'markdown/assert prose', 'before': '透過時', 'after': '通過時'}],
    'all_code_cell_bytes_unchanged': all_code_unchanged,
    'existing_kernel_output': {'path': str(executed_path.relative_to(ROOT)),
        'permanent_copy': str(executed_saved.relative_to(ROOT)), 'sha256': digest(executed_saved),
        'owner_executed_this_kernel_run': False, 'inspection': executed_summary},
    'owner_short_cpu_exercise_runs': computations, 'unchanged_repository_sources': unchanged_sources,
    'source_sha256': report['source_sha256'], 'intro_sha256': report['intro_sha256'],
    'figure_unchanged': True, 'source_and_intro_unchanged': True,
    'environment': {'python': sys.version.split()[0], 'torch': importlib.metadata.version('torch'),
        'device': 'CPU/Python only', 'new_kernel_runs_by_owner': '0'},
    'supports': 'W.1 bootstrap, Tiny Perceptron entrypoint, original/modified exact restoration, '
        'silent passing assertion, and unchanged SVG remain supported. No new model or whole-course claim.'}
(OUT / 'notebook-dependency-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(results, ensure_ascii=False, indent=2))
