"""Bounded notebook metadata / strict pin fixture controls, not an actual site or Windows run."""

import ast
import contextlib
import copy
import hashlib
import io
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from scripts import check_site, export_course

ROOT = Path('/tmp/tpb')
ART = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-windows-ci-followup'
before = ROOT / 'docs/technical-reviews/artifacts/c-v3-windows-colab-fixture-fix/before-test_course_export.py'
current = ROOT / 'tests/test_course_export.py'
old_tree, tree = ast.parse(before.read_text()), ast.parse(current.read_text())
excluded = {'pinned_notebooks', 'test_formal_colab_rejects_changed_worktree_newlines'}
old_rest = [node for node in old_tree.body if getattr(node, 'name', None) not in excluded]
current_rest = [node for node in tree.body if getattr(node, 'name', None) not in excluded]
assert ast.dump(ast.Module(old_rest, [])) == ast.dump(ast.Module(current_rest, []))
fixture = next(node for node in tree.body if getattr(node, 'name', None) == 'pinned_notebooks')
assert 'params=["false", "true"]' in ast.get_source_segment(current.read_text(), fixture.decorator_list[0])

notebook = {
    'metadata': {'lesson_id': '1.1', 'reviewed': True},
    'cells': [
        {'cell_type': 'code', 'source': ['import math\n'], 'metadata': {'course_setup': True}},
        {'cell_type': 'code', 'source': ["print('REVIEWED_CODE')\n"], 'metadata': {}},
    ],
}
original_before = copy.deepcopy(notebook)
executed = copy.deepcopy(notebook)
executed['metadata'] = {'lesson_id': 'wrong'}
executed['cells'][0]['metadata'] = {}
executed['cells'][1]['metadata'] = {'course_setup': True, 'course_figure': 'unreviewed.svg'}
for cell in executed['cells']:
    cell.update(execution_count=2, outputs=[{'output_type': 'stream', 'text': 'actual\n'}])

with tempfile.TemporaryDirectory(prefix='c3-windows-owner-controls-', dir='/tmp') as directory:
    root = Path(directory)
    path = root / 'executed.ipynb'
    path.write_bytes(json.dumps(executed).encode('utf-8'))
    checked = export_course.checked_notebook(notebook, path)
    assert notebook == original_before
    assert checked['metadata'] == notebook['metadata']
    assert [cell['metadata'] for cell in checked['cells']] == [cell['metadata'] for cell in notebook['cells']]
    assert [cell['source'] for cell in checked['cells']] == [cell['source'] for cell in notebook['cells']]
    assert checked['cells'][1]['outputs'] == executed['cells'][1]['outputs']
    checked['cells'][1]['outputs'][0]['text'] = 'changed returned output'
    assert executed['cells'][1]['outputs'][0]['text'] == 'actual\n'
    assert sum(c['cell_type'] == 'code' and not c.get('metadata', {}).get('course_setup') and 'course_figure' not in c.get('metadata', {}) for c in checked['cells']) == 1

    site = root / 'site'
    site.mkdir()
    (root / 'course').mkdir()
    (root / 'course/lesson-index.json').write_text(json.dumps([{'id': '1.1', 'notebook': 'notebooks/01/1.1.ipynb'}]))
    for base in [root, site]:
        target = base / 'notebooks/01/1.1.ipynb'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(notebook))
    (site / 'build-info.json').write_text(json.dumps({'lessons': 1, 'builder': 'zensical', 'builder_version': 'fixture', 'colab_mode': 'upload', 'publication_ready': False}))
    (site / 'search.json').write_text(json.dumps({'items': [{'location': '1.1.html'}]}))
    html = '<meta name="generator" content="zensical-fixture"><div id="1.1"></div><div id="colab-upload-instructions"></div><a href="notebooks/01/1.1.ipynb">Download</a><a href="https://colab.research.google.com/">Upload</a><pre class="language-python highlight">REVIEWED_CODE</pre>'
    page = site / '1.1.html'
    page.write_text(html)
    with patch.object(check_site, 'ROOT', root):
        with contextlib.redirect_stdout(io.StringIO()):
            check_site.check(site)
        page.write_text(html.replace('class="language-python highlight"', 'class="removed-code"'))
        try:
            check_site.check(site)
        except ValueError as error:
            assert '程式區塊遺失' in str(error)
        else:
            raise AssertionError('missing visible code accepted')
        page.write_text(html)
        corrupt = copy.deepcopy(notebook)
        corrupt['cells'][1]['metadata'] = {'course_setup': True}
        (site / 'notebooks/01/1.1.ipynb').write_text(json.dumps(corrupt))
        try:
            check_site.check(site)
        except ValueError as error:
            assert '下載與目前教材不同' in str(error)
        else:
            raise AssertionError('unreviewed downloaded metadata accepted')

author = json.loads((ROOT / 'docs/course-revision-20261101-c-v3/evidence/windows-ci-test-fixture-author-v1.json').read_text())
for row in author['current_stable_files']:
    assert hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256']
result = {
    'created_at': datetime.now(UTC).isoformat(), 'reviewer_task': '/root/omni_p5_full_tools_review',
    'all_previous_test_functions_and_assertions_except_fixture_AST_unchanged': True,
    'fixture_autocrlf_both_false_true_explicit': True,
    'checked_notebook_preserves_original_root_and_cell_metadata_and_source': True,
    'only_execution_count_outputs_merged_and_outputs_deep_copied': True,
    'original_notebook_unmodified': True,
    'private_site_control_positive': True,
    'private_site_missing_code_rejected': True,
    'private_site_unreviewed_downloaded_metadata_rejected': True,
    'all5_stable_file_hashes_match_author_frozen_map': True,
    'scope': 'Private small synthetic notebook/HTML control; not a Zensical build, model, published site or Windows run.'
}
with (ART / 'owner-stable-controls-result-v1.json').open('x') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print('PASS: old test assertions intact; original notebook metadata/source retained; only count/outputs merged; synthetic site positive passes and both missing-code/metadata-swap negatives reject; stable5 SHA match.')
