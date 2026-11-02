"""Same-owner source22 format equivalence and bounded CI005 policy probes only."""

import ast
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import tempfile
import tokenize
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/tmp/tpb').resolve()
ROUND = ROOT / 'docs/course-revision-20261101-c-v3'
OUT = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-canonical-19-25'
OLD_ROOT = Path('/workspace/work/omni-book-integrated-v1')
RUFF = Path(sys.executable).with_name('ruff')
STARTED = datetime.now(UTC).isoformat()
BLOCKS = re.compile(r'(^```python[ \t]*\n)(.*?)(^```[ \t]*$)', re.M | re.S)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
file_sha = lambda path: sha(Path(path).read_bytes())
load = lambda path: json.loads(Path(path).read_text())
refs = []


def pinned(relative, expected):
    path = ROOT / relative
    actual = file_sha(path)
    assert actual == expected, relative
    refs.append({'path': relative, 'sha256': actual})
    return path


old_freeze_path = pinned('docs/course-revision-20261101-c-v3/source-freeze-v21.json', 'd9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea')
new_freeze_path = pinned('docs/course-revision-20261101-c-v3/source-freeze-v22.json', '0f4e9d63bfcb2243234c17c62242ff79cd4ab640a21d1f313f94c99946a93a59')
equivalence_path = pinned('docs/course-revision-20261101-c-v3/evidence/format-only-source22-equivalence-v1.json', '16a9e24b2a20b62662e060f82fdcd1808f9995c2312a6f6b7bbacaafb2740f33')
kernel_path = pinned('docs/course-revision-20261101-c-v3/evidence/kernel-refresh-source22-v1.json', 'c490237cce379d5e2d47bb4d4fc4103ddd78ce48779f08e1fff51784e620d3ed')
policy_path = pinned('docs/course-revision-20261101-c-v3/.ruff.toml', '8f149fe42b9677cba63337d0898cd1996a453f9d31682f30b6b4939ef6c32fce')
failure_path = pinned('docs/course-revision-20261101-c-v3/evidence/ci-format-fix-author-v2.json', '74390fadff354e33a39ecbdc88f13d946886f10e63a49838c32bc22ecc4e4130')
old_pages = {row['page_id']: row for row in load(old_freeze_path)['pages']}
new_pages = {row['page_id']: row for row in load(new_freeze_path)['pages']}
assert set(old_pages) == set(new_pages)
changed = [key for key in old_pages if old_pages[key] != new_pages[key]]
assert changed == ['19.25'] and len(new_pages) == 370
old, new = old_pages['19.25'], new_pages['19.25']
old_body, new_body = (ROOT / old['snapshot']).read_text(), (ROOT / new['snapshot']).read_text()
assert sha(old_body.encode()) == old['source_sha256'] == '3a68a27b5e0cccc451a2c7c2c8afcfd49f009e9981b09ce6f450bcdca62bfd2b'
assert sha(new_body.encode()) == new['source_sha256'] == 'f356ae70929bd581f13338d2378125db59e509c6eeaf5144d4748d6899c1ad46'
chapter = ROOT / 'course/chapters/19.md'
assert file_sha(chapter) == 'd72c66984bac350585d79a6d21369096b372926d77069b90f1b8e81e15ebe4e5'
assert re.search(r'^## 19\.25 .+?(?=^## |\Z)', chapter.read_text(), re.M | re.S).group() == new_body
assert old['figures_sha256'] == new['figures_sha256'] == {}
old_blocks, new_blocks = list(BLOCKS.finditer(old_body)), list(BLOCKS.finditer(new_body))
assert len(old_blocks) == len(new_blocks) == 2
ast_text = lambda text: ast.dump(ast.parse(text), include_attributes=False)
significant = lambda text: [(token.type, token.string) for token in tokenize.generate_tokens(io.StringIO(text).readline)
                            if token.type not in {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE}]
without_grouping = lambda text: [token for token in significant(text) if token not in {(tokenize.OP, '('), (tokenize.OP, ')')}]
ast_checks = []
for index, (before, after) in enumerate(zip(old_blocks, new_blocks), 1):
    assert ast_text(before[2]) == ast_text(after[2])
    assert without_grouping(before[2]) == without_grouping(after[2])
    removed_grouping_tokens = len(significant(before[2])) - len(significant(after[2]))
    assert removed_grouping_tokens == (0 if index == 1 else 2)
    ast_checks.append({'python_fence': index, 'before_sha256': sha(before[2].encode()), 'after_sha256': sha(after[2].encode()),
                       'AST_sha256': sha(ast_text(after[2]).encode()), 'AST_identical': True,
                       'tokens_except_grouping_parentheses_and_layout_identical': True,
                       'comments_and_literals_identical': True, 'removed_grouping_tokens': removed_grouping_tokens,
                       'specific_change': 'layout only' if index == 1 else 'Ruff also removes one redundant outer parenthesis pair around accepted = (...); AST unchanged'})
masked = lambda text: BLOCKS.sub(lambda match: match[1] + '<PYTHON-BODY>\n' + match[3], text)
assert masked(old_body) == masked(new_body)
all_fences = re.compile(r'^```([^\n]*)\n(.*?)^```[ \t]*$', re.M | re.S)
nonpython = lambda text: [(match[1], match[2]) for match in all_fences.finditer(text) if match[1].strip() != 'python']
assert nonpython(old_body) == nonpython(new_body)

old_nb_path = OLD_ROOT / old['notebook']
new_nb_path = ROOT / new['notebook']
assert file_sha(old_nb_path) == old['notebook_sha256'] == 'c98795a52ea10ec5587067673d423790c5b8e1fd168c2ca983c2110b1e496282'
assert file_sha(new_nb_path) == new['notebook_sha256'] == 'ac0bb7be7f5d2972db2d8bc34a252024f81c152cdc28a6c616da3dd0d20a2aaa'
old_nb, new_nb = load(old_nb_path), load(new_nb_path)
assert old_nb['metadata'] == new_nb['metadata'] and len(old_nb['cells']) == len(new_nb['cells']) == 7
nb_code = []
for index, (before, after) in enumerate(zip(old_nb['cells'], new_nb['cells'])):
    assert before['cell_type'] == after['cell_type']
    if before['cell_type'] == 'markdown':
        assert before == after
    else:
        before_source, after_source = ''.join(before['source']), ''.join(after['source'])
        assert ast_text(before_source) == ast_text(after_source)
        assert without_grouping(before_source) == without_grouping(after_source)
        assert {key: value for key, value in before.items() if key != 'source'} == {key: value for key, value in after.items() if key != 'source'}
        nb_code.append({'cell_index': index, 'AST_identical': True, 'tokens_except_grouping_parentheses_and_layout_identical': True,
                        'removed_grouping_tokens': len(significant(before_source)) - len(significant(after_source))})
assert ''.join(new_nb['cells'][3]['source']) == new_blocks[0][2].rstrip('\n')
assert ''.join(new_nb['cells'][5]['source']) == new_blocks[1][2].rstrip('\n')
kernel = load(kernel_path)
assert kernel['source_freeze'] == {'path': new_freeze_path.relative_to(ROOT).as_posix(), 'sha256': file_sha(new_freeze_path)}
assert len(kernel['actual_executions']) == 1 and kernel['actual_executions'][0]['lesson'] == '19.25'
execution = kernel['actual_executions'][0]
executed_path = ROOT / 'outputs/omni-c-v3-kernel-bundle-v22/19/19.25.ipynb'
assert execution['source_sha256'] == file_sha(new_nb_path)
assert file_sha(executed_path) == execution['executed_sha256'] == '047e3f1c4d761066585aa07b0a89e273c5e67094153d60ebfadeef9de3b40a4b'
executed = load(executed_path)
assert len(executed['cells']) == len(new_nb['cells'])
for source, result in zip(new_nb['cells'], executed['cells']):
    assert source['cell_type'] == result['cell_type'] and source['source'] == result['source']
    assert not any(output.get('output_type') == 'error' for output in result.get('outputs', []))
streams = lambda notebook: [''.join(output['text']) for cell in notebook['cells'] for output in cell.get('outputs', []) if output.get('output_type') == 'stream']
stdout = streams(executed)
assert stdout == ['M與分數形狀 (1, 12, 8) (1, 4)\n候選跟著重排 True\nnone不變 True\n搬動整筆仍查到B True\n', "('record-main', 'record-main', True)\n('record-other', None, False)\n"]
prior_kernel = load(ROOT / 'outputs/omni-c-v3-kernel-bundle-v21/19/19.25.ipynb')
assert streams(prior_kernel) == stdout
old_bundle, new_bundle = ROOT / 'outputs/omni-c-v3-kernel-bundle-v21', ROOT / 'outputs/omni-c-v3-kernel-bundle-v22'
old_files = {path.relative_to(old_bundle).as_posix(): path for path in old_bundle.rglob('*.ipynb')}
new_files = {path.relative_to(new_bundle).as_posix(): path for path in new_bundle.rglob('*.ipynb')}
assert set(old_files) == set(new_files) and len(old_files) == 332
retained = [key for key in old_files if key != '19/19.25.ipynb']
assert all(file_sha(old_files[key]) == file_sha(new_files[key]) for key in retained)
hardlinked = sum((old_files[key].stat().st_dev, old_files[key].stat().st_ino) == (new_files[key].stat().st_dev, new_files[key].stat().st_ino) for key in retained)

original_report_path = ROUND / 'technical-v21/canonical_19_25/original-v1.json'
prior_current_path = ROUND / 'technical-v21/canonical_19_25/current-v1.json'
assert file_sha(original_report_path) == '496fed49905327ddb7b4bb36b0a1060f0c8660194cc7938a0228d7d2f8c392eb'
assert file_sha(prior_current_path) == '607c3679843c7bfcd5d82b38fb9406b9ff078268028b47c3a5097d38fb0096ac'
prior_current = load(prior_current_path)
identity_checks = []
for group in ('sources', 'artifacts'):
    for row in prior_current[group]:
        if 'path' in row:
            assert file_sha(ROOT / row['path']) == row['sha256'], row['path']
            identity_checks.append({'group': group, 'id': row['id'], 'path': row['path'], 'sha256': row['sha256']})
archive_path = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')
assert file_sha(archive_path) == '552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88'
release = load(ROOT / 'docs/decision-product/v3/public-release.json')
assert len(release['implementation']) == 17
assert all(file_sha(ROOT / path) == expected for path, expected in release['implementation'].items())

# Independently inspect the genuine author failure's raw logs; never adopt pass flags.
failure = load(failure_path)['issues'][0]
assert failure['id'] == 'C3-CI-FORMAT-005'
lint_path, format_path = Path(failure['raw_lint']['external_path']), Path(failure['raw_format']['external_path'])
assert file_sha(lint_path) == failure['raw_lint']['sha256']
assert file_sha(format_path) == failure['raw_format']['sha256']
raw_lint = load(lint_path)
raw_format = format_path.read_text()
lint_files = sorted({Path(row['filename']).relative_to(ROOT).as_posix() for row in raw_lint})
format_files = sorted(set(re.findall(r'^\s*-->\s+([^:\n]+):', raw_format, re.M)))
assert len(raw_lint) == 56 and len(lint_files) == 3
assert len(format_files) == 15
canonical_names = {'course/chapters/19.md', 'notebooks/19/19.25.ipynb'}
archived_format_files = set(format_files) - canonical_names
assert len(archived_format_files) == 13 and canonical_names <= set(format_files)
policy = tomllib.loads(policy_path.read_text())
assert (policy_path.parent / policy['extend']).resolve() == ROOT / 'pyproject.toml'
assert policy['extend-exclude'] == ['source-freeze-*','numbered-source-freeze-*','readability-*','technical-*','original-owner-callbacks-*','administrative-*','evidence']
# Seven glob strings, all scoped under this round; no production/course/NB exclusion.
assert len(policy['extend-exclude']) == 7
commands = []
def run(args, text=None, expected=0, max_output=2000):
    result = subprocess.run([str(RUFF), *args], input=text, text=True, capture_output=True, cwd=ROOT)
    assert result.returncode == expected, (args, result.returncode, result.stdout[:1000], result.stderr[:1000])
    commands.append({'command': [str(RUFF), *args], 'stdin_sha256': sha(text.encode()) if text is not None else None,
                     'returncode': result.returncode, 'stdout': result.stdout[:max_output], 'stderr': result.stderr[:max_output],
                     'full_stdout_sha256': sha(result.stdout.encode()), 'full_stderr_sha256': sha(result.stderr.encode())})
    return result

version = run(['--version']).stdout.strip()
discovery = run(['check','--no-cache','--show-files','.'], max_output=0).stdout.splitlines()
discovered = {Path(path).relative_to(ROOT).as_posix() for path in discovery}
assert 'notebooks/19/19.25.ipynb' in discovered
assert set(release['implementation']) <= discovered
assert 'tests/test_course_product_entry.py' in discovered
assert not set(lint_files) & discovered and not archived_format_files & discovered
format_discovery = run(['format','--no-cache','--check','--verbose','.'], max_output=0)
format_paths = {Path(path).relative_to(ROOT).as_posix() for path in re.findall(r'format_path; path=(.+)$', format_discovery.stderr, re.M)}
assert canonical_names <= format_paths
assert set(release['implementation']) <= format_paths and 'tests/test_course_product_entry.py' in format_paths
assert not archived_format_files & format_paths
format_discovery_lines = [line for line in format_discovery.stderr.splitlines() if any(str(ROOT/path) in line for path in canonical_names) and ('Included path' in line or 'format_path;' in line)]
settings = []
for path in ['course/chapters/19.md','notebooks/19/19.25.ipynb','tiny_perceptron/decision_product/service.py','tests/test_course_product_entry.py']:
    result = run(['check','--no-cache','--show-settings',path], max_output=0)
    assert f'Settings path: "{ROOT / "pyproject.toml"}"' in result.stdout
    assert 'formatter.line_width = 120' in result.stdout
    settings.append({'path':path,'config': 'pyproject.toml','line_width':120,'stdout_sha256':sha(result.stdout.encode())})
for path, bad, good in [
    ('course/chapters/19.md', '# Fixture\n\n```python\nx=  1\n```\n', '# Fixture\n\n```python\nx = 1\n```\n'),
    ('notebooks/19/19.25.ipynb', json.dumps({'cells':[{'cell_type':'code','metadata':{},'source':['x=  1'],'execution_count':None,'outputs':[]}], 'metadata':{},'nbformat':4,'nbformat_minor':5}, indent=1)+'\n', json.dumps({'cells':[{'cell_type':'code','metadata':{},'source':['x = 1'],'execution_count':None,'outputs':[]}], 'metadata':{},'nbformat':4,'nbformat_minor':5}, indent=1)+'\n'),
    ('tiny_perceptron/decision_product/_format_fixture.py', 'x=  1\n', 'x = 1\n'),
    ('tests/_format_fixture.py', 'x=  1\n', 'x = 1\n'),
]:
    run(['format','--no-cache','--check','--stdin-filename',path,'-'],bad,expected=1)
    run(['format','--no-cache','--check','--stdin-filename',path,'-'],good)
for path in ['tiny_perceptron/decision_product/_lint_fixture.py','tests/_lint_fixture.py']:
    result = run(['check','--no-cache','--output-format','json','--stdin-filename',path,'-'], 'print(undefined_format_policy_fixture)\n', expected=1)
    assert any(row['code'] == 'F821' for row in json.loads(result.stdout))
with tempfile.TemporaryDirectory(prefix='format-policy-probe-', dir=ROUND) as directory:
    directory = Path(directory)
    frozen, active = directory / 'technical-frozen-fixture', directory / 'tools'
    frozen.mkdir(); active.mkdir()
    (frozen / 'probe.py').write_text('print(undefined_format_policy_fixture)\n')
    (active / 'probe.py').write_text('print(undefined_format_policy_fixture)\n')
    relative = directory.relative_to(ROOT).as_posix()
    visible = run(['check','--no-cache','--show-files',relative]).stdout
    assert str(active / 'probe.py') in visible and str(frozen / 'probe.py') not in visible
    result = run(['check','--no-cache','--output-format','json',relative], expected=1)
    errors = json.loads(result.stdout)
    assert len(errors) == 1 and errors[0]['code'] == 'F821' and errors[0]['filename'] == str(active / 'probe.py')
    fixture = active / 'fixture.md'
    before = '# Format policy fixture\n\n```python\nx=  1\n```\n\nProse stays here.\n'
    fixture.write_text(before)
    formatted = subprocess.run([sys.executable,str(ROOT/'scripts/format_course_code.py'),str(fixture)],text=True,capture_output=True,cwd=ROOT,check=True)
    after = fixture.read_text()
    assert 'x = 1\n' in after and masked(before) == masked(after)
    assert ast_text(BLOCKS.search(before)[2]) == ast_text(BLOCKS.search(after)[2])
    commands.append({'command':[sys.executable,'scripts/format_course_code.py',str(fixture)],'returncode':formatted.returncode,'stdout':formatted.stdout,'stderr':formatted.stderr,'result':'only temporary fixture changed; prose and AST preserved'})
# Actual CI gate invocations are read-only; this is a format-specific check, not full CI.
run(['check','--no-cache','.'])
assert file_sha(chapter) == 'd72c66984bac350585d79a6d21369096b372926d77069b90f1b8e81e15ebe4e5'
assert file_sha(new_nb_path) == new['notebook_sha256']
assert file_sha(original_report_path) == '496fed49905327ddb7b4bb36b0a1060f0c8660194cc7938a0228d7d2f8c392eb'
assert file_sha(prior_current_path) == '607c3679843c7bfcd5d82b38fb9406b9ff078268028b47c3a5097d38fb0096ac'
receipt = {
    'schema_version':1,'reviewer_task':'/root/c_v3_technical_coord/canonical_19_25','started_at':STARTED,'completed_at':datetime.now(UTC).isoformat(),
    'command':'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/tmp/tpb UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-19-25/source22-check.py',
    'environment':{'python':sys.version.split()[0],'platform':platform.platform(),'device':'cpu; static AST/JSON/hash and Ruff only','ruff':version},
    'scope':'Same-owner format-only source22 callback and C3-CI-FORMAT-005 companion. No training, model inference, new kernel execution, final opening/inference, network, publish, stage or commit. Existing source22 kernel is inspected, not rerun; prior scientific/local CPU scope retained.',
    'pinned_inputs':refs,'source22_equivalence':{'changed_pages':changed,'retained_complete_page_records':369,'python_fences':ast_checks,'masked_prose_sha256':sha(masked(new_body).encode()),'all_prose_and_nonpython_bytes_identical':True,'nonpython_fence_count':len(nonpython(new_body)),'full_chapter_sha256':file_sha(chapter)},
    'notebook_equivalence':{'prior_external_path':str(old_nb_path),'prior_sha256':file_sha(old_nb_path),'current_path':new['notebook'],'current_sha256':file_sha(new_nb_path),'code_cells':nb_code,'markdown_cells_byte_identical':True,'metadata_and_non_source_cell_fields_identical':True,'two_canonical_python_fences_match_cells':[3,5],'fence_to_cell_normalization':'Notebook generator removes only the final newline from each code cell; all other code bytes match current canonical fences.'},
    'kernel_inspection':{'receipt':{'path':kernel_path.relative_to(ROOT).as_posix(),'sha256':file_sha(kernel_path)},'executed_external_path':str(executed_path),'executed_sha256':file_sha(executed_path),'executed_inputs_match_current_notebook':True,'errors':0,'numerical_stdout':stdout,'numerical_stdout_same_as_source21':True,'actual_execution_record':execution,'bundle_count':332,'retained_notebook_hashes':len(retained),'retained_hardlinks':hardlinked},
    'prior_evidence_identity':{'original':{'path':original_report_path.relative_to(ROOT).as_posix(),'sha256':file_sha(original_report_path)},'current_v1':{'path':prior_current_path.relative_to(ROOT).as_posix(),'sha256':file_sha(prior_current_path)},'native_sources_artifacts':identity_checks,'qualified17_unchanged':True,'qualified_zip_sha256':file_sha(archive_path)},
    'format_policy':{'config':{'path':policy_path.relative_to(ROOT).as_posix(),'sha256':file_sha(policy_path)},'extends':'pyproject.toml','exclude_patterns':policy['extend-exclude'],'actual_ruff_binary':str(RUFF),'ruff_binary_sha256':file_sha(RUFF),'raw_failure':{'author_receipt':failure_path.relative_to(ROOT).as_posix(),'lint_raw_external_path':str(lint_path),'lint_raw_sha256':file_sha(lint_path),'lint_count':len(raw_lint),'lint_files':lint_files,'format_raw_external_path':str(format_path),'format_raw_sha256':file_sha(format_path),'format_file_count':len(format_files),'archived_format_file_count':len(archived_format_files),'format_files':format_files},'actual_discovery':{'lint_files':len(discovered),'format_files':len(format_paths),'canonical_markdown_and_notebook_in_real_formatter_trace':format_discovery_lines,'all_17_product_sources_present':True,'product_entry_test_present':True,'archived_finding_files_absent':True,'note':'Ruff check --show-files lists lintable sources (699 here) and excludes Markdown from lint; real format --check --verbose supplies distinct formatter discovery (1940 here), including canonical Markdown and Notebook.'},'effective_settings':settings,'positive_negative_fixtures':'canonical Markdown/Notebook/product/test formatting rejects bad and accepts good; product/test undefined-name lint fails; round tools sibling lint fails while immutable-pattern sibling is excluded','canonical_and_original_files_not_modified':True,'full_root_lint_and_format_check_exit0':True},
    'commands':commands,
}
with (OUT/'source22-execution.json').open('x',encoding='utf-8') as handle:
    json.dump(receipt,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
print(json.dumps({'artifact':str(OUT/'source22-execution.json'),'changed_pages':changed,'AST_identical':True,'prose_identical':True,'retained_kernels':len(retained),'format_policy':'pass','ruff':version,'command_count':len(commands)},ensure_ascii=False))
