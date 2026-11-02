"""Bounded source22 collector callback: identity/AST/one saved kernel, no execution of lessons."""

import ast
import difflib
import hashlib
import importlib.util
import io
import json
import platform
import re
import tokenize
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/tmp/tpb').resolve()
ROUND = 'docs/course-revision-20261101-c-v3/'
BASE = 'docs/technical-reviews/artifacts/c-v3-v21-canonical-course/'
OUT = ROOT / (BASE + 'source22-bounded-delta-v1.json')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path.read_bytes())}


def load(reference):
    raw = (ROOT / reference['path']).read_bytes()
    assert sha(raw) == reference['sha256'], reference
    return json.loads(raw)


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


record = {'started_at': datetime.now(UTC).isoformat(), 'actor': '/root/c_v3_technical_coord/canonical_course', 'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'CPU static identities/AST/saved output only', 'torch': 'not imported; zero NN or kernel execution'}, 'scope': 'Source22 format-only collector binder callback. Preserve genuine own original/source21 current and all existing scientific/786-raw/authority/three CPU/GET checks; independently inspect only changed source22 bindings, AST-equivalent19.25, its one new saved CPU output, and same-owner W7 current-v3 inside exact new final26.'}
original_ref = {'path': ROUND + 'technical-v21/canonical_course/original-v1.json', 'sha256': 'b239f8647639af0d9949de0e3772f41fe4aa86d8259aa9b517c551f1e5df048c'}
prior_ref = {'path': ROUND + 'technical-v21/canonical_course/current-v1.json', 'sha256': '6e6a2f345b2887154c24fa09bc2f17f1a3bb08ea49d1e4f12211ff18bb92d89d'}
prior_receipt_ref = {'path': ROUND + 'technical-coordination-v21/receipts/canonical-course-current-v1.json', 'sha256': 'f542e56686ffe524e5149312ffc416f92341b84742ce80338f63df5dba5aae25'}
original, prior, receipt = load(original_ref), load(prior_ref), load(prior_receipt_ref)
assert receipt['report'] == prior_ref and receipt['received_at'] >= prior['created_at']
record['preserved_own_basis'] = {'original': original_ref, 'source21_current': prior_ref, 'source21_actual_receipt': prior_receipt_ref, 'actually_received_at': receipt['received_at'], 'original_issue_resolutions_preserved': len(prior['original_issue_resolutions'])}
f21_ref = {'path': ROUND + 'source-freeze-v21.json', 'sha256': 'd9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea'}
f22_ref = {'path': ROUND + 'source-freeze-v22.json', 'sha256': '0f4e9d63bfcb2243234c17c62242ff79cd4ab640a21d1f313f94c99946a93a59'}
format_ref = {'path': ROUND + 'evidence/format-only-source22-equivalence-v1.json', 'sha256': '16a9e24b2a20b62662e060f82fdcd1808f9995c2312a6f6b7bbacaafb2740f33'}
f21, f22, root_format = load(f21_ref), load(f22_ref), load(format_ref)
p21 = {row['page_id']: row for row in f21['pages']}
p22 = {row['page_id']: row for row in f22['pages']}
assert set(p21) == set(p22) and len(p21) == 370
changed = [key for key in p21 if p21[key] != p22[key]]
assert changed == ['19.25'] and p21['course'] == p22['course']
assert sha((ROOT / 'course/README.md').read_bytes()) == prior['source_sha256'] == p22['course']['source_sha256']
old = (ROOT / p21['19.25']['snapshot']).read_bytes()
new = (ROOT / p22['19.25']['snapshot']).read_bytes()
assert sha(old) == p21['19.25']['source_sha256'] and sha(new) == p22['19.25']['source_sha256']
pattern = re.compile(r'(^```python\s*\n)(.*?)(^```[ \t]*$)', re.M | re.S)
old_matches, new_matches = list(pattern.finditer(old.decode())), list(pattern.finditer(new.decode()))
assert len(old_matches) == len(new_matches) == 2
ast_rows = []
ignored = {tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
for i, (left, right) in enumerate(zip(old_matches, new_matches)):
    ast_left = ast.dump(ast.parse(left[2]), include_attributes=False)
    ast_right = ast.dump(ast.parse(right[2]), include_attributes=False)
    assert ast_left == ast_right
    tokens = lambda text: [(tok.type, tok.string) for tok in tokenize.generate_tokens(io.StringIO(text).readline) if tok.type not in ignored]
    token_left, token_right = tokens(left[2]), tokens(right[2])
    changes = []
    for tag, a, b, c, d in difflib.SequenceMatcher(a=token_left, b=token_right).get_opcodes():
        if tag != 'equal':
            changes.append({'kind': tag, 'old': token_left[a:b], 'new': token_right[c:d]})
    ast_rows.append({'python_block': i + 1, 'old_code_sha256': sha(left[2].encode()), 'new_code_sha256': sha(right[2].encode()), 'AST_identical': True, 'AST_sha256': sha(ast_left.encode()), 'significant_token_stream_identical': token_left == token_right, 'significant_token_changes': changes})
assert ast_rows[0]['significant_token_stream_identical'] is True
assert ast_rows[1]['significant_token_stream_identical'] is False
assert [x['old'] for x in ast_rows[1]['significant_token_changes']] == [[(tokenize.OP, '(')], [(tokenize.OP, ')')]]
outside_old = pattern.sub(lambda m: m[1] + '<PYTHON BODY>\n' + m[3], old.decode())
outside_new = pattern.sub(lambda m: m[1] + '<PYTHON BODY>\n' + m[3], new.decode())
assert outside_old == outside_new
chapter = (ROOT / 'course/chapters/19.md').read_text()
headings = list(re.finditer(r'^## (19\.\d+) .+$', chapter, re.M))
assert len(headings) == 25
reconstructed = chapter[:headings[0].start()] + ''.join((ROOT / p21[match[1]]['snapshot']).read_text() for match in headings)
probe21 = json.loads((ROOT / (BASE + 'probe-course-v1.json')).read_text())
prior_chapter_sha = next(row['sha256'] for row in probe21['navigation']['links'] if row['path'] == 'course/chapters/19.md')
assert sha(reconstructed.encode()) == prior_chapter_sha == '6fb67227641d879216eaf5506763552a6a3173a106140c8f0ff94b700edfd61a'
assert (chapter[:headings[-1].start()] + old.decode()) == reconstructed
assert chapter[headings[-1].start():].encode() == new
record['format_delta'] = {'canonical21': f21_ref, 'canonical22': f22_ref, 'root_raw_observation': format_ref, 'changed_canonical_ids': changed, 'other369_full_page_rows_identical': True, 'course_full_row_identical': True, 'course_sha256': prior['source_sha256'], 'source21_19_25': ref(ROOT / p21['19.25']['snapshot']), 'source22_19_25': ref(ROOT / p22['19.25']['snapshot']), 'source21_chapter19_reconstructed_sha256': prior_chapter_sha, 'current_chapter19': ref(ROOT / 'course/chapters/19.md'), 'only19_25_changed_in_chapter19': True, 'all_prose_nonpython_bytes_identical': True, 'python_blocks': ast_rows, 'raw_delta': '\n'.join(difflib.unified_diff(old.decode().splitlines(), new.decode().splitlines(), n=1)), 'interpretation': 'AST-equivalent formatting, including deletion of one redundant outer parenthesis pair around accepted; not a claim of exact token or whitespace-only identity.'}

kernel_ref = {'path': ROUND + 'evidence/kernel-refresh-source22-v1.json', 'sha256': 'c490237cce379d5e2d47bb4d4fc4103ddd78ce48779f08e1fff51784e620d3ed'}
kernel = load(kernel_ref)
assert kernel['source_freeze'] == f22_ref and kernel['retained_current_count'] == 331 and len(kernel['actual_executions']) == 1
execution = kernel['actual_executions'][0]
source_path = ROOT / p22['19.25']['notebook']
output_path = ROOT / 'outputs/omni-c-v3-kernel-bundle-v22/19/19.25.ipynb'
old_output_path = ROOT / 'outputs/omni-c-v3-kernel-bundle-v21/19/19.25.ipynb'
assert sha(source_path.read_bytes()) == execution['source_sha256'] == p22['19.25']['notebook_sha256']
assert sha(output_path.read_bytes()) == execution['executed_sha256'] == '047e3f1c4d761066585aa07b0a89e273c5e67094153d60ebfadeef9de3b40a4b'
source_nb, output_nb, old_output_nb = [json.loads(path.read_text()) for path in [source_path, output_path, old_output_path]]
cells = lambda nb: [(cell['cell_type'], cell['source']) for cell in nb['cells']]
assert cells(source_nb) == cells(output_nb)
errors = [out for cell in output_nb['cells'] for out in cell.get('outputs', []) if out.get('output_type') == 'error']
assert not errors
stdout = lambda nb: ''.join(''.join(out['text']) if isinstance(out['text'], list) else out['text'] for cell in nb['cells'] for out in cell.get('outputs', []) if out.get('output_type') == 'stream' and out.get('name') == 'stdout')
assert stdout(output_nb) == stdout(old_output_nb)
assert '候選跟著重排 True' in stdout(output_nb) and "('record-other', None, False)" in stdout(output_nb)
record['one_new_saved_kernel'] = {'tracked_raw_receipt': kernel_ref, 'raw_execution': execution, 'source': ref(source_path), 'output': ref(output_path), 'source_cell_types_and_bytes_exact': True, 'error_outputs': 0, 'stdout': stdout(output_nb), 'stdout_equal_prior_saved_source21_example': True, 'retained331_basis': 'Source21 own retention verified332 saved sources/outputs. Source22 changes only19.25 source/Notebook; other369 canonical full rows identical and root actual receipt retains331. This callback inspects only19.25 saved source/output, and executes zero kernels.', 'new_kernels_executed_by_this_actor': 0, 'new_neural_product_or_scientific_measurement': False}

old_bundle_ref = {'path': 'docs/technical-reviews/artifacts/c-v3-v21-numbered/numbered-section-reviews-final-v1.json', 'sha256': '6b0849e632c43964359de7deb0f2abf30cb6224cea42de0003f6f8efbfd69f04'}
new_bundle_ref = {'path': 'docs/technical-reviews/artifacts/c-v3-v21-numbered/numbered-section-reviews-source22-final-v1.json', 'sha256': 'd103c2981757718d0eb2869142ce415984ab754aa7edc59697373a6b8ca5b1fb'}
old_bundle, new_bundle = load(old_bundle_ref), load(new_bundle_ref)
sections21, sections22 = load(old_bundle['section_freeze']), load(new_bundle['section_freeze'])
assert new_bundle['canonical_source_freeze'] == f22_ref
assert new_bundle['section_freeze']['sha256'] == '3f2671acc39c31215473924da29987c80ac444565184634dda9c8e033fd291c6'
sections21 = {row['page_id']: row for row in sections21['pages']}
sections22 = {row['page_id']: row for row in sections22['pages']}
assert set(sections21) == set(sections22) and len(sections21) == 26
assert all(sections21[k]['source_sha256'] == sections22[k]['source_sha256'] and sections21[k]['figures_sha256'] == sections22[k]['figures_sha256'] for k in sections21)
old_entries = {row['lesson_id']: row for row in old_bundle['sections']}
new_entries = {row['lesson_id']: row for row in new_bundle['sections']}
changed_entries = [key for key in old_entries if old_entries[key] != new_entries[key]]
assert changed_entries == ['W.7']
reports, owners = {}, defaultdict(list)
for key, entry in new_entries.items():
    assert entry['original_report'] == old_entries[key]['original_report']
    assert entry['original_section_freeze'] == old_entries[key]['original_section_freeze']
    assert entry['original_collection'] == old_entries[key]['original_collection']
    load(entry['original_report'])
    report = load(entry['report'])
    reports[key] = report
    owners[report['reviewer_task']].append(key)
assert len(owners) == 26
validator = module('course22_validator', 'scripts/check_technical_reviews.py')
setup = module('course22_setup', 'scripts/omni_revision/review_setup.py')
setup.ROOT = ROOT
rows = []
for key, report in reports.items():
    page = sections22[key]
    errors = validator._validate(ROOT, ROOT / page['source'], key, setup.frozen_body(page), report, owners, set())
    assert not errors, (key, errors)
    rows.append({'lesson_id': key, 'report': new_entries[key]['report'], 'same_entry_as_source21': key != 'W.7', 'original_and_original_receipt_retained': True, 'native_current_errors': errors})
w7_receipt_ref = {'path': ROUND + 'technical-coordination-v21/receipts/w7-current-v3.json', 'sha256': '161129b35252f96c891615e2403cdb9723c7afc8fc1e895a6e13f1a9b50b41f6'}
w7_receipt = load(w7_receipt_ref)
w7 = reports['W.7']
assert w7_receipt['report'] == new_entries['W.7']['report']
assert w7_receipt['received_at'] == new_entries['W.7']['current_collection']['received_at']
assert w7_receipt['reviewer_task'] == w7['reviewer_task'] == '/root/omni_p6_technical_w7'
assert w7['created_at'] < w7_receipt['received_at']
w7_delta = next(item for item in w7['artifacts'] if item['id'] == 'a3-source22')
load(w7_delta)
assert 'one redundant outer parenthesis pair' in w7['current_basis']
index = load(new_bundle['issue_inventory'])
for item in index['reports']:
    load(item)
for item in index['issues']:
    issue_report = load(item['report'])
    assert int(item['json_pointer'].rsplit('/', 1)[1]) < len(issue_report['issues'])
record['source22_final26_binding'] = {'source21_bundle': old_bundle_ref, 'source22_bundle': new_bundle_ref, 'section_freeze22': new_bundle['section_freeze'], 'canonical_freeze22': f22_ref, 'all26_body_and_figure_identities_unchanged': True, 'changed_entries': changed_entries, 'other25_entries_exactly_unchanged': True, 'original_and_original_receipts_all_unchanged': True, 'rows': rows, 'W7_same_original_owner': w7['reviewer_task'], 'W7_current_created_at': w7['created_at'], 'W7_current_actual_receipt': w7_receipt_ref, 'W7_actually_received_at': w7_receipt['received_at'], 'W7_raw_delta_artifact': {'path': w7_delta['path'], 'sha256': w7_delta['sha256']}, 'W7_scope_read': w7['current_basis'], 'retained_scopes': '21 immutable retained entries +same-owner W7 bounded callback+four source21 fresh originals/current receipts retained; not26 new readings', 'issue_inventory': new_bundle['issue_inventory'], 'index_report_refs_checked': len(index['reports']), 'index_issue_pointers_checked': len(index['issues']), 'not_global_T_or_C_seal': True}

# The qualified predictor is unchanged. No network, NN, raw786 or paper reread.
descriptor = json.loads((ROOT / 'docs/decision-product/v3/public-release.json').read_text())
assert ref(ROOT / 'docs/decision-product/v3/public-release.json') == probe21['descriptor']
for item in probe21['implementation_identity']:
    assert sha((ROOT / item['path']).read_bytes()) == item['sha256']
record['scientific_retention'] = {'source21_own_probe': ref(ROOT / (BASE + 'probe-course-v1.json')), 'source21_own_retention': ref(ROOT / (BASE + 'retention-course-v1.json')), '17_product_implementation_identities_unchanged': True, 'release_descriptor_identity_unchanged': True, 'qualified_ZIP_identity_retained': {'sha256': descriptor['sha256'], 'bytes': descriptor['bytes']}, 'no_new_download_or_CPU_product_run': True, 'no_repeat786_raw_or_mature_source_read': True, 'no_reading_time_dependency_added': True, 'original12_issue_resolutions_and_scientific_limits_retained': True}
record['completed_at'] = datetime.now(UTC).isoformat()
with OUT.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'artifact': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT.read_bytes()), 'changed_canonical': changed, 'course_unchanged': True, 'python_AST_identical': [row['AST_identical'] for row in ast_rows], 'token_stream_identical': [row['significant_token_stream_identical'] for row in ast_rows], 'new_kernel_output_sha256': execution['executed_sha256'], 'changed_final26_entry': changed_entries, 'all26_native_current_errors_empty': True, 'completed_at': record['completed_at']}, ensure_ascii=False, indent=2))
