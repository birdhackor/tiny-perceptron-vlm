"""Prompted historical patch recheck, actual saved instances and small controls."""

import hashlib
import importlib.util
import json
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parent.parent
PROOFS = Path(__file__).resolve().parent
REQUEST = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/historical-four-tool-request-v1.json'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest(raw), 'bytes': len(raw)}


def load(reference):
    raw = (ROOT / reference['path']).read_bytes()
    assert digest(raw) == reference['sha256'], reference['path']
    return json.loads(raw)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


assert ref(REQUEST)['sha256'] == '57b4742a61cc6846f2a89d8e22808d231fcda14f8a35a06305b6c9f6aa8b1da7'
request = json.loads(REQUEST.read_bytes())
api = module('historical_four_actual_incremental', ROOT / 'docs/review-tools/incremental_revision.py')
history = module('historical_four_actual_recheck', ROOT / 'docs/review-tools/historical_fix_rechecks.py')
first_full = load(request['current_full_scope_report'])
assert len(first_full['checked_files']) == 460
current_checked = {row['path']: row['sha256'] for row in first_full['checked_files']}
target_files = request['issues'][0]['original_disposition']['affected_files']
for name in target_files + ['docs/review-tools/historical_fix_rechecks.py']:
    assert digest((ROOT / name).read_bytes()) == current_checked[name], name

# Use the real immutable active-pointer chain and prior closed dispositions.
# No candidate current phase, collection, seal or all-stage gate is fabricated.
prior = history.Rechecks.__new__(history.Rechecks)
prior.root, prior.api = ROOT, api
prior.manifest = {'author_tasks': ['/root']}
prior.prior(request['prior_active_pointer'])
assert history.identity(prior.old_disposition_ref) == history.identity(request['issues'][0]['disposition_ref'])
original_reports = {}
original_findings = []
old_closure = None
for item in request['issues']:
    old = api.node(prior.old_dispositions, item['disposition_ref']['json_pointer'])
    assert old == item['original_disposition']
    assert old['action'] == 'fixed' and old['final_necessity'] == 'necessary'
    original = load(item['original_issue']['report'])
    original_reports[item['original_issue']['report']['path']] = original
    issue = api.node(original, item['original_issue']['json_pointer'])
    assert issue['id'] == old['issue_id'] and issue['necessary'] is True
    closure_reference = old['verification'][0]
    assert closure_reference['kind'] == 'tool_audit'
    closure = load(closure_reference['report'])
    assert closure['verdict'] == 'pass' and old['issue_id'] in closure['resolved_issue_ids']
    assert api.owner(original) == api.owner(closure)
    assert prior.old_tasks[api.owner(original)]['role'] == 'technical'
    assert api.owner(first_full) != api.owner(original)
    assert api.owner(first_full) not in prior.old_tasks
    prior.unavailable(request['unavailability'], api.owner(original), datetime.now(UTC))
    old_checked = {row['path']: row['sha256'] for row in closure['checked_files']}
    assert all(old_checked[name] == prior.prior_manifest['implementation'][name] for name in target_files)
    original_findings.append({'issue_id': old['issue_id'], 'disposition_ref': item['disposition_ref'],
                              'original_report': item['original_issue']['report'],
                              'json_pointer': item['original_issue']['json_pointer'],
                              'necessity': old['final_necessity'], 'original_severity': issue['severity'],
                              'original_action': old['action'], 'historical_fix_verification': closure_reference})
    old_closure = closure

artifact_refs = {}
for report in original_reports.values():
    for artifact in report.get('artifacts', []):
        artifact_refs[Path(artifact['path']).name] = artifact
case_sources = {}
for name in ('author-patch-results-current-03.json', 'fence-admission-results-current-04.json',
             'fence-semantics-results-current-04.json'):
    source = load(artifact_refs[name])
    assert source['exit_code'] == 0
    case_sources[name] = json.loads(source['stdout'])

guide = (ROOT / 'docs/technical-review-guide.md').read_text()
for phrase in ('author_patches', 'undo_patches', '行內程式碼', '數學式', '連結', '沒有圍欄的程式區塊',
               '程式圍欄寫法單純', '閘門沒認出、網站或 Notebook 卻顯示成程式'):
    assert phrase in guide, phrase

controls = []


def check_case(label, body, expected_accept, issue_id, patch=None, reviewed=None, origin=None):
    patch = patch or {'id': 'probe', 'path': 'course/chapters/99.md', 'old': 'old-token', 'new': 'new-token',
                      'requested_by': 'author', 'requested_on': '2026-10-20', 'reason': 'independent synthetic patch control'}
    reviewed = reviewed if reviewed is not None else body.replace(patch['new'], patch['old'])
    with tempfile.TemporaryDirectory(prefix='P5-four-historical-') as directory:
        root = Path(directory)
        path = root / patch['path']
        path.parent.mkdir(parents=True)
        path.write_bytes(body.encode())
        listed = {'schema_version': 1, 'kind': api.PATCHES_KIND, 'statement': api.PATCHES_STATEMENT,
                  'patches': [patch]}
        raw = json.dumps(listed, ensure_ascii=False).encode()
        (root / 'patches.json').write_bytes(raw)
        manifest = {'author_patches': {'path': 'patches.json', 'sha256': digest(raw)}}
        try:
            checked = api.author_patches(root, manifest, {'99.1': {'source': patch['path']}})
            versions = api.file_versions(root, checked)
            assert versions == {patch['path']: {digest(reviewed.encode()): (patch['id'],)}}
            accepted, error = True, None
        except ValueError as caught:
            accepted, error = False, str(caught)
        assert accepted == expected_accept, (label, accepted, error)
        assert path.read_bytes() == body.encode()
        controls.append({'case': label, 'issue_id': issue_id, 'accepted': accepted,
                         'expected_accept': expected_accept, 'original_source': origin,
                         'current_input_sha256': digest(body.encode()),
                         'reviewed_input_sha256': digest(reviewed.encode()), 'error': error,
                         'author_source_or_original_fixture_mutated': False})


legacy_rows = {row['case']: row for row in case_sources['author-patch-results-current-03.json']['cases']}
for name in ('link_url', 'link_label', 'reference_link', 'image_source', 'html_image'):
    row = legacy_rows[name]
    check_case(name, row['input']['current_body'], True, 'AUTHOR-PATCH-MARKUP',
               row['input']['patch'], row['input']['reviewed_body'], artifact_refs['author-patch-results-current-03.json'])
for name in ('indented_fake_close', 'quoted_fake_close', 'indented_code', 'html_pre'):
    row = legacy_rows[name]
    check_case(name, row['input']['current_body'], name in {'indented_code', 'html_pre'}, 'AUTHOR-PATCH-CODEBLOCK',
               row['input']['patch'], row['input']['reviewed_body'], artifact_refs['author-patch-results-current-03.json'])
semantic_rows = {row['case']: row for row in case_sources['fence-semantics-results-current-04.json']['cases']}
for suffix in ('nbsp', 'emspace', 'ideographic_space', 'vertical_tab'):
    for fence in ('backtick', 'tilde'):
        name = 'tail_' + suffix + '_' + fence
        check_case(name, semantic_rows[name]['text'], False, 'AUTHOR-PATCH-FENCE-WHITESPACE',
                   origin=artifact_refs['fence-semantics-results-current-04.json'])
for name in ('invalid_backtick_info', 'invalid_superfences_options'):
    check_case(name, semantic_rows[name]['text'], False, 'AUTHOR-PATCH-FENCE-STATE',
               origin=artifact_refs['fence-semantics-results-current-04.json'])

# Preserve all eight old full-admission instances as separate current controls,
# including their stricter present policy treatment of trailing ASCII spaces.
for row in case_sources['fence-admission-results-current-04.json']['cases']:
    check_case('old_admission_' + row['case'], row['input']['current_body'], False,
               'historical_admission_instance', row['input']['patch'], row['input']['reviewed_body'],
               artifact_refs['fence-admission-results-current-04.json'])

check_case('plain_prose_positive', '文字 new-token。\n', True, 'shared_control')
check_case('prose_after_closed_plain_block', '```python\nprint(4)\n```\nnew-token\n', True, 'shared_control')
check_case('prose_after_closed_CRLF_block', '```python\r\nprint(4)\r\n```\r\nnew-token\r\n', True, 'shared_control')
check_case('inside_plain_fenced_code', '```python\nprint("new-token")\n```\n', False, 'shared_control')
check_case('outside_marker_but_unclosed_fence', 'new-token\n```python\nprint(4)\n', False, 'shared_control')
check_case('rewind_creates_nonplain_fence', '用``X 開頭\n', False, 'shared_control',
           {'id': 'rewind', 'path': 'course/chapters/99.md', 'old': '` 開頭', 'new': 'X 開頭',
            'requested_by': 'author', 'requested_on': '2026-10-20', 'reason': 'rewind-only malformed fence control'})

assert len(controls) == 33 and all(row['accepted'] == row['expected_accept'] for row in controls)
current_issues_missing_ids = [row['issue_id'] for row in original_findings
                              if row['issue_id'] not in first_full['resolved_issue_ids']]
assert len(current_issues_missing_ids) == 4
result = {
    'reviewer_task': '/root/omni_p5_full_tools_review', 'created_at': datetime.now(UTC).isoformat(),
    'request': ref(REQUEST), 'first_complete460_source_verdict_unchanged': request['current_full_scope_report'],
    'provenance_receipt': ref(BASE / 'scope-first-complete460-provenance-v1.json'),
    'current_files_sha256': {name: digest((ROOT / name).read_bytes()) for name in target_files},
    'current_policy': ref(ROOT / 'docs/technical-review-guide.md'),
    'actual_immutable_prior_pointer_and_completed_gate_checked_by_native_helper': True,
    'accepted_original_closed_dispositions_exact': True,
    'authentic_old_reviewer_unavailability_checked_by_native_helper': request['unavailability'],
    'original_findings_and_instances': original_findings,
    'historical_input_sources': [artifact_refs[name] for name in case_sources],
    'control_count': len(controls), 'controls_per_group': dict(Counter(row['issue_id'] for row in controls)),
    'controls': controls, 'all_expected': True,
    'additional_policy_context_outside_author460_inventory': ref(ROOT / 'docs/technical-review-guide.md'),
    'prior_reviewer_harness_failure_preserved': {'script': ref(PROOFS / 'historical-four-tool-controls-v1.py'), 'log': ref(PROOFS / 'historical-four-tool-controls-v1.log'), 'reason': 'v1 assumed the technical policy guide appeared in author460 inventory. The guide is actual necessary static context, separately SHA bound, not added to the460 checked_files or earlier reports. v1 failed before controls.'},
    'decision': {row['issue_id']: 'still_fixed_under_current_explicit_policy_and_plain_fence_admission'
                 for row in original_findings},
    'semantic_reasons': {
        'AUTHOR-PATCH-MARKUP': 'All five original link/image instances still accept exact listed undo. Current normative guide and code explicitly place these locations under requesting-author responsibility; no claim of HTML/Markdown node exclusion remains.',
        'AUTHOR-PATCH-CODEBLOCK': 'Both original indented/quoted false closers reject the whole nonplain file. Original indented-code/HTML-pre instances remain accepted, explicitly excluded by current normative guide; plain fenced code still rejects.',
        'AUTHOR-PATCH-FENCE-WHITESPACE': 'All eight original Unicode/control-whitespace by backtick/tilde instances reject under exact whole-line PLAIN_FENCE, without strip-based close simulation. CRLF plain positive still accepts.',
        'AUTHOR-PATCH-FENCE-STATE': 'Both original bad-backtick-info and invalid SuperFences options reject before any false opener can consume a true opener. The original whole-file malformed and every rewound version checks remain active.'
    },
    'original_four_necessities_unchanged': ['necessary'] * 4,
    'first_complete_report_missing_four_prompted_resolved_ids': current_issues_missing_ids,
    'next_report_requirement': 'Native closed_item requires full.resolved_issue_ids to contain all four. Preserve a44 original full460; write new same-source current only after this substantive recheck, notify root for actual receipt, then write appendix referencing that new full and original a44.',
    'limits': ['Actual current author_patches/file_versions calls on private fixtures; not a new all-stage candidate gate, preflight, seal, parser/browser certificate or full-suite reexecution.',
               'All original19 named examples plus8 historical admission instances retain individual results. Legacy result files are input provenance, not substituted current proof.',
               'The policy leaves inline code/math/link/HTML/indented code and rendering transformations to the requesting author. This recheck confirms present declared behavior and original closure scope; it does not extend automatic semantic protection.',
               'No author/source/checker/old report edits, new formal neural final, training, SDK/cloud/provider/workflow/paid operation, publication or collection/seal.'],
}
for name in target_files:
    assert digest((ROOT / name).read_bytes()) == current_checked[name]
out = PROOFS / 'historical-four-tool-controls-result-v2.json'
with out.open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'historical_items': 4, 'controls': len(controls), 'all_expected': True,
                  'native_prior_and_unavailability_verified': True, 'source_or_prior_reports_changed': False}))
