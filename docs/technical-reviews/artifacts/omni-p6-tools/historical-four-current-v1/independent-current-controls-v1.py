"""Own prompted historical recheck; original saved inputs, current native functions.

No current stage/seal or full-candidate admission is fabricated. The original
reports and the previous replacement review are inputs, never this reviewer's work.
"""

import hashlib
import importlib.util
import json
import socket
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path('/workspace/work/omni-book-integrated-v1')
OUT = Path(__file__).resolve().parent
TASK = '/root/omni_p6_technical_tools'
WAVE = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/wave04'
REQUEST = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/historical-four-tool-request-v1.json'
EXTERNAL = Path('/workspace/work/omni-p6-tools-historical-current-v1')
EXTERNAL.mkdir(exist_ok=True)


def offline(*args, **kwargs):
    raise RuntimeError('network prohibited in independent CPU historical controls')


socket.create_connection = offline
socket.getaddrinfo = offline
socket.socket.connect = offline


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(raw), 'bytes': len(raw)}


observed_refs = {}


def read(reference):
    path = ROOT / reference['path']
    actual = ref(path)
    assert actual['sha256'] == reference['sha256'], reference
    observed_refs[actual['path']] = actual
    return json.loads(path.read_bytes())


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


request_ref = ref(REQUEST)
assert request_ref['sha256'] == '57b4742a61cc6846f2a89d8e22808d231fcda14f8a35a06305b6c9f6aa8b1da7'
request = read(request_ref)
api = mod('own_p6_historical_incremental', ROOT / 'docs/review-tools/incremental_revision.py')
history = mod('own_p6_historical_native', ROOT / 'docs/review-tools/historical_fix_rechecks.py')
original_ref = ref(WAVE / 'technical-tools/original-v1.json')
full_ref = ref(WAVE / 'technical-tools/current-v1.json')
supplement_ref = ref(WAVE / 'technical-tools/data-supplement-v1.json')
assert original_ref['sha256'] == '2ebf862da7355fdc11d94ffdda7fc097e35569df56294629aab5501d2a84d20e'
assert full_ref['sha256'] == '9518087928fa2baa55f5b27dbff9551465e305d66fb4ab130bf75d20fb23988b'
assert supplement_ref['sha256'] == 'eabb3d0035397a8b4cac4eeaf32d2f18cb01657775b17ad08ef6d89567f2c108'
original, full = read(original_ref), read(full_ref)
supplement = read(supplement_ref)
intake_ref = ref(WAVE / 'technical-collection-intake-v30.json')
assert intake_ref['sha256'] == '3afae0b6ac39caca385b1a873c9c7c94520a0d8b54235feb8365af885771408f'
intake = read(intake_ref)
own_receipts = []
for report_ref, report in [(original_ref, original), (full_ref, full), (supplement_ref, supplement)]:
    matches = [(i, row) for i, row in enumerate(intake['collection_index'])
               if history.identity(row) == history.identity(report_ref)]
    assert len(matches) == 1
    i, row = matches[0]
    assert row['reviewer_task'] == TASK and row['role'] == row['collection_role'] == 'technical'
    assert row['wave'] == 'omni-mainline-04-final'
    assert api.at(report['created_at']) <= api.at(row['received_at']) <= datetime.now(UTC)
    own_receipts.append({'report': intake_ref, 'json_pointer': f'/collection_index/{i}', **row})

# A fingerprint stability pass, not a fresh semantic read of the complete460.
original_files = {row['path']: row['sha256'] for row in original['checked_files']}
full_files = {row['path']: row['sha256'] for row in full['checked_files']}
assert len(original_files) == len(full_files) == 460 and original_files == full_files
for path, expected in full_files.items():
    assert sha((ROOT / path).read_bytes()) == expected, path

# Use the actual pointer chain, old assignments, completed historical gate and
# exact prior dispositions. This does not close any current OPEN phase.
prior = history.Rechecks.__new__(history.Rechecks)
prior.root, prior.api = ROOT, api
prior.manifest = {'author_tasks': ['/root']}
prior.prior(request['prior_active_pointer'])
read(request['prior_active_pointer'])
read(request['unavailability']['report'])
issues = []
artifact_refs = {}
for requested in request['issues']:
    old = api.node(prior.old_dispositions, requested['disposition_ref']['json_pointer'])
    assert old == requested['original_disposition']
    assert history.identity(requested['disposition_ref']) == history.identity(prior.old_disposition_ref)
    original_report = read(requested['original_issue']['report'])
    issue = api.node(original_report, requested['original_issue']['json_pointer'])
    assert issue['id'] == old['issue_id'] and issue['necessary'] is True
    assert old['final_necessity'] == 'necessary' and old['action'] == 'fixed'
    assert len(old['verification']) == 1 and old['verification'][0]['kind'] == 'tool_audit'
    verification = read(old['verification'][0]['report'])
    assert verification['verdict'] == 'pass' and issue['id'] in verification['resolved_issue_ids']
    assert api.owner(original_report) == api.owner(verification) == '/root/rv1019_technical_review-tools'
    assert prior.old_tasks[api.owner(original_report)]['role'] == 'technical'
    assert TASK not in prior.old_tasks and TASK != api.owner(original_report)
    prior.unavailable(request['unavailability'], api.owner(original_report), datetime.now(UTC))
    old_checked = {row['path']: row['sha256'] for row in verification['checked_files']}
    for path in old['affected_files']:
        assert old_checked[path] == prior.prior_manifest['implementation'][path]
        assert full_files[path] == sha((ROOT / path).read_bytes())
    for artifact in original_report.get('artifacts', []):
        artifact_refs[Path(artifact['path']).name] = artifact
    for artifact in verification.get('artifacts', []):
        # Full hash verification of historical evidence, not reexecution or
        # a claim to have semantically reread each historical probe.
        path = ROOT / artifact['path']
        assert ref(path)['sha256'] == artifact['sha256']
        observed_refs[artifact['path']] = ref(path)
    issues.append({'issue_id': issue['id'], 'severity': issue['severity'], 'necessary': issue['necessary'],
                   'disposition_ref': requested['disposition_ref'],
                   'original_report': requested['original_issue']['report'],
                   'json_pointer': requested['original_issue']['json_pointer'],
                   'historical_fix_verification': old['verification'][0],
                   'original_closed_disposition_preserved': old})

saved = {}
for name in ['author-patch-results-current-03.json', 'fence-admission-results-current-04.json',
             'fence-semantics-results-current-04.json']:
    result = read(artifact_refs[name])
    assert result['exit_code'] == 0
    saved[name] = json.loads(result['stdout'])

policy_path = ROOT / 'docs/technical-review-guide.md'
policy_text = policy_path.read_text()
for clause in ['author_patches', 'undo_patches', '程式圍欄寫法單純', '行內程式碼', '數學式', '連結',
               '沒有圍欄的程式區塊', '閘門沒認出、網站或 Notebook 卻顯示成程式的字']:
    assert clause in policy_text

results = []


def probe(name, group, text, expected, origin=None, patch=None, reviewed=None):
    patch = patch or {'id': 'own-probe', 'path': 'course/chapters/99.md', 'old': 'old-token',
                      'new': 'new-token', 'requested_by': 'author', 'requested_on': '2026-10-20',
                      'reason': 'own offline historical current recheck'}
    reviewed = reviewed if reviewed is not None else text.replace(patch['new'], patch['old'])
    with tempfile.TemporaryDirectory(prefix='own-control-', dir=EXTERNAL) as temporary:
        directory = Path(temporary)
        file_path = directory / patch['path']
        file_path.parent.mkdir(parents=True)
        file_path.write_bytes(text.encode())
        payload = {'schema_version': 1, 'kind': api.PATCHES_KIND,
                   'statement': api.PATCHES_STATEMENT, 'patches': [patch]}
        patch_path = directory / 'patches.json'
        patch_path.write_text(json.dumps(payload, ensure_ascii=False))
        reference = {'path': 'patches.json', 'sha256': sha(patch_path.read_bytes())}
        error = None
        try:
            admitted = api.author_patches(directory, {'author_patches': reference},
                                          {'99.1': {'source': patch['path']}})
            versions = api.file_versions(directory, admitted)
            assert versions == {patch['path']: {sha(reviewed.encode()): (patch['id'],)}}
            accepted = True
        except ValueError as caught:
            accepted, error = False, str(caught)
        assert accepted == expected, (name, accepted, expected, error)
        assert file_path.read_bytes() == text.encode()
        results.append({'name': name, 'group': group, 'expected_accept': expected,
                        'actual_accept': accepted, 'error': error, 'input_origin': origin,
                        'current_input_sha256': sha(text.encode()),
                        'reviewed_input_sha256': sha(reviewed.encode()),
                        'method': 'actual current author_patches then file_versions; no stubs'})


patch_rows = {row['case']: row for row in saved['author-patch-results-current-03.json']['cases']}
for name in ['link_url', 'link_label', 'reference_link', 'image_source', 'html_image',
             'indented_fake_close', 'quoted_fake_close', 'indented_code', 'html_pre']:
    row = patch_rows[name]
    group = 'AUTHOR-PATCH-MARKUP' if name in ['link_url', 'link_label', 'reference_link', 'image_source', 'html_image'] else 'AUTHOR-PATCH-CODEBLOCK'
    probe(name, group, row['input']['current_body'], name not in ['indented_fake_close', 'quoted_fake_close'],
          {'report': artifact_refs['author-patch-results-current-03.json'], 'case': name},
          row['input']['patch'], row['input']['reviewed_body'])

semantic_rows = {row['case']: row for row in saved['fence-semantics-results-current-04.json']['cases']}
for whitespace in ['nbsp', 'emspace', 'ideographic_space', 'vertical_tab']:
    for fence in ['backtick', 'tilde']:
        name = f'tail_{whitespace}_{fence}'
        probe(name, 'AUTHOR-PATCH-FENCE-WHITESPACE', semantic_rows[name]['text'], False,
              {'report': artifact_refs['fence-semantics-results-current-04.json'], 'case': name})
for name in ['invalid_backtick_info', 'invalid_superfences_options']:
    probe(name, 'AUTHOR-PATCH-FENCE-STATE', semantic_rows[name]['text'], False,
          {'report': artifact_refs['fence-semantics-results-current-04.json'], 'case': name})
for row in saved['fence-admission-results-current-04.json']['cases']:
    probe('original-admission-' + row['case'], 'original_admission_instance', row['input']['current_body'], False,
          {'report': artifact_refs['fence-admission-results-current-04.json'], 'case': row['case']},
          row['input']['patch'], row['input']['reviewed_body'])

for name, text, expected in [
    ('ordinary_prose', '正文 new-token。\n', True),
    ('after_plain_closed_block', '```python\nprint(4)\n```\nnew-token\n', True),
    ('after_CRLF_closed_block', '```python\r\nprint(4)\r\n```\r\nnew-token\r\n', True),
    ('inside_plain_block', '```python\nprint("new-token")\n```\n', False),
    ('unclosed_elsewhere', 'new-token\n```python\nprint(4)\n', False),
]:
    probe(name, 'own_control', text, expected)
probe('rewound_version_nonplain', 'own_control', '用``X 開頭\n', False,
      patch={'id': 'own-rewind', 'path': 'course/chapters/99.md', 'old': '` 開頭', 'new': 'X 開頭',
             'requested_by': 'author', 'requested_on': '2026-10-20', 'reason': 'rewind validation control'})
assert len(results) == 33

target_files = request['issues'][0]['original_disposition']['affected_files']
result = {
    'schema_version': 1, 'reviewer_task': TASK, 'created_at': datetime.now(UTC).isoformat(),
    'kind': 'own prompted four historical issues current controls; bounded, not full460 rereading',
    'request': request_ref, 'preprompt_full_review': full_ref, 'original_review': original_ref,
    'recorded_genuine_receipts': own_receipts,
    'prior_active_pointer': request['prior_active_pointer'], 'unavailability': request['unavailability'],
    'native_prior_chain_old_assignments_dispositions_unavailability_checked': True,
    'original_historical_findings_and_dispositions': issues,
    'current_files_sha256': {name: full_files[name] for name in target_files},
    'current_policy': ref(policy_path), 'control_count': len(results),
    'controls_per_group': dict(Counter(row['group'] for row in results)),
    'controls': results, 'all_expected': True,
    'all460_same_source_fingerprint_check': {'count': len(full_files), 'method': 'hash current bytes against own immutable original and current1 checked_files; not new complete semantic read'},
    'checked_historical_evidence_refs': list(observed_refs.values()),
    'preprompt_current_missing_four_resolved_ids': [i['issue_id'] for i in issues if i['issue_id'] not in full.get('resolved_issue_ids', [])],
    'native_schema_requirement': 'historical_fix_rechecks.py closed_item requires each old.issue_id in full.resolved_issue_ids; write a new current2 only after these actual prompted checks, then obtain its genuine receipt before final appendix',
    'limits': [
        'Original necessary flags, original P1/P2 severities, distinct old issue/disposition pointers and original verification arrays remain unchanged.',
        'Five markup and two un-fenced code cases are accepted by design under current explicit author responsibility; no Markdown/HTML node exclusion claimed.',
        'Own execution is current author_patches/file_versions on the authentic saved inputs plus six controls, not replay of old full stage=all/preflight/inventory fixtures.',
        'Old historical verification artifacts are hash-checked byref and their relevant closure assertions read; old large semantic searches/rendering matrices are not rerun or claimed as own work.',
        'No whole current-stage native group or final candidate all-stage gate run while T04 is OPEN; no seal or collection invented.',
        'I01/I02 general production counterexamples remain; prior exact candidate evidence remedy remains scoped and final C04/finalauthor ordinary capsule delta is pending.',
        'Immutable data supplement contains two ignored outputs metadata inputs; final exact candidate verification must use the durable final ordinary proof capsule, not assume those temporary paths publishable.',
        'No author/checker/source edits, model/scientific/training/teacher/final forward, network/cloud/provider/paid/workflow/publication operation.',
    ],
}
destination = OUT / 'independent-current-controls-v1.json'
with destination.open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'saved': ref(destination), 'historical_issues': 4,
                  'controls': len(results), 'all_expected': True, 'fingerprints': len(full_files)}))
