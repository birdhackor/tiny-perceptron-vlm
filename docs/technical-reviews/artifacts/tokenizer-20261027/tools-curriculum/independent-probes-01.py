"""Fresh reviewer probes; mutate synthetic fixtures only, never reviewed source/evidence."""

import copy
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SCRATCH = Path('/workspace/work/tokenizer-dynamic-20261027/tools-curriculum/probes-02')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


test = load('fresh_tool_review_fixture', ROOT / 'tests/test_historical_rechecks.py')
review, inventory = test.review, test.inventory
preflight = inventory.preflight
results = []


def replace(value, old, new):
    if isinstance(value, dict):
        if value.get('path') == old['path'] and value.get('sha256') == old['sha256']:
            value.update(new)
        for child in list(value.values()):
            replace(child, old, new)
    elif isinstance(value, (list, tuple)):
        for child in value:
            replace(child, old, new)


def edit(root, built, name, change):
    path = root / name
    old = {'path': name, 'sha256': test.sha(path.read_bytes())}
    value = json.loads(path.read_bytes())
    change(value)
    new = test.save(root, name, value)
    pending = [(old, new)]
    while pending:
        before, after = pending.pop(0)
        replace(built, before, after)
        for other in sorted(root.glob('current/*.json')):
            if other == path:
                continue
            document = json.loads(other.read_bytes())
            previous = copy.deepcopy(document)
            replace(document, before, after)
            if document != previous:
                relative = other.relative_to(root).as_posix()
                prior = {'path': relative, 'sha256': test.sha(other.read_bytes())}
                current = test.save(root, relative, document)
                pending.append((prior, current))


def run(name, change=None, expect='reject', old_role='reader', variant=None, local_inventory=True):
    root = SCRATCH / name
    root.mkdir(parents=True, exist_ok=False)
    built = test.fixture(root, variant=variant, old_role=old_role)
    # Existing fixture changes the old role but keeps a technical dispatch window.
    # Correct that synthetic-only window before using the broader page preflight.
    if old_role == 'reader':
        edit(root, built, 'current/assignments.json', lambda value: next(item for item in value['tasks'] if item['task'] == test.OLD_PAGE).update(dispatch_window={'not_before': test.at(1, True), 'not_after': test.at(2, True)}))
    if change:
        change(root, built)
    manifest, live, tasks, sealed = built
    row = {'case': name, 'expected': expect}
    try:
        value = review.check_dispositions(root, manifest, ('reader', 'technical'), live, tasks, sealed)
        row.update(gate_dispositions='accepted', gate_result=value)
    except Exception as error:
        row.update(gate_dispositions='rejected', error=f'{type(error).__name__}: {error}')
    if local_inventory:
        result = inventory.inventory(root, manifest, live, 'sealed', 'technical')
        row['inventory'] = 'rejected' if result['failing'] else 'accepted'
        row['inventory_errors'] = result['errors'] + [message for item in result['dispositions']['failing'] for message in item['errors']]
    wanted = 'accepted' if expect == 'accept' else 'rejected'
    row['as_expected'] = row['gate_dispositions'] == wanted and (not local_inventory or row['inventory'] == wanted)
    results.append(row)
    return root, built, row


def json_change(name, change):
    return lambda root, built: edit(root, built, name, change)


run('valid-old-technical', expect='accept', old_role='technical')
root, built, positive = run('valid-old-reader', expect='accept')
# Preflight independently checks the current page; synthetic trace state is explicit here.
test.save(root, 'current/state.json', {'page_id': 'readme', 'reading_plan': review.document(root, built[0]['pages'][0]['reader']['sessions'][0]['opening'])['reading_plan']})
draft_result = preflight.check_draft(root, built[0], built[1], 'sealed', 'technical')
positive['preflight_failing'] = draft_result['failing']
positive['as_expected'] &= not draft_result['failing']

run('no-reader-proof', variant='fresh_reader_absent')
run('no-sidecar', lambda root, built: built[0].pop('historical_fix_rechecks'))
run('reader-page-omitted', lambda root, built: built[0]['pages'].clear())
run('reader-current-wrong-page', json_change('current/reader-current.json', lambda value: value.update(page_id='another-page')))
run('reader-assignment-wrong-page', json_change('current/assignments.json', lambda value: next(item for item in value['tasks'] if item['task'] == test.NEW_READER).update(pages=['another-page'])))
run('reader-author', lambda root, built: built[0]['author_tasks'].append(test.NEW_READER))
run('reader-wrong-role', lambda root, built: built[2].update({test.NEW_READER: 'continuity'}), local_inventory=False)
run('reader-trace-wrong-owner', json_change('current/reader-opening.json', lambda value: value.update(reviewer_task=test.NEW_PAGE)))
run('reader-note-wrong-owner', json_change('current/reader-note-0.json', lambda value: value.update(reviewer_task=test.NEW_PAGE)))
run('reader-note-wrong-page', json_change('current/reader-note-0.json', lambda value: value.update(page_id='another-page')))
run('reader-note-wrong-index', json_change('current/reader-note-0.json', lambda value: value.update(unit_index=17)))
run('reader-note-empty-quote', json_change('current/reader-note-0.json', lambda value: value['concept_checks'][0].update(source_quote='')))
run('reader-trace-wrong-source', lambda root, built: built[0]['pages'][0]['reader']['sessions'][0].update(source={'path': 'prior/readme.md', 'sha256': test.sha((root / 'prior/readme.md').read_bytes())}))
run('reader-trace-missing-figure', lambda root, built: built[0]['pages'][0]['reader']['sessions'][0].update(figures={}))
run('reader-plan-wrong-owner', json_change('current/reader-plan.json', lambda value: value.update(reviewer_task=test.NEW_PAGE)))
run('reader-plan-missing-page', json_change('current/reader-plan.json', lambda value: value.update(main_pages=['another-page'])))
run('reader-plan-created-after-start', json_change('current/reader-plan.json', lambda value: value.update(created_at=test.at(5))))
run('reader-freeze-created-after-start', json_change('current/freeze.json', lambda value: value.update(created_at=test.at(5))))
run('reader-missing-sessions', lambda root, built: built[0]['pages'][0]['reader'].update(sessions=[]))
run('reader-other-wave-session', lambda root, built: built[0]['pages'][0]['reader']['sessions'][0].update(wave='old'))
run('reader-missing-original-receipt', lambda root, built: built[3]['new']['reader']['collection_index'].pop(0), local_inventory=False)
run('reader-duplicate-current-receipt', lambda root, built: built[3]['new']['reader']['collection_index'].append(copy.deepcopy(built[3]['new']['reader']['collection_index'][-1])), local_inventory=False)
run('reader-basis-wrong-owner', json_change('current/reader-original.json', lambda value: value.update(reviewer_task=test.NEW_PAGE)))
run('reader-current-created-after-receipt', json_change('current/reader-current.json', lambda value: value.update(created_at=test.at(10))))
run('reader-no-original-report', lambda root, built: built[0]['pages'][0]['reader'].pop('original_report'))
run('reader-no-basis', lambda root, built: built[0]['pages'][0]['reader'].pop('basis'))
run('reader-unknown-collection-role', lambda root, built: built[0]['pages'][0]['reader'].update(collection_role='unrecorded'))

# A complete, collected checkpoint closure must work as well as the all-sufficient path.
def add_closure(root, built, wrong_owner=False, missing_resolution=False):
    edit(root, built, 'current/reader-note-0.json', lambda value: value['concept_checks'][0].update(verdict='需補說明'))
    proof = built[0]['pages'][0]['reader']
    note = proof['sessions'][0]['notes'][0]
    body = {
        'reviewer_task': test.NEW_PAGE if wrong_owner else test.NEW_READER,
        'page_id': 'readme', 'created_at': test.at(24), 'source_sha256': built[1]['readme']['source_sha256'],
        'figures_sha256': built[1]['readme']['figures_sha256'], 'verdict': 'pass',
        'basis_report': proof['report'], 'issues': [],
        'checkpoint_resolutions': [] if missing_resolution else [{'note': note, 'concept_index': 0,
            'original_verdict': '需補說明', 'final_status': 'resolved',
            'reason': 'Synthetic complete recheck after reading.', 'evidence': ['synthetic explicit evidence']}],
    }
    ref = test.save(root, 'current/reader-closure.json', body)
    proof['checkpoint_closure'] = {'wave': 'new', 'collection_role': 'technical', 'received_at': test.at(25), 'report': ref, 'json_pointer': ''}
    built[0]['issue_reports'].append(ref)
    built[3]['new']['technical']['report_sha256s'].append(ref['sha256'])
    built[3]['new']['technical']['collection_index'].append({**ref, 'received_at': test.at(25)})
    edit(root, built, 'new/seal-technical.json', lambda value: value.update(copy.deepcopy(built[3]['new']['technical'])))


run('reader-valid-checkpoint-closure', add_closure, expect='accept')
run('reader-checkpoint-closure-wrong-owner', lambda root, built: add_closure(root, built, wrong_owner=True))
run('reader-checkpoint-closure-no-resolution', lambda root, built: add_closure(root, built, missing_resolution=True))

output = {
    'reviewer_task': '/root/tech_tools_curriculum_20261027', 'created_at': datetime.now(UTC).isoformat(),
    'command': 'uv run python docs/technical-reviews/artifacts/tokenizer-20261027/tools-curriculum/independent-probes-01.py',
    'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'CPU; repository metadata validators'},
    'scope': 'Synthetic unmodified existing fixture with independent mutations; real helper, technical _validate, reader trace/closure, preflight, and issue inventory. No complete publication manifest or all gate.',
    'results': results, 'all_as_expected': all(item['as_expected'] for item in results),
}
destination = ROOT / 'docs/technical-reviews/artifacts/tokenizer-20261027/tools-curriculum/independent-probes-02.json'
destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'cases': len(results), 'all_as_expected': output['all_as_expected'], 'unexpected': [item for item in results if not item['as_expected']]}, ensure_ascii=False, indent=2))
raise SystemExit(0 if output['all_as_expected'] else 1)
