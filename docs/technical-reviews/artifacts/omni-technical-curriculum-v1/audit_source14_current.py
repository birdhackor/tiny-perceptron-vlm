"""Bounded source14 dependency/current/history check; never rewrites original evidence."""

import argparse
import contextlib
import copy
import difflib
import hashlib
import importlib.util
import io
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TASK = '/root/omni_technical_coordinator/technical_curriculum'
BASE = 'docs/course-revision-20261029-omni/reviews/technical/technical_curriculum/'
ORIGINAL = BASE + 'technical-original-v1.json'
OLD_APPENDIX = BASE + 'technical-historical-appendix-rv1019-reader-ref-v1.json'
FREEZE13 = 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'
FREEZE14 = 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json'
REQUEST = 'docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json'
PROTOCOL_RECEIPT = 'docs/technical-reviews/artifacts/omni-author-visuals-index-fix01/protocol-previous-source-receipt-v1.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--current')
    parser.add_argument('--appendix')
    args = parser.parse_args()
    inputs, checks = {}, []

    def doc(path):
        inputs[path] = sha(ROOT / path)
        return json.loads((ROOT / path).read_text())

    def check(name, expected, observed):
        checks.append(dict(name=name, expected=expected, observed=observed, passed=expected == observed))
        assert expected == observed, name

    original, old_appendix = doc(ORIGINAL), doc(OLD_APPENDIX)
    check('immutable original', 'fa022b1b8f78093813fd52ce681b37e1ac05523ae46b829eebc1f8cf60bfb902', inputs[ORIGINAL])
    check('immutable historical appendix v1', 'adc977e17c7121d9d3e0c47deda882020e5adf96a0646ffd114d02851fe19e33', inputs[OLD_APPENDIX])
    f13, f14 = doc(FREEZE13), doc(FREEZE14)
    page13 = next(p for p in f13['pages'] if p['page_id'] == 'curriculum')
    page14 = next(p for p in f14['pages'] if p['page_id'] == 'curriculum')
    check('curriculum source14 equals personally reviewed source13', page13, page14)
    check('current curriculum and figure bytes', [page14['source_sha256'], *page14['figures_sha256'].values()],
          [sha(ROOT / page14['source']), *[sha(ROOT / p) for p in page14['figures_sha256']]])
    protocol_source = next(s for s in original['sources'] if s['id'] == 'r_protocol')
    receipt = doc(PROTOCOL_RECEIPT)
    old_path = ROOT / receipt['snapshot']['path']
    check('post-edit reconstruction equals original registered protocol SHA', protocol_source['sha256'], sha(old_path))
    check('reconstruction provenance explicitly post-edit',
          'exact_registered_previous_source_reconstructed_after_author_edit', receipt['kind'])
    new_path = ROOT / protocol_source['path']
    before, after = old_path.read_text(), new_path.read_text()
    check('protocol common stage/review text unchanged', before.split('未發布的本機預覽')[0], after.split('未發布的本機預覽')[0])
    stage_quote = '正式GPU工程在三輪後'
    check('c20 stage boundary in both versions', [True, True], [stage_quote in before, stage_quote in after])
    delta = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), n=2))
    source_deltas, cache_count = [], 0
    for source in original['sources']:
        if source['kind'] == 'repository_code':
            actual = sha(ROOT / source['path'])
            if actual != source['sha256']:
                source_deltas.append(dict(id=source['id'], path=source['path'], original=source['sha256'], current=actual))
        for cached in source.get('inspected_originals', []):
            check('original primary cache ' + cached['name'], cached['sha256'], sha(Path(cached['cache_path'])))
            cache_count += 1
    check('only direct source drift is protocol', ['r_protocol'], [s['id'] for s in source_deltas])
    for artifact in original['artifacts'] + old_appendix['artifacts']:
        check('preserved artifact ' + artifact['id'], artifact['sha256'], sha(ROOT / artifact['path']))
    execution = doc('docs/technical-reviews/artifacts/omni-technical-curriculum-v1/execution.json')
    for path, digest in execution['input_sha256'].items():
        check('reused primary execution input ' + path, digest, sha(ROOT / path))
    appendix_source_deltas = []
    for source in old_appendix['sources']:
        if source['kind'] == 'repository_code':
            actual = sha(ROOT / source['path'])
            if actual != source['sha256']:
                appendix_source_deltas.append(dict(id=source['id'], path=source['path'], original=source['sha256'], current=actual))
        if source['id'] == 'hp':
            check('BPE original paper cache', source['original_sha256'], sha(Path(source['cache_path'])))
    check('historical appendix direct sources remain current or immutable', [], appendix_source_deltas)
    pins = doc('docs/technical-reviews/artifacts/omni-technical-curriculum-v1/source-slice-pins.json')
    current_pins, file_changes = [], []
    for pin in pins['slices']:
        path = ROOT / pin['path']
        raw = path.read_text()
        heads = list(re.finditer(r'^## ([A-Z\d]+\.\d+) (.+)$', raw, re.M))
        index = next(i for i, h in enumerate(heads) if h[1] == pin['lesson_id'])
        section = raw[heads[index].start():heads[index + 1].start() if index + 1 < len(heads) else len(raw)]
        check('exact necessary route slice ' + pin['lesson_id'], pin['section_sha256'], hashlib.sha256(section.encode()).hexdigest())
        updated = copy.deepcopy(pin)
        updated['file_sha256'] = sha(path)
        if updated['file_sha256'] != pin['file_sha256']:
            file_changes.append(dict(lesson_id=pin['lesson_id'], path=pin['path'], old_file_sha256=pin['file_sha256'],
                                     current_file_sha256=updated['file_sha256'], exact_section_unchanged=True))
        current_pins.append(updated)
    check('changed containing file affects only unchanged linked 12.x slices',
          ['12.15','12.16','12.17','12.18'], [p['lesson_id'] for p in file_changes])
    # These are navigation/count checks only, not reviews of source14's modified 12.2/first-steps.
    chapters = list((ROOT / 'course/chapters').glob('*.md'))
    count = lambda p: len(re.findall(r'^## [A-Z\d]+\.\d+ ', p.read_text(), re.M))
    chapter_count = sum(count(p) for p in chapters)
    all_count = chapter_count + sum(count(ROOT / 'course' / p) for p in ['README.md','first-steps.md','training.md','glossary.md'])
    check('current published numbered counts', [331,357], [chapter_count,all_count])
    api = module('curriculum_source14_api', 'docs/review-tools/incremental_revision.py')
    v2 = module('curriculum_source14_v2', 'docs/review-tools/historical_fix_rechecks_v2.py')
    validator = module('curriculum_source14_validator', 'scripts/check_technical_reviews.py')
    setup = module('curriculum_source14_setup', 'scripts/omni_revision/review_setup.py')
    original_snapshots = [dict(source=protocol_source['path'], snapshot=receipt['snapshot'])]
    setup.check_original(original, {'path':FREEZE13, 'sha256':inputs[FREEZE13]}, 'curriculum', 'curriculum',
                         TASK, validator, api, original_source_snapshots=original_snapshots)
    calls = [dict(method='review_setup.check_original', status='passed',
                  scope='real source13 original plus exact registered old-protocol snapshot; not current protocol')]
    group = doc(REQUEST)['groups'][0]
    item = group['items'][0]
    manifest = doc('docs/course-revision-20261029-omni/reviews/assembly-reader-01/manifest-reader.json')
    intake = doc('docs/course-revision-20261029-omni/reviews/technical-stage/assignments-intake.json')
    assignment = next(a for a in intake['tasks'] if a.get('task') == TASK)
    wave = assignment['wave']
    seals = next(w for w in manifest['review_waves'] if w['id'] == wave)['seals']
    checker = v2.Rechecks.__new__(v2.Rechecks)
    checker.root, checker.api = ROOT, api
    checker.manifest, checker.live, checker.frozen = manifest, {'curriculum': page14}, {'curriculum': page14}
    checker.sealed = {wave:{role:api.document(ROOT, ref) for role, ref in seals.items()}}
    checker.entries, checker.used, checker.reports, checker.checked_readers = {}, set(), [], set()
    checker.assignments = {a['task']: a for a in api.document(ROOT, manifest['assignments'])['tasks']}
    checker.assignments[TASK] = assignment
    checker.tasks = {task:a['role'] for task,a in checker.assignments.items()}
    checker.prior(group['prior_active_pointer'])
    checker.unavailable(group['unavailability'], group['old_owner'], datetime.now(UTC))
    calls.extend([dict(method='v2.Rechecks.prior',status='passed'),dict(method='v2.Rechecks.unavailable',status='passed')])
    old = api.node(checker.old_dispositions, item['disposition_ref']['json_pointer'])
    check('original historical dictionary and necessity preserved',
          [item['original_report'],item['json_pointer'],'fixed','optional'],
          [old['report'],old['json_pointer'],old['action'],old['final_necessity']])
    check('original verification remains in old dictionary', True, item['historical_fix_verification'] in old['verification'])
    checker.decisions = {v2._legacy.issue_key(d):d for d in api.document(ROOT,manifest['dispositions'])['items']}
    check('current actual dispositions preserve exact /items/71', old, checker.decisions[v2._legacy.issue_key(old)])
    raw6 = (ROOT / 'course/chapters/06.md').read_text()
    section6 = re.search(r'^## 6\.2 .*?(?=^## )', raw6, re.M | re.S)[0]
    code = re.search(r'```python\n(.*?)```', section6, re.S)[1]
    outputs = []
    for source in [code, code.replace('corpus = [list("abab"), list("abac")]', 'corpus = [list("abab"), list("abac"), list("acacacac")]')]:
        namespace, stdout = {}, io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(source,'current-source14-6.2-inline','exec'), namespace)
        outputs.append(dict(stdout=stdout.getvalue(),selected=list(namespace['pair']),
                            pairs={''.join(k):v for k,v in namespace['pairs'].items()},
                            merged=[namespace['merge'](r,namespace['pair']) for r in namespace['corpus']]))
    check('source14 BPE inline and exercise actual outputs',
          [[['ab','ab'],['ab','a','c']], [['a','b','a','b'],['a','b','ac'],['ac','ac','ac','ac']]],
          [o['merged'] for o in outputs])
    quote = 'BPE（位元組對編碼，[6.2](../course/chapters/06.md#6.2)）可回讀'
    check('source14 historical fix quote remains', True, quote in (ROOT / page14['source']).read_text())
    if args.current:
        current = doc(args.current)
        checker.current_page(current, '', 'curriculum', TASK)
        errors = validator._validate(ROOT, ROOT / page14['source'], 'curriculum', (ROOT / page14['snapshot']).read_text(),
                                     current, {TASK:['curriculum']}, {'/root/omni_reader_entries'})
        check('current source14 _validate', [], errors)
        calls.append(dict(method='v2.Rechecks.current_page and technical._validate',status='passed'))
    if args.appendix:
        appendix = doc(args.appendix)
        checker.validate_page('curriculum',TASK,current,appendix)
        calls.append(dict(method='v2.Rechecks.validate_page',status='passed'))
        checker.closed_item(dict(group,collection={'wave':wave}),appendix['historical_issue_rechecks'][0],current,appendix,
                            TASK,'curriculum',api.at(appendix['created_at']))
        calls.append(dict(method='v2.Rechecks.closed_item',status='passed',scope='one accepted_20261019 reader fix plus actual current reader proof; full collection wrapper not run'))
        try:
            checker.phase_receipt({'path':args.current,'sha256':inputs[args.current]},TASK,wave,'technical',assignment['dispatch_window'])
            calls.append(dict(method='v2.Rechecks.phase_receipt',status='passed'))
        except (KeyError,ValueError) as error:
            calls.append(dict(method='v2.Rechecks.phase_receipt',status='pending',reason=str(error)))
    dependencies = []
    for claim in original['claims']:
        sid = {r['source_id'] for r in claim['evidence']}
        reason = ('Only direct changed dependency r_protocol: full new protocol read, c20 stage boundary unchanged; new final visual/tool-scope paragraphs are outside this page judgement.'
                  if 'r_protocol' in sid else 'Direct registered sources/artifacts/primary caches unchanged; reused evidence remains limited to its original input, hardware, historical data and scientific scope.')
        if claim['id'] in {'c73','c74','c75'}:
            reason += ' The containing chapter12 file changed; exact referenced12.15–18 slices are unchanged and newly pinned.'
        dependencies.append(dict(claim_id=claim['id'],status='still_verified',reason=reason))
    result = dict(created_at=datetime.now(UTC).isoformat(),reviewer_task=TASK,
                  command='uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-curriculum-v1/audit_source14_current.py --output '+args.output+
                          ((' --current '+args.current) if args.current else '')+((' --appendix '+args.appendix) if args.appendix else ''),
                  environment={'python':platform.python_version(),'device':'cpu; metadata and standard-library inline BPE only'},
                  source_inputs_sha256=inputs,current_page=page14,current_freeze={'path':FREEZE14,'sha256':inputs[FREEZE14]},
                  protocol_delta=delta,source_deltas=source_deltas,primary_cache_count=cache_count,
                  original_source_snapshots=original_snapshots,route_file_changes=file_changes,
                  claim_dependencies=dependencies,checks=checks,api_calls=calls,bpe_outputs=outputs,current_quote=quote,
                  actual_available_seals=list(checker.sealed[wave]),complete_gate='pending actual collection/seals; no reader/coherence or whole-tool pass claimed')
    if not args.current:
        updated = dict(source_freeze={'path':FREEZE14,'sha256':inputs[FREEZE14]},assigned_page=page14,
                       scope='Current precise curriculum link pins after source14 bounded check; only chapter12 containing-file bytes changed, all selected slices unchanged.',slices=current_pins)
        with (OUT/'source-slice-pins-current-v1.json').open('x') as f:
            f.write(json.dumps(updated,ensure_ascii=False,indent=2)+'\n')
    with (ROOT / args.output).open('x') as f:
        f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(checks=len(checks),claims=len(dependencies),api_calls=calls,path=args.output,sha256=sha(ROOT/args.output)),ensure_ascii=False))


if __name__ == '__main__':
    main()
