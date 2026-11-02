"""Only /groups/0: authenticate the 10/19 chain and execute the linked BPE example."""

import argparse
import contextlib
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
CACHE = Path('/workspace/work/reference/omni-technical-curriculum-v1')
TASK = '/root/omni_technical_coordinator/technical_curriculum'
ORIGINAL = 'docs/course-revision-20261029-omni/reviews/technical/technical_curriculum/technical-original-v1.json'
REQUEST = 'docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json'
FREEZE = 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--appendix')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    api = module('historical_curriculum_api', 'docs/review-tools/incremental_revision.py')
    v2 = module('historical_curriculum_v2', 'docs/review-tools/historical_fix_rechecks_v2.py')
    inputs = {}

    def document(path):
        inputs[path] = sha(ROOT / path)
        return json.loads((ROOT / path).read_text())

    def refdoc(ref):
        result = api.document(ROOT, ref)
        inputs[ref['path']] = ref['sha256']
        return result

    checks = []

    def check(label, expected, observed):
        checks.append(dict(label=label, expected=expected, observed=observed, passed=expected == observed))
        assert expected == observed, label

    group = document(REQUEST)['groups'][0]
    item = group['items'][0]
    check('bounded request identity', ['accepted_20261019', 'curriculum', '/root/rv1019_reader_ref', 'reader', 1],
          [group['prior_generation'], group['page_id'], group['old_owner'], group['old_role'], len(group['items'])])
    full = document(ORIGINAL)
    check('own immutable original exists before this callback',
          ['fa022b1b8f78093813fd52ce681b37e1ac05523ae46b829eebc1f8cf60bfb902', TASK, 'pass'],
          [inputs[ORIGINAL], full['reviewer_task'], full['verdict']])
    freeze = document(FREEZE)
    page = next(p for p in freeze['pages'] if p['page_id'] == 'curriculum')
    body = (ROOT / page['snapshot']).read_text()
    check('current whole file equals fix13 snapshot', [page['source_sha256']] * 2,
          [sha(ROOT / page['source']), sha(ROOT / page['snapshot'])])
    check('current sole figure remains frozen', page['figures_sha256'],
          {p: sha(ROOT / p) for p in page['figures_sha256']})

    # The actual reader assembly supplies author declarations and existing reader proof.
    manifest_path = 'docs/course-revision-20261029-omni/reviews/assembly-reader-01/manifest-reader.json'
    manifest = document(manifest_path)
    own_intake = document('docs/course-revision-20261029-omni/reviews/technical-stage/assignments-intake.json')
    assignment = next(a for a in own_intake['tasks'] if a.get('task') == TASK)
    wave = assignment['wave']
    wave_record = next(w for w in manifest['review_waves'] if w['id'] == wave)
    sealed = {wave: {role: refdoc(ref) for role, ref in wave_record['seals'].items()}}
    recheck = v2.Rechecks.__new__(v2.Rechecks)
    recheck.root, recheck.api = ROOT, api
    recheck.manifest, recheck.live, recheck.frozen = manifest, {'curriculum': page}, {'curriculum': page}
    recheck.sealed, recheck.entries, recheck.used, recheck.reports = sealed, {}, set(), []
    recheck.checked_readers = set()
    reader_assignments = refdoc(manifest['assignments'])['tasks']
    recheck.assignments = {a['task']: a for a in reader_assignments}
    recheck.assignments[TASK] = assignment
    recheck.tasks = {t: a['role'] for t, a in recheck.assignments.items()}
    recheck.prior(group['prior_active_pointer'])
    check('v2 prior authenticates original generation',
          'docs/course-review-20261019/manifest-final-01.json',
          refdoc(group['prior_active_pointer'])['manifest'])
    check('original disposition file remains the authenticated one',
          {k: item['disposition_ref'][k] for k in ('path', 'sha256')}, recheck.old_disposition_ref)
    old = api.node(recheck.old_dispositions, item['disposition_ref']['json_pointer'])
    check('original optional fixed disposition', ['fixed', 'optional', item['original_report'], item['json_pointer']],
          [old['action'], old['final_necessity'], old['report'], old['json_pointer']])
    check('original verification identity preserved', True, item['historical_fix_verification'] in old['verification'])
    original_report = refdoc(item['original_report'])
    original_issue = api.node(original_report, item['json_pointer'])
    check('original owner and optional necessity', ['/root/rv1019_reader_ref', False],
          [api.owner(original_report), original_issue['necessary']])
    old_report = refdoc(item['historical_fix_verification']['report'])
    old_page = api.node(old_report, item['historical_fix_verification']['json_pointer'])
    prior_page = recheck.old_pages['curriculum']
    check('original closure matches accepted 10/19 frozen page',
          ['curriculum', 'pass', prior_page['source_sha256'], prior_page['figures_sha256']],
          [old_page['page_id'], old_page['verdict'], old_page['source_sha256'], old_page['figures_sha256']])
    api.artifact(ROOT, {'path': prior_page['snapshot'], 'sha256': prior_page['source_sha256']})
    for name, digest in prior_page['figures_sha256'].items():
        check('prior figure snapshot: ' + name, digest, api.sha(api.artifact(ROOT, recheck.old_figures[name])))
    current_dispositions = refdoc(manifest['dispositions'])
    recheck.decisions = {v2._legacy.issue_key(d): d for d in current_dispositions['items']}
    check('actual current dispositions retain exact old item', old,
          recheck.decisions.get(v2._legacy.issue_key(old)))
    refdoc(group['unavailability']['report'])
    recheck.unavailable(group['unavailability'], group['old_owner'], datetime.now(UTC))
    quote = 'BPE（位元組對編碼，[6.2](../course/chapters/06.md#6.2)）可回讀'
    check('current expansion and direct return link', True, quote in body)
    linked = ROOT / 'course/chapters/06.md'
    inputs['course/chapters/06.md'] = sha(linked)
    raw = linked.read_text()
    section = re.search(r'^## 6\.2 .*?(?=^## )', raw, re.M | re.S)[0]
    check('link resolves to the BPE algorithm introduction', True,
          'Byte Pair Encoding，BPE' in section and '## 6.2 常見片段怎麼合併？' in section)
    code = re.search(r'```python\n(.*?)```', section, re.S)[1]

    def execute(source):
        ns, output = {}, io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(source, 'course/chapters/06.md#6.2:inline', 'exec'), ns)
        return dict(stdout=output.getvalue(), pairs={''.join(k): v for k, v in ns['pairs'].items()},
                    selected=list(ns['pair']), merged=[ns['merge'](row, ns['pair']) for row in ns['corpus']])

    base = execute(code)
    check('actual embedded counts', {'ab': 3, 'ba': 2, 'ac': 1}, base['pairs'])
    check('actual embedded merge', [['ab', 'ab'], ['ab', 'a', 'c']], base['merged'])
    changed = execute(code.replace('corpus = [list("abab"), list("abac")]',
                                   'corpus = [list("abab"), list("abac"), list("acacacac")]'))
    check('actual exercise chooses ac5 over ab3', [5, 3, ['a', 'c']],
          [changed['pairs']['ac'], changed['pairs']['ab'], changed['selected']])
    check('actual exercise merge', [['a', 'b', 'a', 'b'], ['a', 'b', 'ac'], ['ac', 'ac', 'ac', 'ac']], changed['merged'])
    calls = [dict(method='v2.Rechecks.prior', status='passed'),
             dict(method='v2.Rechecks.unavailable', status='passed')]
    if args.appendix:
        appendix = document(args.appendix)
        recheck.validate_page('curriculum', TASK, full, appendix)
        calls.append(dict(method='v2.Rechecks.validate_page', status='passed', scope='both own full original and this appendix; exact fix13 page'))
        recheck.current_page(appendix, '', 'curriculum', TASK)
        calls.append(dict(method='v2.Rechecks.current_page', status='passed'))
        bounded_group = dict(group, collection={'wave': wave})
        try:
            recheck.closed_item(bounded_group, appendix['historical_issue_rechecks'][0], full, appendix,
                                TASK, 'curriculum', api.at(appendix['created_at']))
            calls.append(dict(method='v2.Rechecks.closed_item', status='passed',
                              scope='one original reader fix plus actual current reader trace/closure; collection wrapper not invoked'))
        except (ValueError, KeyError) as error:
            calls.append(dict(method='v2.Rechecks.closed_item', status='pending', reason=str(error)))
        try:
            recheck.phase_receipt({'path': ORIGINAL, 'sha256': inputs[ORIGINAL]}, TASK, wave, 'technical', assignment['dispatch_window'])
            calls.append(dict(method='v2.Rechecks.phase_receipt', status='passed'))
        except (ValueError, KeyError) as error:
            calls.append(dict(method='v2.Rechecks.phase_receipt', status='pending', reason=str(error),
                              scope='actual reader assembly has no completed technical seal; no received_at/phase completion invented'))
    result = dict(created_at=datetime.now(UTC).isoformat(), reviewer_task=TASK,
                  command='uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-curriculum-v1/audit_historical_rv1019_reader_ref.py' +
                          ((' --appendix ' + args.appendix) if args.appendix else '') + ' --output ' + args.output,
                  environment={'python': platform.python_version(), 'device': 'cpu; standard-library inline BPE only'},
                  scope='Only request /groups/0 and old_owner /root/rv1019_reader_ref; no new reader/coherence judgement or complete gate result.',
                  source_inputs_sha256=inputs, prior_generation=group['prior_generation'],
                  prior_active_pointer=group['prior_active_pointer'], disposition_ref=item['disposition_ref'],
                  original_issue_quote=original_issue['quote'], current_quote=quote,
                  historical_verification=item['historical_fix_verification'],
                  authenticated_prior_page={k: prior_page[k] for k in ('source_sha256','figures_sha256','snapshot')},
                  current_page=page, section_6_2_sha256=hashlib.sha256(section.encode()).hexdigest(),
                  embedded_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                  base=base, exercise=changed, checks=checks, api_calls=calls,
                  actual_available_seals=list(sealed[wave]),
                  complete_sidecar_or_gate='pending actual technical collection/seal and coordinator assembly; never claimed passed')
    output = ROOT / args.output
    with output.open('x') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(checks=len(checks), api_calls=calls, output=args.output, sha256=sha(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
