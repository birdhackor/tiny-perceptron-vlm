"""source14/bundle4 training bounded callback; no training or reader-answer review."""

import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import importlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from unittest.mock import patch
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REV = ROOT / 'docs/course-revision-20261029-omni/reviews'
sys.path.insert(0, str(ROOT))
TASK = '/root/omni_technical_coordinator/technical_training'


def now(): return datetime.now(UTC).isoformat()
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
    p = Path(p)
    return {'path': str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), 'sha256': digest(p)}
def load(p): return json.loads(Path(p).read_bytes())
def loadref(r):
    p = ROOT / r['path']; assert digest(p) == r['sha256'], r
    return load(p)
def save(p, value):
    with Path(p).open('x', encoding='utf-8') as f: f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def node(d, pointer):
    for k in pointer.split('/')[1:]: d = d[int(k)] if isinstance(d, list) else d[k.replace('~1', '/').replace('~0', '~')]
    return d
def section(body, sid):
    m = re.search(r'^## ' + re.escape(sid) + r' .+$', body, re.M); assert m
    e = re.search(r'^## ', body[m.end():], re.M)
    return body[m.start():m.end()+e.start() if e else len(body)]


def main():
    started = now()
    first_path = REV / 'technical/technical_training/technical-original-v1.json'
    prior_path = REV / 'technical/technical_training/technical-current-v1.json'
    history_path = REV / 'technical/technical_training/technical-historical-appendix-rv1019-technical-training-v1.json'
    before = [ref(p) for p in [first_path, prior_path, history_path]]
    assert before[0]['sha256'] == '376398ac0de4919142fde2a6ba0294606ba7617bcf824d5f94f715dacca0e72a'
    assert before[1]['sha256'] == '21128994cbc8e27470d7e3873485d8a19c93ade81e72090802af64ad3e3f0fd1'
    prior = load(prior_path)
    side_path = ROOT / 'docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v4.json'
    assert digest(side_path) == '1ddabe842cbb17502ae1fb08783825c16f5669c22b8c294c0e9f83046c8b19da'
    side = load(side_path); freeze = loadref(side['canonical_source_freeze']); numbered = loadref(side['section_freeze'])
    index = loadref(side['issue_inventory']); oldside = loadref(side['previous_bundle'])
    selected = loadref(side['actual_collection_snapshot']); raw = loadref(selected['raw_collection_snapshot'])
    manifest_path = REV / 'technical-stage/numbered-intake-manifest-v5.json'; manifest = load(manifest_path)
    assignments = loadref(manifest['assignments'])['tasks']
    assert len(side['sections']) == 26 and len(index['reports']) == 35 and len(index['issues']) == 5
    selected_provenance = []
    for n, receipt in enumerate(selected['collection_index']):
        matches = [i for i, row in enumerate(raw['collection_index']) if row == receipt]
        assert len(matches) == 1, (n, matches)
        selected_provenance.append({'selected_index': n, 'raw_index': matches[0], 'path': receipt['path'], 'sha256': receipt['sha256']})
    assert set(selected['report_sha256s']) == {r['sha256'] for r in selected['collection_index']}
    assert selected['status'] == raw['status'] == 'in_progress'
    frozen = {p['page_id']: p for p in freeze['pages']}
    oldf = loadref(prior['source_freeze']); oldpages = {p['page_id']: p for p in oldf['pages']}
    changed = [pid for pid in frozen if frozen[pid] != oldpages[pid]]
    assert changed == ['12.2']
    page = frozen['training']; body = (ROOT / page['snapshot']).read_text()
    assert page == oldpages['training'] and digest(ROOT / page['snapshot']) == digest(ROOT / page['source']) == prior['source_sha256']
    assert page['figures_sha256'] == {}
    old_entries = {e['lesson_id']: e for e in oldside['sections']}
    metadata, t_reports, owners = [], [], []
    for e in side['sections']:
        current, original = loadref(e['report']), loadref(e['original_report'])
        old_freeze = loadref(e['original_section_freeze'])
        sid, owner = e['lesson_id'], current['reviewer_task']; owners.append(owner)
        assert original['reviewer_task'] == owner and original['lesson_id'] == current['lesson_id'] == sid
        assert current['reviewer_context'] == original['reviewer_context'] == 'fresh'
        assert owner not in manifest['author_tasks'] and owner != '/root/omni_reader_entries'
        assigned = next(r for r in assignments if r['task'] == owner and r.get('lesson_id') == sid)
        assert assigned['role'] == 'technical' and assigned['context'] == 'fresh'
        actual_receipts = []
        for key, ck, report in [('original_report', 'original_collection', original), ('report', 'current_collection', current)]:
            matches = [r for r in selected['collection_index'] if all(r[k] == e[key][k] for k in ['path', 'sha256']) and r['received_at'] == e[ck]['received_at']]
            assert len(matches) == 1
            receipt = matches[0]
            assert receipt['reviewer_task'] == owner and receipt['lesson_id'] == sid and receipt['collection_role'] == 'technical'
            assert datetime.fromisoformat(report['created_at']) <= datetime.fromisoformat(receipt['received_at'])
            actual_receipts.append(receipt)
        assert datetime.fromisoformat(assigned['dispatch_window']['not_before']) <= datetime.fromisoformat(original['created_at'])
        if 'current_dispatch_window' in e:
            assert datetime.fromisoformat(e['current_dispatch_window']['not_before']) <= datetime.fromisoformat(current['created_at'])
        np = next(p for p in numbered['pages'] if p['page_id'] == sid)
        op = next(p for p in old_freeze['pages'] if p['page_id'] == sid)
        assert current['source_sha256'] == np['source_sha256'] == digest(ROOT / np['snapshot'])
        assert current['figure_sha256'] == np['figures_sha256']
        assert original['source_sha256'] == op['source_sha256'] == digest(ROOT / op['snapshot'])
        metadata.append({'lesson_id': sid, 'owner': owner, 'original': e['original_report'], 'current': e['report'],
                         'original_receipt': actual_receipts[0], 'current_receipt': actual_receipts[1],
                         'current_changed_since_bundle3': e != old_entries[sid], 'source_and_identity_verified': True})
        if sid.startswith('T.'):
            assert e == old_entries[sid] and section(body, sid) == (ROOT / np['snapshot']).read_text()
            assert current['verdict'] == 'pass' and all(c['status'] == 'verified' for c in current['claims'])
            for report in [original, current]:
                for a in report['artifacts']: assert digest(ROOT / a['path']) == a['sha256']
            t_reports.append({'lesson_id': sid, 'original': e['original_report'], 'current': e['report'],
                             'owner': owner, 'original_issues': original['issues'],
                             'original_issue_resolutions': current.get('original_issue_resolutions', []),
                             'same_report_bytes_and_full_slice_as_current1': True})
    assert len(set(owners)) == 26 and len(t_reports) == 11
    t9 = next(e for e in t_reports if e['lesson_id'] == 'T.9')
    assert len(t9['original_issues']) == len(t9['original_issue_resolutions']) == 1
    t4_index = loadref(prior['schema_diagnostic_history']['index'])
    for k in ['raw_initial_original', 'raw_initial_failed_selfcheck', 'formal_original_and_current']: loadref(t4_index[k])
    reused = []
    for s in prior['sources']:
        if s['kind'] == 'repository_code':
            assert digest(ROOT / s['path']) == s['sha256']; reused.append({'id': s['id'], 'path': s['path'], 'sha256': s['sha256']})
    for a in prior['artifacts']: assert digest(ROOT / a['path']) == a['sha256']
    primary = load(HERE / 'primary-source-receipts.json')
    for s in primary['sources']:
        assert digest(s['full_text_local_path']) == s['full_text_sha256']
        lines = Path(s['full_text_local_path']).read_text().splitlines()
        for x in s['excerpts']:
            lo, hi = map(int, x['lines'].split('-')); assert '\n'.join(lines[lo-1:hi]) == x['text']
    scope_path = REV / 'history-v2-review-scope-v4.json'; scope = load(scope_path)
    own_dependencies = {s['path'] for s in reused}
    affected_deps = sorted(own_dependencies & set(scope['changed_since_previous_scope']))
    assert affected_deps == []
    workflow = ROOT / '.github/workflows/windows-student-commands.yml'
    wf = workflow.read_text(); prompt = 'color=blue;shape=square;pitch=low;color?'
    assert wf.count('--prompt "' + prompt + '"') == 2 and body.count('--prompt "' + prompt + '"') == 2
    from scripts.export_course import DOCUMENTS, reading_markdown
    targets = {(ROOT / p).resolve(): n + '.md' for n, p in DOCUMENTS.items()}
    exported = reading_markdown(body, ROOT / 'course/training.md', targets, {}, 'source14-training-callback')
    fences = lambda t: re.findall(r'^```([^\n]*)\n(.*?)^```\s*$', t, re.M | re.S)
    assert fences(body) == fences(exported)
    # Current raw Notebook identity audit is explicitly file/record work, not our execution of331 lessons.
    cum_path = ROOT / 'docs/course-revision-20261029-omni/evidence/notebooks-technical-fix14-cumulative.json'
    cum = load(cum_path); assert cum['source_freeze'] == side['canonical_source_freeze']
    records = {r['lesson']: r for r in cum['records']}; methods = Counter(); code_cells = 0
    for p in freeze['pages']:
        if 'notebook' not in p: continue
        pid = p['page_id']; r = records[pid]
        assert digest(ROOT / p['notebook']) == p['notebook_sha256'] == r['notebook_sha256']
        assert digest(r['source']) == digest(r['bundle_path']) == r['executed_sha256']
        nb, executed = load(ROOT / p['notebook']), load(r['bundle_path'])
        joined = lambda value: ''.join(value) if isinstance(value, list) else value
        assert [(c['cell_type'], joined(c['source'])) for c in nb['cells']] == [(c['cell_type'], joined(c['source'])) for c in executed['cells']]
        for c in executed['cells']:
            if c['cell_type'] != 'code': continue
            code_cells += 1
            assert c['execution_count'] is not None and not any(o['output_type'] == 'error' for o in c.get('outputs', []))
        methods[r['method']] += 1
    assert len(records) == 331 and methods == {'reuse_matching_cells': 330, 'new_independent_kernel': 1}
    kernel = load(ROOT / 'outputs/omni-notebooks-technical-fix14-run/validation-kernel.json')
    assert kernel['total'] == kernel['passed'] == 1 and kernel['failed'] == 0
    # Estimate rows are current-source metadata, not evidence of actual reader completion.
    preview_path = ROOT / 'outputs/omni-reading-times-technical-fix14-preview.json'
    estimate_path = ROOT / 'course/reading-times.json'
    preview, estimates = load(preview_path), load(estimate_path)
    filter_path = ROOT / 'docs/course-revision-20261029-omni/evidence/preview-estimates-technical-fix14-v1.json'
    filter_receipt = load(filter_path)
    assert digest(preview_path) == filter_receipt['preview_metadata_sha256'] and digest(estimate_path) == filter_receipt['original_metadata_sha256']
    matching = [r for r in estimates['pages'] if r['source_sha256'] == frozen[r['page_id']]['source_sha256'] and r['figures_sha256'] == frozen[r['page_id']]['figures_sha256']]
    assert preview['pages'] == matching and len(matching) == 367
    assert sorted({r['page_id'] for r in estimates['pages']} - {r['page_id'] for r in matching}) == ['12.2', 'first-steps']
    # Reader reports are parsed only for page identity and artifact references. No answers/checks/notes content is read.
    catalogue_path = REV / 'reader-evidence-catalogue-technical-fix07-v1.json'; catalogue = load(catalogue_path)
    overlays, reader_refs = {}, []
    for n in [5, 6, 7, 8]:
        rp = REV / f'reader-late-callback-intake-v{n}.json'; intake = load(rp); reader_refs.append(ref(rp))
        for r in intake['entries']:
            report = loadref(r)
            if r['kind'] != 'reader-current-v1': continue
            for pid in r.get('page_ids', [r.get('page_id')]):
                p = next(x for x in report['pages'] if x['page_id'] == pid)
                assert p['source_sha256'] == frozen[pid]['source_sha256'] and p['figures_sha256'] == frozen[pid]['figures_sha256']
                overlays[pid] = {'source_sha256': p['source_sha256'], 'report': {'path': r['path'], 'sha256': r['sha256']}}
    retained = []
    for r in catalogue['rows']:
        pid = r['page_id']
        if pid in ['first-steps', '12.2'] or pid in overlays or r['status'] != 'current_owner_source_identity': continue
        p = node(loadref(r['current_report']), r['current_pointer'])
        assert p['source_sha256'] == frozen[pid]['source_sha256'] and p.get('figures_sha256', p.get('figure_sha256')) == frozen[pid]['figures_sha256']
        retained.append(pid)
    assert len(retained) == 358 and len(overlays) == 5
    # Live local preview only. No browser, cloud runtime or publication inference.
    site = ROOT / 'outputs/omni-review-site-technical-fix14'; http = []
    class Text(HTMLParser):
        def __init__(self): super().__init__(); self.parts = []
        def handle_data(self, data): self.parts.append(data)
    for tail, local in [('build-info.json', site / 'build-info.json'), ('training/', site / 'training/index.html'),
                        ('notebooks/12/12.2.ipynb', site / 'notebooks/12/12.2.ipynb')]:
        with urlopen('http://127.0.0.1:8775/' + tail, timeout=10) as response:
            raw_http = response.read(); assert response.status == 200
        assert hashlib.sha256(raw_http).hexdigest() == digest(local)
        if tail == 'training/':
            parser = Text(); parser.feed(raw_http.decode()); text = ''.join(parser.parts)
            assert prompt in text and 'greedy' in text
        http.append({'url': 'http://127.0.0.1:8775/' + tail, 'status': 200, 'sha256': digest(local)})
    build = load(site / 'build-info.json'); assert build['reading_time_pages'] == 367 and build['publication_ready'] is False
    runs = []
    commands = [
        ['uv', 'run', '--extra', 'cpu', 'python', 'scripts/omni_revision/review_setup.py', 'check-numbered', '--manifest', str(manifest_path.relative_to(ROOT)), '--intake', '--output', str((HERE / 'numbered-intake-current-v2.json').relative_to(ROOT))],
        ['uv', 'run', '--extra', 'cpu', 'pytest', '-q', 'tests/test_windows_student_commands.py::test_page_windows_commands_match_workflow'],
    ]
    for command in commands:
        p = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=50)
        row = {'argv': command, 'completed_at': now(), 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
        assert p.returncode == 0, row
        runs.append(row)
    # Historical callback semantically rechecks unchanged per-entry device contracts, with fresh parser/default/rejection probes.
    history = load(history_path)
    for a in history['artifacts']: assert digest(ROOT / a['path']) == a['sha256']
    for s in history['sources']:
        if s['kind'] == 'repository_code': assert digest(ROOT / s['path']) == s['sha256']
    historical_quote = next(l for l in body.splitlines() if l.startswith('- **Python 怎麼寫。**'))
    for row in history['historical_issue_rechecks']: assert row['pages']['training']['source_quote'] == historical_quote
    from tiny_perceptron.training import choose_device
    with patch('torch.cuda.is_available', return_value=True), patch('torch.backends.mps.is_available', return_value=True):
        assert str(choose_device()) == 'cuda' and str(choose_device('cpu')) == 'cpu'
    history_runs = []
    for argv in [['scripts/quantize.py', '/workspace/work/omni-technical-training-v1/checkpoints/attributes.pt', '--device', 'cpu'],
                 ['scripts/prepare_data.py', '--device', 'cpu'], ['scripts/fetch_training_assets.py', '--device', 'cpu']]:
        p = subprocess.run([sys.executable, *argv], cwd=ROOT, text=True, capture_output=True, timeout=20)
        assert p.returncode == 2 and 'unrecognized arguments: --device cpu' in p.stderr
        history_runs.append({'argv': [sys.executable, *argv], 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr})
    assert [ref(p) for p in [first_path, prior_path, history_path]] == before
    result = {'schema_version': 1, 'reviewer_task': TASK, 'started_at': started, 'completed_at': now(),
              'scope': 'bounded source14/bundle4 whole-training callback with explicit unchanged109-claim evidence reuse; not new whole-book blind reading',
              'prior_reports_preserved': before, 'source_freeze': side['canonical_source_freeze'], 'numbered_freeze': side['section_freeze'],
              'bundle': ref(side_path), 'issue_index': side['issue_inventory'], 'canonical_delta': changed,
              'training_body_and_all11_slices_unchanged': True, 'all26_metadata_receipts': metadata, 'training_section_lineages': t_reports,
              'collection': {'selected': side['actual_collection_snapshot'], 'raw': selected['raw_collection_snapshot'],
                             'selected_rows': len(selected_provenance), 'raw_rows': len(raw['collection_index']),
                             'selected_verbatim_row_provenance': selected_provenance, 'status': 'in_progress; not sealed'},
              'reuse': {'repository_sources': reused, 'all_prior_artifacts_checked': len(prior['artifacts']),
                        'primary_originals_and_excerpts_unchanged': len(primary['sources']),
                        'all109_claims': 'Reassessed in unchanged full-page context; local CPU mechanism, historical L4 results, unknown new training and unmeasured speeds retain prior limits.'},
              'current_code_scope': {'inventory': ref(scope_path), 'listed_files': len(scope['checked_file_scope']), 'changed_direct_prior_sources': affected_deps,
                                     'newly_read_relevant': [ref(workflow), ref(ROOT / 'tests/test_windows_student_commands.py')],
                                     'limit': 'Windows training command synchronization and helper receipt behavior only; no210-file tool audit or Windows execution'},
              'exporter_fences': {'source': ref(ROOT / 'scripts/export_course.py'), 'fenced_blocks': len(fences(body)), 'unchanged_block_bytes': True},
              'notebook_scope': {'cumulative_receipt': ref(cum_path), 'total': 331, 'methods': dict(methods), 'recorded_code_cells': code_cells,
                                  'kernel_validation': ref(ROOT / 'outputs/omni-notebooks-technical-fix14-run/validation-kernel.json'),
                                  'limit': 'Personally checked raw identities/cell sources/error-free records. Author kernel1 and330matching reuse are not my331 fresh executions.'},
              'reader_identity_scope': {'catalogue': ref(catalogue_path), 'intakes': reader_refs, 'retained': 358, 'overlays': overlays,
                                        'exact_current_source_identities': 363, 'pending_full_readers': ['first-steps', '12.2'],
                                        'limit': 'Metadata identity only; no reader answers or semantic reading used as technical proof.'},
              'estimates': {'filter': ref(filter_path), 'preview': ref(preview_path), 'original': ref(estimate_path), 'matching_rows': 367,
                            'omitted_stale': ['12.2', 'first-steps'], 'limit': 'Exact retained AI estimate rows, not reader counts, a new estimate or publication-ready set.'},
              'local_preview_http': http, 'build_info': build, 'actual_commands': runs,
              'historical_reassessment': {'old_appendix': ref(history_path), 'source_quote': historical_quote, 'fresh_no_device_rejections': history_runs,
                                         'device_semantics': 'Same8implementation contracts and samefullpagecontext; auto limitedto train/evaluate/infer, fixed defaultcpu, quantize defaultcpu, data/quantCLIs noflag. New source14/bundle4 metadata does not reverse either originalnecessary fix.',
                                         'groups8_excluded': True},
              'unresolved_training_issues': [], 'fullstage': 'pending actual whole-collector receipts/seals, properwave02 readers and final assembly; no gatepass'}
    path = HERE / 'callback-current-v2.json'; save(path, result)
    print(json.dumps({'result': ref(path), 'sections': len(t_reports), 'owners': len(set(owners)), 'commands': [r['exit_code'] for r in runs],
                      'reader_identities': 363, 'AI_estimates': 367, 'notebook_methods': dict(methods)}, ensure_ascii=False))


if __name__ == '__main__': main()
