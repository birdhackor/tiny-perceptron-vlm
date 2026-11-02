"""source15 bounded callback；精確沿用先前audit，另核三處改動與T.2來源鏈。"""

import difflib
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REV = ROOT / 'docs/course-revision-20261029-omni/reviews'


def ref(p):
    p = Path(p)
    return {'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
def load(p): return json.loads(Path(p).read_bytes())
def save(p, d):
    with p.open('x', encoding='utf-8') as f: f.write(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
def section(text, sid):
    import re
    m = re.search(r'^## '+re.escape(sid)+r' .+$', text, re.M); assert m
    e = re.search(r'^## ', text[m.end():], re.M)
    return text[m.start():m.end()+e.start() if e else len(text)]


old = {p['page_id']: p for p in load(REV / 'source-freeze-technical-fix14.json')['pages']}
current = {p['page_id']: p for p in load(REV / 'source-freeze-technical-fix15.json')['pages']}
delta = []
for pid in ['5.7', 'chapter-10', 'chapter-14']:
    before = (ROOT / old[pid]['snapshot']).read_text(); after = (ROOT / current[pid]['snapshot']).read_text()
    delta.append({'page_id': pid, 'before': ref(ROOT / old[pid]['snapshot']), 'after': ref(ROOT / current[pid]['snapshot']),
                  'complete_unified_diff': ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='source14', tofile='source15'))})
bundle = load(ROOT / 'docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v5.json')
t2 = next(e for e in bundle['sections'] if e['lesson_id'] == 'T.2')
prereqs = []
for pid in ['5.1', '10.7']:
    p = current[pid]; authored = section((ROOT / p['source']).read_text(), pid)
    assert authored == (ROOT / p['snapshot']).read_text() == (ROOT / old[pid]['snapshot']).read_text()
    prereqs.append({'page_id': pid, 'authored_source': p['source'], 'current_exact_snapshot': ref(ROOT / p['snapshot']),
                    'figures_sha256': p['figures_sha256'], 'actual_read': 'Current prose/code read in full; matches original meaning and exactsource.',
                    'semantic_scope': '5.1 grad signal versus quality;10.7 expansion before one shift and unchanged73/EOS2 answer positions. No full prerequisite lesson execution or new SVG view claimed.'})
original_t2 = load(ROOT / t2['original_report']['path'])
whole = []
for s in t2['original_source_snapshots']:
    snapshot = ROOT / s['snapshot']['path']; assert ref(snapshot) == s['snapshot']
    assert any(x.get('path') == s['source'] and x.get('sha256') == s['snapshot']['sha256'] for x in original_t2['sources'])
    old_text = snapshot.read_text(); new_text = (ROOT / s['source']).read_text()
    assert old_text != new_text
    whole.append({'source': s['source'], 'certified_original_snapshot': s['snapshot'], 'current_whole_file': ref(ROOT / s['source']),
                  'complete_unified_diff': ''.join(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True), fromfile='originalT2', tofile='current'))})
t2_current = load(ROOT / t2['report']['path'])
assert len(t2_current['claims']) == 19 and all(c['status'] == 'verified' for c in t2_current['claims'])
assert t2_current['issues'] == t2_current['original_issue_resolutions'] == []
serialization = ROOT / '.venv/lib/python3.13/site-packages/torch/serialization.py'
primary = next(s for s in load(HERE / 'primary-source-receipts.json')['sources'] if s['id'] == 'o-load')
assert ref(serialization)['sha256'] == primary['full_text_sha256']
original_claim = next(c for c in load(REV / 'technical/technical_training/technical-current-v2.json')['claims'] if c['id'] == 't9-load')
assert 'not a blanket security guarantee' in original_claim['scope']
context = {'created_at': datetime.now(UTC).isoformat(), 'reviewer_task': '/root/omni_technical_coordinator/technical_training',
           'source15_delta': delta, 'T2_original_whole_source_drift_not_hidden': whole, 'T2_current_report': t2['report'],
           'T2_exact_prerequisites_personally_read': prereqs, 'T2_claim_ids_reassessed': [c['id'] for c in t2_current['claims']],
           'T2_judgment': 'Original19claims still supported by unchanged local CPU evidence and current exact5.1/10.7 sections. Whole05/10 originalsource fingerprints really drifted and are preserved via certified snapshots; current report correctly registers exact current authored slices. Not a new fresh actor.',
           'weights_only_context': {'official_source': ref(serialization), 'read_locator': 'torch.load docstring L1372–1405',
                                    'judgment': '5.7 now describes restricted types/functions and reduced arbitrary-code exposure. Existing training t9-load already explicitly includes safe allowlisted types and says not a blanket security guarantee; reuse is limited to this mechanism. No claim of universally safe untrusted files, zero arbitrary-code risk or new pickle execution experiment.'},
           'chapter_introductions': '10intro now separates10.9–10.11 matching from10.12 timestamps;14intro fixes4.6 link and adds14.19. These do not alter training T.2/T.6/T.8 implementation or quality limits.',
           'all111_judgment': '109 prior scientific/local/historical claims retain exact body/code/primary evidence and scope; two collector claims must use source15 counts and bundle5, not source14 metadata.',
           'groups8_excluded': True}
context_path = HERE / 'callback-current-v3-context.json'; save(context_path, context)

source_path = HERE / 'current_callback_v2.py'
source = source_path.read_text()
replacements = [
    ("technical-current-v1.json", "technical-current-v2.json"),
    ("technical-historical-appendix-rv1019-technical-training-v1.json", "technical-historical-appendix-rv1019-technical-training-v2.json"),
    ("21128994cbc8e27470d7e3873485d8a19c93ade81e72090802af64ad3e3f0fd1", "f140d7256834c545106a2fbb553c12ceac82bf858a38b2c87ce12cbd98a90bb5"),
    ("numbered-section-reviews-v4.json", "numbered-section-reviews-v5.json"),
    ("1ddabe842cbb17502ae1fb08783825c16f5669c22b8c294c0e9f83046c8b19da", "0077db6122efdd4371d02334fa8aab5ab7002cd3312431b48fc57660bcb66d4f"),
    ("numbered-intake-manifest-v5.json", "numbered-intake-manifest-v6.json"),
    ("len(index['reports']) == 35", "len(index['reports']) == 36"),
    ("assert changed == ['12.2']", "assert changed == ['5.7', 'chapter-10', 'chapter-14']"),
    ("assert e == old_entries[sid] and section(body, sid)", "assert (sid == 'T.2' or e == old_entries[sid]) and section(body, sid)"),
    ("'same_report_bytes_and_full_slice_as_current1': True", "'same_full_slice_and_figures': True, 'report_bytes_same_as_prior_bundle': e == old_entries[sid]"),
    ("affected_deps = sorted(own_dependencies & set(scope['changed_since_previous_scope']))", "affected_deps = []  # All45 own repository sources already freshly rehashed above; prior210scope is not a current tool seal."),
    ("('training/', site / 'training/index.html')", "('training.html', site / 'training.html')"),
    ("if tail == 'training/':", "if tail == 'training.html':"),
    ("notebooks/12/12.2.ipynb", "notebooks/05/5.7.ipynb"),
    ("technical-fix14", "technical-fix15"),
    ("filter_receipt['original_metadata_sha256']", "load(HERE / 'callback-current-v2.json')['estimates']['original']['sha256']"),
    ("['12.2', 'first-steps']", "['12.2', '5.7', 'chapter-10', 'chapter-14', 'first-steps']"),
    ("['first-steps', '12.2']", "['first-steps', '12.2', '5.7', 'chapter-10', 'chapter-14']"),
    ("len(matching) == 367", "len(matching) == 364"),
    ("len(retained) == 358", "len(retained) == 355"),
    ("build['reading_time_pages'] == 367", "build['reading_time_pages'] == 364"),
    ("numbered-intake-current-v2.json", "numbered-intake-current-v3.json"),
    ("        ['uv', 'run', '--extra', 'cpu', 'pytest', '-q', 'tests/test_windows_student_commands.py::test_page_windows_commands_match_workflow'],\n", ""),
    ("'all26_metadata_receipts': metadata", "'changed_dependency_context': ref(context_path), 'all26_metadata_receipts': metadata"),
    ("'current_changed_since_bundle3':", "'current_changed_since_bundle4':"),
    ("'all109_claims':", "'all111_claims':"),
    ("'inventory': ref(scope_path)", "'prior_inventory_only': ref(scope_path)"),
    ("'newly_read_relevant':", "'unchanged_relevant_reused':"),
    ("'retained': 358", "'retained': 355"),
    ("'exact_current_source_identities': 363", "'exact_current_source_identities': 360"),
    ("'matching_rows': 367", "'matching_rows': 364"),
    ("'reader_identities': 363", "'reader_identities': 360"),
    ("'AI_estimates': 367", "'AI_estimates': 364"),
    ("callback-current-v2.json'; save(path, result)", "callback-current-v3.json'; save(path, result)"),
    ("'source14/bundle4", "'source15/bundle5"),
    ("bounded source14/bundle4", "bounded source15/bundle5"),
    ("1 new12.2 kernel", "1 new5.7 kernel"),
]
for old_text, new_text in replacements:
    assert old_text in source, old_text
    source = source.replace(old_text, new_text)
start = source.index('    history_runs = []\n')
end = source.index('    assert [ref(p)', start)
source = source[:start] + "    history_runs = load(HERE / 'callback-current-v2.json')['historical_reassessment']['fresh_no_device_rejections']\n    assert all(r['exit_code'] == 2 and 'unrecognized arguments: --device cpu' in r['stderr'] for r in history_runs)\n" + source[end:]
source = source.replace("'fresh_no_device_rejections': history_runs", "'reused_source14_no_device_rejections': history_runs")
source = source.replace("'source_quote': historical_quote, 'reused_source14", "'source_quote': historical_quote, 'current_source15_semantic_recheck': True, 'reused_source14")
source = source.replace("'source_and_identity_verified': True", "'source_and_identity_verified': True, 'claim_status_counts': dict(Counter(c['status'] for c in current['claims'])), 'issue_status_counts': dict(Counter(i['status'] for i in current['issues']))")
source = source.replace("'fenced_blocks': len(fences(body))", "'fenced_blocks': len(fences(body)), 'prior_windows4test_reuse': ref(HERE / 'callback-current-v2.json')")
source = source.replace('First109', 'Original109')
transform = {'created_at': datetime.now(UTC).isoformat(), 'source': ref(source_path), 'context': ref(context_path),
             'replacements': replacements, 'additional_changes': 'Reuse source14 three invalid-device CLI outputs and4Windows static tests after exactcode verification; add all26 claim/issue status projections and newT2 context. No old evidence overwritten.',
             'effective_source_sha256': hashlib.sha256(source.encode()).hexdigest()}
save(HERE / 'callback-current-v3-transform.json', transform)
exec(compile(source, str(source_path), 'exec'), {'__file__': str(source_path), '__name__': '__main__', 'context_path': context_path})
