"""Assemble owner's source-only callback; retain original body-reading traces unchanged."""
import copy
import datetime
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[4]
A = 'docs/technical-reviews/artifacts/p7_technical_f/'
OLD = 'docs/course-revision-20261007-phase7/reviews/freeze-04/reports/technical/f/p7_technical_f-recheck-20261007T055405.json'
MANIFEST = 'docs/course-revision-20261007-phase7/reviews/freeze-06/manifest.json'
def read(path):
    return json.loads((R / path).read_text())
def sha(path):
    return hashlib.sha256((R / path).read_bytes()).hexdigest()

old = read(OLD)
report = copy.deepcopy(old)
proof = read(A + 'source06-reading-time-contract.json')
observed = json.loads(proof['stdout'])
assert proof['exit_code'] == 0 and observed['unchanged_primary_page_count'] == 54
assert observed['script_sha256'] == sha('scripts/reading_time.py')
manifest = read(MANIFEST)
frozen = {p['page_id']: p for p in manifest['pages']}
assert all(p['source_sha256'] == frozen[p['page_id']]['source_sha256'] and p['figures_sha256'] == frozen[p['page_id']]['figures_sha256'] for p in old['pages'])
now = datetime.datetime.now(datetime.timezone.utc)
report['created_at'] = now.isoformat()
report['manifest_sha256'] = sha(MANIFEST)
report['source_only_predecessor_report'] = {'path': OLD, 'sha256': sha(OLD), 'verdict': old['verdict'], 'preserved_unchanged': True}
report['source_only_predecessor_summary'] = old['summary']
report['summary'] = '完整54頁原owner技術報告仍pass。全部正文／圖檔SHA與freeze-04本人報告相同；此輪只回查publishing引用的新reading_time.py程式，更新實際來源SHA／版本／定位及短CPU契約證據。原53頁紀錄、兩輪完整閱讀trace、初判／三issue處置及所有失敗／判分／provenance與視覺限制保持不變，未新增閱讀trace或全組重讀聲稱。'
report['source_only_callback'] = {'page_id': 'publishing', 'source_path': 'scripts/reading_time.py', 'old_source_sha256': observed['only_stale_repository_source'][0]['old_sha256'], 'new_source_sha256': sha('scripts/reading_time.py'), 'manifest_sha256': sha(MANIFEST), 'all_54_body_and_figure_versions_unchanged': True, 'proof': {'path': A + 'source06-reading-time-contract.json', 'sha256': sha(A + 'source06-reading-time-contract.json')}, 'reading_scope': 'Personally re-read revised reading_time.py full source and unchanged docs/publishing.md after actual GO. Source-only followup, no new incremental reading session, no recreated checkpoints, no new whole-group reading or visual-viewing claim. Existing original publishing body trace remains valid.', 'actual_execution_scope': 'CPU AST and four synthetic render cases; existing20 reading-time tests; actual validate --help and validate --executed. No model/GPU/training/site build/all-notebook execution/GitHub Action/remote publication.', 'checkpoint_and_trace_scope': 'All original counts, completion metadata, immutable trace paths/hashes, necessary resolved issues and original limitations retained exactly.'}
report['execution_scope']['source_only_callback'] = report['source_only_callback']['actual_execution_scope']

page = next(p for p in report['pages'] if p['page_id'] == 'publishing')
source = next(s for s in page['sources'] if s['id'] == 'reading-time')
build = next(c for c in page['claims'] if c['id'] == 'build')
page['source_only_prior_contract'] = {'report': {'path': OLD, 'sha256': sha(OLD)}, 'reading_time_source': copy.deepcopy(source), 'build_claim': copy.deepcopy(build), 'scope': 'Historical source and claim retained for comparison; active current source binding is in sources, not this prior record.'}
source.update(sha256=sha('scripts/reading_time.py'), version='SHA256:' + sha('scripts/reading_time.py'), inspection_note='Personally read current source: load_estimates164–202 rejects invalid/stale/missing complete metadata; routes205–233 and totals236–249 deduplicate/round outward; render_page252–296 inserts banner and closed details on index/course only while preserving authored body; main299–327 validate parser accepts --metadata/--routes/--executed and require_complete=True. Actually ran own synthetic source/render audit,20 existing tests, validate --help and validate --executed. Supports source/CLI/metadata/render contract only, not empirical reading speeds or deployment.')
specs = [
 ('reading-time-source06-contract', 'source06-reading-time-contract.json', 'Actual AST/source hashes and four owner synthetic render cases;54 frozen source/figure byte checks; single stale repository source identified.', 'exit0: correct body preservation; closed home/course details; missing estimates do not fabricate minutes.'),
 ('reading-time-source06-tests', 'source06-reading-time-tests.json', 'Actual existing reading-time test file execution, including stale source/figure refusal, input validation, route deduplication, render preservation and mocked build refusal.', 'exit0:20 passed in0.22s; exporter test uses mocked site builder, not full site build.'),
 ('reading-time-source06-help', 'source06-reading-time-validate-help.json', 'Actual revised reading_time.py validate --help invocation.', 'exit0: --metadata, --routes, --executed all displayed.'),
 ('reading-time-source06-validate', 'source06-reading-time-validate.json', 'Actual revised reading_time.py validate --executed against current repository metadata and routes.', 'exit0:325 pages,4 routes; current source and figure SHA validation accepted. This is metadata validation, not empirical estimates or execution of notebooks.')
]
for artifact_id, filename, description, result in specs:
    path = A + filename
    record = read(path)
    assert record['exit_code'] == 0 and record['inputs']['scripts/reading_time.py'] == source['sha256']
    page['artifacts'].append({'id': artifact_id, 'kind': 'execution', 'path': path, 'sha256': sha(path), 'description': description, 'command': json.dumps(record['command']), 'result': result, 'environment': record['environment']})
page['sources'].append({'id': 'reading-time-source06-run', 'kind': 'execution', 'title': 'Owner current reading-time source/render CPU check', 'verified': True, 'artifact_id': 'reading-time-source06-contract'})
for evidence in build['evidence']:
    if evidence['source_id'] == 'reading-time':
        evidence.update(locator='main299–327 (validate flags306–309; build inventory319; complete metadata325); load_estimates164–202; render_page252–296', supports='Current validate --executed source/CLI and source-bound metadata contract; render-only insertion preserves authored content and collapses index/course planning totals.')
build['artifact_ids'].extend([a[0] for a in specs])
build['verification']['observed'] += '; current source-only callback: AST/render audit4 cases,20 tests passed, validate --help flags verified, validate --executed exit0 for325 pages/4 routes.'
build['verification']['details'] += ' Current reading-time source separately personally re-read and executed; old4 AST/help record retained as original evidence, not used to claim new reading_time.py was executed then. New validate only checks metadata source bindings; tests mock full site builder.'
page['claims'].append({'id': 'reading-time-source06-contract', 'kind': 'software', 'statement': 'Current reading-time renderer preserves original body and uses a closed details block for index/course totals and method; incomplete estimates do not fabricate durations. validate --executed requires every current canonical page estimate to match source and figure SHA and accepts its documented flags.', 'location': 'publishing §2 reading_time.py validate --executed maintenance command; revised source render_page252–296/main299–327', 'scope': 'Personally read source and bounded CPU AST/render fixtures,20 existing tests, actual help and325-page/4-route metadata validation. No empirical reading speed judgment, no complete site rebuild, no new screenshot or notebook/GitHub Action/deployment claim.', 'status': 'verified', 'evidence': [{'source_id': 'reading-time', 'locator': 'load_estimates164–202;render_page252–296;main299–327', 'supports': 'Actual rejection/presentation/CLI branches and current validation bindings'}, {'source_id': 'reading-time-source06-run', 'locator': 'stdout.ast_function_line_spans and render_cases; only_stale_repository_source and actual_snapshot_and_figure_hashes_checked', 'supports': 'Personally executed four synthetic render branches and actual byte/hash checks; meaningful scope distinct from empirical estimates'}], 'artifact_ids': [a[0] for a in specs], 'verification': {'method': 'executed', 'expected': 'CLI documented flags work; current source-bound full metadata accepted; fixtures keep body intact, details closed, no invented missing estimates.', 'observed': 'AST spans correct; four render cases pass;20 tests pass;help lists three flags;actual validate --executed exit0:325 pages/4 routes.', 'details': 'CPU Python3.13.5. Synthetic2–5 minute inputs are test fixtures only. Existing exporter test mocks subprocess builder. Validate reads current estimates for schema/hash matching without certifying reader comprehension or empirical timing; no new reading trace created.'}})
page['checks']['factual_accuracy']['claim_ids'].append('reading-time-source06-contract')
page['checks']['source_verification']['details'] += ' Current reading_time.py personally re-read with exact new locators; new actual CPU source/render/help/test/validate evidence, not SHA-only replacement.'
page['checks']['source_verification']['claim_ids'].append('reading-time-source06-contract')
page['checks']['limitations']['details'] += ' Source-only callback adds CPU metadata/render validation only; no new all-page reading trace, new image views or full site build.'
page['checks']['limitations']['claim_ids'].append('reading-time-source06-contract')
page['summary'] += ' Source-only followup:新版reading_time.py来源与parser/render契约本人真核，20tests／4合成render／help／325页4路线validate实跑通过；正文及原完整阅读／视觉记录未变。'
page['source_only_callback'] = copy.deepcopy(report['source_only_callback'])
assert report['trace_files'] == old['trace_files'] and report['issues'] == old['issues'] and report['completion'] == old['completion']
assert all(p == old['pages'][i] for i, p in enumerate(report['pages']) if p['page_id'] != 'publishing')
out = 'docs/course-revision-20261007-phase7/reviews/freeze-06/reports/technical/f/p7_technical_f-source-only-' + now.strftime('%Y%m%dT%H%M%S') + '.json'
(R / out).parent.mkdir(parents=True, exist_ok=True)
with (R / out).open('x') as f:
    f.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'path': out, 'sha256': sha(out), 'verdict': report['verdict'], 'pages': len(report['pages']), 'unchanged_page_records': 53}, ensure_ascii=False))
