"""Original reviewer's source-only callback with exact prior report preservation."""
from pathlib import Path
import copy, datetime, hashlib, json

ROOT = Path.cwd()
BASE = ROOT / 'docs/technical-reviews/artifacts/p6-public-a'
REPORT = ROOT / 'docs/technical-reviews/public-pages/p6-a.json'
PRE = BASE / 'history/pre-ci-dependency-callback-report.json'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    return json.loads(Path(p).read_bytes())
assert sha(PRE) == 'a35e8963d474f2779ef5cff77feeed05256b21447b922f0bd60ada2c84d5136f'
if sha(REPORT) != sha(PRE):
    own_current = load(REPORT)
    assert own_current['source_dependency_callbacks'][-1]['prior_report_sha256'] == sha(PRE)
    assert own_current['source_dependency_callbacks'][-1]['reviewer'] == '/root/p6_fact_public_a'
before = load(PRE)
r = copy.deepcopy(before)
m = load(BASE / 'ci-dependency/measurements.json')
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
new_sources = {}
for sid in ['ci', 'pages', 'evaluate']:
    s = next(x for x in r['sources'] if x['id'] == sid)
    observed = m['ci_source'] if sid == 'ci' else m['additional_changed_sources'][sid]
    assert s['sha256'] == observed['old_sha256']
    s['source_history'] = [{'sha256': s['sha256'], 'version': s['version'],
        'inspection_note': s['inspection_note'], 'frozen_snapshot_path': observed['old_snapshot_path'],
        'formal_report_path': str(PRE.relative_to(ROOT)), 'formal_report_sha256': sha(PRE)}]
    s['sha256'] = observed['new_sha256']
    s['version'] = 'Current source personally reverified by the original reviewer in a source-dependency-only callback: ' + now
    s['current_frozen_snapshot_path'] = observed['current_snapshot_path']
    if sid == 'ci':
        s['inspection_note'] = 'Full current workflow personally read. The only change is pytest-step ONEDNN_MAX_CPU_ISA=AVX2 on Windows, empty on other OSes. Linux/Windows/macOS matrix, check_env, pytest command and test selection/skip policy remain unchanged. This is source-contract evidence; no new native Windows pass is asserted.'
    else:
        s['inspection_note'] = observed['verification']
    new_sources[sid] = copy.deepcopy(s)
official_ids = {
    'onednn-5689.json': 'onednn_amx_issue', 'onednn-5689-comments.json': 'onednn_amx_comments',
    'runner-images-14483.json': 'runner_amx_issue', 'onednn-dispatcher-control.md': 'onednn_dispatch_control',
    'onednn-exact-isa.cpp': 'onednn_isa_implementation', 'onednn-exact-utils.cpp': 'onednn_env_implementation'}
notes = {
    'onednn_amx_issue': 'Original reported Windows/EMR BF16/AMX illegal instruction and reproduction details, not this project test output.',
    'onednn_amx_comments': 'Original contributor comment 5110685239 distinguishes working bare-metal AMX from suspected VM support; investigation 5136593460 identifies inconsistent CPUID and contributor 5136659092 calls it a hypervisor bug.',
    'runner_amx_issue': 'Original runner-images report shows AMX advertised in CPUID.7/XCR0 while CPUID.1D tile fields are zero on Windows-2025 EMR.',
    'onednn_dispatch_control': 'Complete official fixed-commit dispatcher-control document read: AVX2 environment limit, build-enable prerequisite, ISA partial order and function-setting precedence.',
    'onednn_isa_implementation': 'Original fixed-commit init_max_cpu_isa reads MAX_CPU_ISA behind DNNL_ENABLE_MAX_CPU_ISA and accepts the avx2 option.',
    'onednn_env_implementation': 'Original fixed-commit getenv_string_user tries ONEDNN_ and DNNL_ prefixes and lowercases the option, accepting ONEDNN_MAX_CPU_ISA=AVX2.'}
for s in m['official_original_sources']:
    sid = official_ids[s['name']]
    r['sources'].append({'id': sid, 'kind': 'official_source', 'title': s['name'],
        'url': s['url'], 'path': s['path'], 'sha256': s['sha256'],
        'version': 'Fixed maintainer commit in URL' if 'raw.githubusercontent.com' in s['url'] else 'Original official repository issue/API snapshot personally read',
        'accessed_on': '2026-10-06', 'verified': True, 'checked_original': True,
        'authority_reason': 'Original oneDNN maintainer repository/code/documentation or original GitHub runner-images issue, rather than a project-authored explanation.',
        'inspection_note': notes[sid]})
measurement_path = str((BASE / 'ci-dependency/measurements.json').relative_to(ROOT))
r['sources'].append({'id': 'ci_dependency_measurements', 'kind': 'execution', 'title': measurement_path,
    'path': measurement_path, 'sha256': sha(ROOT / measurement_path), 'version': now,
    'verified': True, 'checked_original': True,
    'inspection_note': 'Read-only exact delta/AST and SHA verification by the original reviewer: seven unchanged public pages; 184 unchanged other source/artifact dependencies; three changed source contracts. No model or native Windows execution.'})
def artifact(aid, path, kind, description):
    r['artifacts'].append({'id': aid, 'path': path, 'sha256': sha(ROOT / path), 'kind': kind, 'description': description})
artifact('pre_ci_callback_report', str(PRE.relative_to(ROOT)), 'review_history', 'Opaque byte-identical report before this source-only callback; authored page judgments, initial issue history and completed reinspection remain preserved.')
artifact('ci_dependency_execution', measurement_path, 'execution', 'Exact delta/AST/source-hash checks; no tests or native Windows run.')
r['artifacts'][-1].update(command=m['command'], result='All assertions passed: only three expected source dependencies changed, seven public pages unchanged, 184 other source/artifact dependencies unchanged; official dispatch mechanism verified without claiming a new Windows pass.', environment={'device': 'cpu', 'model_execution': False, 'native_windows_execution': False})
artifact('ci_dependency_script', str((BASE / 'verify_ci_dependency_callback.py').relative_to(ROOT)), 'execution_code', 'Exact source-only verification script, including AST equality of unchanged scoring/threshold paths.')
artifact('ci_dependency_stdout', str((BASE / 'ci-dependency.stdout.txt').relative_to(ROOT)), 'execution_stdout', 'Actual successful source-only verification output.')
artifact('ci_dependency_frozen_workflow', m['ci_source']['current_snapshot_path'], 'source_snapshot', 'Personally read complete current CI workflow bytes.')
for sid, observed in m['additional_changed_sources'].items():
    artifact(sid + '_dependency_frozen_source', observed['current_snapshot_path'], 'source_snapshot', 'Current source dependency bytes, verified against the preserved original and its exact delta.')
claim = {'id': 'r15', 'kind': 'software',
    'statement': 'GitHub Actions保留Linux、macOS與Windows核心測試；目前Windows pytest步驟設定oneDNN AVX2分派上限，沒有刪減pytest測試或改dtype。',
    'location': 'README.md:102，維護段的核心測試契約', 'status': 'verified',
    'evidence': [
        {'source_id': 'ci', 'locator': 'jobs.test.strategy.matrix; check_env and pytest steps; pytest-step env', 'supports': 'All three OS matrix entries and commands remain; only Windows gets the oneDNN AVX2 ceiling.'},
        {'source_id': 'onednn_dispatch_control', 'locator': 'Runtime Controls table and build-time prerequisite', 'supports': 'AVX2 is an officially supported oneDNN dispatch ceiling when the runtime-control feature is enabled; function settings take precedence.'},
        {'source_id': 'onednn_env_implementation', 'locator': 'getenv_string_user lines 112–125', 'supports': 'ONEDNN_ prefix and lowercase option conversion accept the configured variable/value.'},
        {'source_id': 'onednn_isa_implementation', 'locator': 'init_max_cpu_isa lines 31–63', 'supports': 'MAX_CPU_ISA feature gate and avx2 option limit oneDNN ISA dispatch.'},
        {'source_id': 'onednn_amx_comments', 'locator': 'comments 5110685239,5136593460,5136659092', 'supports': 'Original investigation/contributor responses explain the Windows runner AMX/CPUID fault that motivates avoiding AMX dispatch.'},
        {'source_id': 'runner_amx_issue', 'locator': '/title,/body CPUID.7/XCR0 versus CPUID.1D', 'supports': 'Original runner report documents inconsistent AMX capability advertisement.'},
        {'source_id': 'ci_dependency_measurements', 'locator': '/ci_source', 'supports': 'After deleting the added env block, the current workflow is byte-identical to the preserved prior workflow; test command/selection/skip policy is unchanged by this edit.'}],
    'artifact_ids': ['ci_dependency_execution', 'ci_dependency_frozen_workflow'],
    'scope': 'This verifies the current CI configuration and the bounded official mechanism. The variable limits oneDNN dispatch, subject to build support and runtime API precedence; it does not change tensor dtype or certify every Windows CPU. No new native Windows run or pass was checked or claimed. Existing original macOS/MPS success retains its historical scope.'}
readme = next(p for p in r['pages'] if p['source'] == 'README.md')
readme['claims'].append(claim)
for key in ['factual_accuracy', 'source_verification', 'limitations']:
    readme['checks'][key]['claim_ids'].append('r15')
for p in r['pages']:
    p['checks']['source_verification']['prior_completed_details'] = p['checks']['source_verification']['details']
    p['checks']['source_verification']['details'] = 'The original completed evidence review is retained for the unchanged authored page. This source-dependency callback personally reverified changed ci/pages/evaluate sources; exact bounded deltas and official mechanism are recorded separately. All current source/artifact hashes match. No new native Windows pass or new model evaluation is asserted.'
    for check in p['checks'].values():
        check['details'] = check['details'].replace('Reinspection: current page and all initial evidence dependency hashes are unchanged;', 'At the previous completed page reinspection, the page and all initial evidence dependency hashes were unchanged;')
for p in r['pages']:
    for c in p['claims']:
        if any(e['source_id'] in ['pages', 'evaluate'] for e in c['evidence']):
            c['evidence'].append({'source_id': 'ci_dependency_measurements',
                'locator': '/additional_changed_sources',
                'supports': 'Current source delta rechecked: Pages adds the v2 LFS package while retaining all build/deploy gates; evaluator changes only path-key rendering and fsync open mode, with thresholds/scoring/summarization/protocol checks identical.'})
            c['artifact_ids'].append('ci_dependency_execution')
r.setdefault('source_dependency_callbacks', []).append({
    'reviewer': '/root/p6_fact_public_a', 'checked_at': now, 'kind': 'source_dependency_only',
    'prior_report_path': str(PRE.relative_to(ROOT)), 'prior_report_sha256': sha(PRE),
    'current_formal_report_locator': str(REPORT.relative_to(ROOT)), 'changed_source_ids': ['ci', 'pages', 'evaluate'],
    'previous_source_versions': {sid: next(x for x in before['sources'] if x['id'] == sid) for sid in new_sources},
    'current_source_versions': new_sources, 'authored_public_pages_unchanged': True,
    'public_claims_personally_reread': m['public_claims_reread'], 'all_other_evidence_dependency_hashes_match': True,
    'other_unchanged_dependency_count': len(m['unchanged_prior_dependencies']),
    'execution_artifact_id': 'ci_dependency_execution', 'scope': m['verified_scope'],
    'native_windows_run_checked': False, 'native_windows_pass_claimed': False,
    'original_issue_judgments_and_model_results_unchanged': True})
r['checks']['source_verification']['details'] = 'All current source/artifact hashes verified. Original reviewer rechecked ci, Pages LFS and evaluator portability deltas against preserved bytes; every authored public page and other dependency is unchanged. Historical anonymous readback and new native Windows success are not established by this pass.'
r['checks']['factual_accuracy']['details'] += ' Current CI contract and original official dispatch-control rationale personally checked; no current native Windows pass is asserted.'
r['independence']['source_dependency_callback'] = 'Only own preserved formal report, current cited public excerpts, current implementation deltas and original official issue/docs/code were read; no other reviewer conclusion was read.'
r['execution_boundary']['source_dependency_callback_performed'] = 'Read-only hashes, exact source deltas and AST comparison; no model execution, native Windows run, tests, training, generation, results edits or remote workflow trigger.'
assert r['source_files'] == before['source_files']
assert r['initial_review'] == before['initial_review'] and r['issues'] == before['issues']
assert r['verdict'] == 'pass'
REPORT.write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'report': str(REPORT.relative_to(ROOT)), 'sha256': sha(REPORT), 'verdict': r['verdict'],
    'changed_sources': {sid: s['sha256'] for sid, s in new_sources.items()},
    'claims': sum(len(p['claims']) for p in r['pages']), 'sources': len(r['sources']),
    'artifacts': len(r['artifacts']), 'native_windows_pass_claimed': False}, ensure_ascii=False, indent=2))
