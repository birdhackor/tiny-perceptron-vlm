"""Update the original review with verified corrections and preserved history."""
from pathlib import Path
import copy, datetime, hashlib, json

ROOT = Path.cwd()
BASE = ROOT / 'docs/technical-reviews/artifacts/p6-public-a'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    return json.loads(Path(p).read_bytes())
initial_path = BASE / 'history/initial-report.json'
assert sha(initial_path) == '1cf5f7fef1587d4e9c20126d60f565ea3c5f1307884f1450a70ad212ac206f1e'
initial = load(initial_path)
r = copy.deepcopy(initial)
m = load(BASE / 'reinspection-measurements.json')
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
changed = set(m['changed_pages'])
for p in r['pages']:
    if p['source'] in changed:
        observed = m['changed_pages'][p['source']]
        p['initial_source_sha256'] = p['source_sha256']
        p['source_sha256'] = observed['current_sha256']
        p['frozen_input_path'] = observed['frozen_path']
        p['frozen_input_sha256'] = observed['current_sha256']
        p['read_scope'] = 'Original reviewer personally reread the full revised UTF-8 Markdown, including every heading, table, paragraph, link and fence. This page has no directly referenced figure.'
        p['read_event'] = {'kind': 'full_current_page_reinspection', 'reviewer': '/root/p6_fact_public_a', 'checked_at': now}
    else:
        p['initial_read_scope'] = p['read_scope']
        p['read_scope'] = 'Initial full-page read and original evidence review retained. The current source SHA was rechecked and is identical; this reinspection does not claim another full-page read or rerun of all initial checks.'
        p['read_event'] = {'kind': 'unchanged_sha_carry_forward', 'reviewer': '/root/p6_fact_public_a', 'checked_at': now}
by_claim = {c['id']: c for p in r['pages'] for c in p['claims']}
original_affected_claims = {cid: copy.deepcopy(by_claim[cid]) for cid in ['s4', 'a8', 'a2']}
for cid in ['s4', 'a8']:
    c = by_claim[cid]
    c['status'] = 'verified'
    c['statement'] = ('v2共8,962個檔案，含12個JSONL與8,950個其他檔案；其他檔案包含圖片、錄音與來源說明。'
        if cid == 's4' else 'v2壓縮大小67,862,431 bytes、解包內容127,161,811 bytes；共8,962個檔案，含12個JSONL與8,950個其他檔案（圖片、錄音與來源說明）。')
    c['scope'] = 'The immutable tar contains 8,335 PNG files, 603 WAV files and 12 source/legal/provenance support files: 8,950 non-JSONL files. These are separate from the 12 JSONL files. This establishes file classification, not independent sample count or model ability.'
    c['evidence'].extend([
        {'source_id': 'v2_original_package', 'locator': 'All 8,962 regular tar members; original SHA-bound payload', 'supports': 'Original package entity inspected again; every member byte count and SHA matches the original manifest.'},
        {'source_id': 'reinspection_measurements', 'locator': '/v2_original_archive_reinspection/{extension_counts,jsonl_files,media_files,support_files,other_files}', 'supports': 'Own recomputation: 12 JSONL, 8,938 media and 12 support files; 8,950 other files is accurate.'}])
    c['artifact_ids'].append('reinspection_execution')
    c['measurement_pointers'] = ['/v2_original_archive_reinspection/file_count', '/v2_original_archive_reinspection/jsonl_files', '/v2_original_archive_reinspection/other_files', '/v2_original_archive_reinspection/support_files']
c = by_claim['a2']
c.update(kind='scope', status='verified',
    statement='修訂頁已刪除2026-10-02歷史發布狀態段；目前沒有宣稱當日已完成冷快取匿名逐包逐檔回讀。',
    location='首批資料介紹段與全頁修訂範圍',
    evidence=[
        {'source_id': 'revised_training_assets_page', 'locator': 'Complete current Markdown', 'supports': 'The historical publication/readback paragraph is absent after a personal full-page reread.'},
        {'source_id': 'reinspection_measurements', 'locator': '/changed_pages/assets~1training~1README.md', 'supports': 'Current frozen page and SHA checked; old paragraph, run URL and anonymous-readback phrase are absent.'}],
    artifact_ids=['reinspection_execution', 'reinspection_training_diff'],
    scope='This verifies removal from the current teaching page only. The initial finding that the historical anonymous cold-cache readback lacked a checked original receipt remains preserved. No new receipt was obtained and the historical action is not marked verified.')
for p in r['pages']:
    p['verdict'] = 'pass'
    if p['source'] == 'docs/asset-storage.md':
        p['reviewer_summary'] = '完整重讀修訂頁；8,950已準確分類為其他檔案，原包逐檔核對再次一致。其他技術、操作與限制敘述仍在初次原證據支持範圍內。'
    elif p['source'] == 'assets/training/README.md':
        p['reviewer_summary'] = '完整重讀修訂頁；其他檔案分類正確，歷史匿名冷快取發布段已刪除。保留原授權、格式、轉換入口與訓練資料范围的親核證據。'
    for key, check in p['checks'].items():
        check['status'] = 'not_applicable' if key == 'figure_consistency' and not p['figure_sha256'] else 'pass'
        if p['source'] in changed:
            check['details'] = {
                'factual_accuracy': 'Full revised page reread; corrected classification verified against original immutable tar and manifest. Historical readback paragraph removed; its original unresolved judgment remains in history.',
                'numeric_verification': 'Archive bytes, all member sizes/hashes, media/support/JSONL counts and record split counts recomputed. Other unchanged initial numeric evidence remains SHA-identical.',
                'figure_consistency': 'This revised page has no directly referenced figure.',
                'source_verification': 'Current wording, original archive and manifest personally checked; every initial source/artifact dependency hash remains valid. Historical receipt is not newly verified.',
                'limitations': 'File counts remain distinct from independent media/samples and model quality. Method, software contract, finished original experiments and future publication instructions remain distinct.'}[key]
        else:
            check['details'] += ' Reinspection: current page and all initial evidence dependency hashes are unchanged; the original completed scope is retained without claiming a rerun.'
def source(sid, path, kind, note, version):
    r['sources'].append({'id': sid, 'kind': kind, 'title': path, 'path': path,
        'sha256': sha(ROOT / path), 'version': version, 'verified': True,
        'checked_original': True, 'inspection_note': note})
source('v2_original_package', 'assets/training/selftrained-v2.tar.gz', 'original_record',
    'Existing immutable tar entity personally rechecked: archive SHA, all 8,962 member byte counts and hashes, 8,938 media files, 12 JSONL and 12 source/legal/provenance support files.',
    'Manifest fixed Git revision 08761dac87a6ef360db95883d9bcd338c44fe76d; SHA 0976073a3bc7c331a65cedb4c31d54f5c6e9a5014ff2443698a8e2b8cad7e78a')
source('reinspection_measurements', str((BASE / 'reinspection-measurements.json').relative_to(ROOT)), 'execution',
    'Original reviewer reran bounded read-only archive/count/hash checks and all initial evidence dependency hashes; no training, generation or new grading.', now)
for sid, path in [('revised_asset_storage_page', 'docs/asset-storage.md'), ('revised_training_assets_page', 'assets/training/README.md')]:
    source(sid, path, 'reviewed_page', 'Personally read the entire current page. Used only to identify its actual wording and removal; technical dataset facts are established by original archive/manifests and original implementation evidence.', now)
def artifact(aid, path, kind, description, command=None, result=None):
    a = {'id': aid, 'path': path, 'sha256': sha(ROOT / path), 'kind': kind, 'description': description}
    if command:
        a.update(command=command, result=result, environment={'python': '3.13.5', 'device': 'cpu', 'gpu_used': False})
    r['artifacts'].append(a)
artifact('reinspection_execution', str((BASE / 'reinspection-measurements.json').relative_to(ROOT)), 'execution',
    'Current page/frozen input checks, unchanged-page scope checks, all evidence dependency hashes, original tar classification and all existing package entity hashes.',
    m['executed_command'], 'All assertions passed: 8,962 tar members exact; 12 JSONL + 8,938 media + 12 support = 12 JSONL + 8,950 other files; both revised page statements accurate; historical readback paragraph removed; five pages and 177 initial dependencies unchanged.')
artifact('reinspection_script', str((BASE / 'verify_reinspection.py').relative_to(ROOT)), 'execution_code', 'Exact bounded read-only reinspection script.')
artifact('reinspection_stdout', str((BASE / 'reinspection.stdout.txt').relative_to(ROOT)), 'execution_stdout', 'Actual successful reinspection output.')
artifact('reinspection_asset_storage_diff', str((BASE / 'reinspection/docs/asset-storage.md.diff').relative_to(ROOT)), 'source_delta', 'Own frozen initial/current page delta; not an author change log.')
artifact('reinspection_training_diff', str((BASE / 'reinspection/assets/training/README.md.diff').relative_to(ROOT)), 'source_delta', 'Own frozen initial/current page delta showing the exact count correction and removed historical paragraph.')
artifact('initial_report_backup', str(initial_path.relative_to(ROOT)), 'review_history', 'Opaque byte-identical complete original formal report; initial verdict, page hashes, claims and two issue judgments are retained.')
for issue in r['issues']:
    issue['initial_status'] = issue['status']
    issue['status'] = 'resolved'
    issue['initial_judgment_preserved'] = True
    issue['resolved_by'] = '/root/p6_fact_public_a'
    issue['resolved_at'] = now
    issue['resolution_artifact_ids'] = ['reinspection_execution', 'initial_report_backup']
    if issue['id'] == 'i1':
        issue['resolution'] = 'Both current complete pages say 8,950 other files, including images, recordings and source explanations. Own original tar/manifest/member checks again establish 8,938 media plus 12 support files. Initial contradicted claims are preserved in the initial report and reinspection history.'
    else:
        issue['resolution'] = 'The historical publication/readback paragraph was removed. This resolves the current page issue by removing the unsupported historical claim; the historical anonymous cold-cache readback remains unverified and no new receipt is claimed.'
        issue['historical_action_verified'] = False
r['source_files'] = {p['source']: p['source_sha256'] for p in r['pages']}
r['verdict'] = 'pass'
r['reviewed_at'] = now
r['review_round'] = 'phase6-original-public-a-reinspection'
r['initial_review'] = {'report_path': str(initial_path.relative_to(ROOT)), 'report_sha256': sha(initial_path),
    'verdict': initial['verdict'], 'source_files': initial['source_files'], 'issues': initial['issues'],
    'affected_claims': original_affected_claims, 'initial_validation_path': str((BASE / 'initial-report-validation.json').relative_to(ROOT)),
    'initial_validation_sha256': sha(BASE / 'initial-report-validation.json')}
r['reinspection'] = {'status': 'completed', 'reviewer': '/root/p6_fact_public_a', 'checked_at': now,
    'full_current_pages_read': list(m['changed_pages']), 'unchanged_sha_carry_forward_pages': list(m['unchanged_pages']),
    'scope': 'The original reviewer fully reread the two revised pages, personally rechecked corrected statements against original immutable tar/manifest/files, verified the removed historical paragraph and checked all existing evidence dependencies. Five other full-page initial reviews and both prior actual figure views are retained after matching current SHA. No complete rerun of initial commands, full training, model generation, GPU or new answer grading is claimed.',
    'initial_report_preserved_opaque': True, 'original_issue_judgments_preserved': True,
    'all_initial_dependency_hashes_match': True, 'dependency_count': len(m['prior_evidence_dependency_hashes']),
    'changed_pages': m['changed_pages'], 'unchanged_pages': m['unchanged_pages'],
    'historical_anonymous_readback_verified': False,
    'executed_command': m['executed_command'], 'execution_artifact_id': 'reinspection_execution'}
r['independence']['reinspection_procedure'] = 'Original reviewer only: full two-page current read, own opaque initial report, original tar/manifests/files and matching retained evidence. No other reader or technical report conclusion was read.'
r['execution_boundary']['reinspection_performed'] = 'Read-only file/entity hashes and immutable tar member classification/count rechecks; no model execution.'
r['checks'] = {
    'factual_accuracy': {'status': 'pass', 'details': 'Current file classification is accurate. Historical readback paragraph removed; the original unresolved historical judgment remains preserved.'},
    'numeric_verification': {'status': 'pass', 'details': 'Original v2 package/member counts, sizes, hashes and splits recomputed; all existing evidence hashes match. Other original numeric checks are retained without claiming reruns.'},
    'figure_consistency': {'status': 'pass', 'details': 'Both current SVG hashes match their initially rendered and personally viewed 640/360 versions. No new figure rendering is claimed.'},
    'source_verification': {'status': 'pass', 'details': 'Original package, manifest and changed statements checked by the same reviewer. All 177 initial source/artifact dependencies are unchanged. Historical anonymous readback is not established by this pass.'},
    'limitations': {'status': 'pass', 'details': 'Current teaching claims distinguish files, records, independent media, model ability, software contracts, original experiments and future procedures. Reinspection scope and original judgments are explicit.'}}
(ROOT / 'docs/technical-reviews/public-pages-p6-a.json').write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'verdict': r['verdict'], 'source_files': r['source_files'],
    'initial_report_sha256': sha(initial_path), 'current_report_sha256': sha(ROOT / 'docs/technical-reviews/public-pages-p6-a.json'),
    'claims': len(by_claim), 'sources': len(r['sources']), 'artifacts': len(r['artifacts']),
    'resolved_issues': [x['id'] for x in r['issues']]}, ensure_ascii=False, indent=2))
