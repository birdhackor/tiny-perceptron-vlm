"""Same-owner bounded current: hash identities only; no NN or network work."""
import copy
import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/workspace/work/omni-book-integrated-v1')
BASE = Path('docs/technical-reviews/artifacts/c-v3-v21-numbered-r4')
ORIGINAL = Path('docs/course-revision-20261101-c-v3/technical-v21/numbered_r4/original-v1.json')
ORIGINAL_SHA = '3ee1c43932862165c1b2f58a48d10e22eeaa5631628eab379f53e130aa638092'
INTAKE = Path('docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/numbered-r4-original-v1.json')
INTAKE_SHA = '634ec62d89fd2a9d54d3258fd52aa00c471882273c66af2ba7ac526f454f8c51'
CURRENT = ORIGINAL.with_name('current-v1.json')
COMMAND = 'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-numbered-r4/save-current-v1.py'


def sha_file(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


assert not (ROOT/CURRENT).exists(), 'Never overwrite a saved current judgment'
assert sha_file(ROOT/ORIGINAL) == ORIGINAL_SHA
assert sha_file(ROOT/INTAKE) == INTAKE_SHA
original = json.loads((ROOT/ORIGINAL).read_text())
intake = json.loads((ROOT/INTAKE).read_text())
assert intake['report'] == {'path':str(ORIGINAL),'sha256':ORIGINAL_SHA}
assert intake['native_validation'] == 'check_original'
assert intake['reviewer_task'] == original['reviewer_task'] == '/root/c_v3_technical_coord/numbered_r4'

source_bytes = (ROOT/'course/README.md').read_bytes()
start = source_bytes.index('## R.4 '.encode())
end = source_bytes.find(b'\n## ', start + 4)
end = len(source_bytes) if end < 0 else end + 1
live_source_sha = hashlib.sha256(source_bytes[start:end]).hexdigest()
assert live_source_sha == original['source_sha256']
freeze = ROOT/'docs/course-revision-20261101-c-v3/numbered-source-freeze-v21/sections/R.4.md'
assert sha_file(freeze) == live_source_sha
identities = []
for source in original['sources']:
    if source['kind'] != 'repository_code':
        continue
    observed = sha_file(ROOT/source['path'])
    assert observed == source['sha256'], source['path']
    identities.append({'path':source['path'],'expected_sha256':source['sha256'],'observed_sha256':observed})
artifacts = []
for item in original['artifacts']:
    observed = sha_file(ROOT/item['path'])
    assert observed == item['sha256'], item['path']
    artifacts.append({'path':item['path'],'expected_sha256':item['sha256'],'observed_sha256':observed})
authority = json.loads((ROOT/BASE/'authority.json').read_text())
installed_originals = []
for source in authority['sources']:
    if 'original_path' not in source:
        continue
    observed = sha_file(Path(source['original_path']))
    assert observed == source['original_sha256']
    installed_originals.append({'title':source['title'],'observed_sha256':observed})
release = json.loads((ROOT/'docs/decision-product/v3/public-release.json').read_text())
zip_path = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')
zip_sha = sha_file(zip_path)
assert zip_sha == release['sha256']
assert zip_path.stat().st_size == release['bytes']
old_run = json.loads((ROOT/BASE/'execution.json').read_text())
assert old_run['result'] == 'All bounded assertions passed'
assert len(old_run['fresh_cpu_examples']) == 3
assert old_run['c_v3_retained_measurement']['per_kind_correct_total'] == {'choice':[1500,1500],'info':[1500,1500],'rubric':[1500,1500]}
old_transport = json.loads((ROOT/BASE/'public-transport.json').read_text())
assert old_transport['http_status'] == 404 and old_transport['verified'] is False
assert original['verdict'] == 'revise' and original['issues'][0]['claim_id'] == 'c16'

environment = {'python':platform.python_version(),'torch':importlib.metadata.version('torch'),'device':'CPU identity checks only; no tensors or inference','platform':platform.platform()}
receipt = {
    'created_at':datetime.now(UTC).isoformat(),
    'reviewer_task':original['reviewer_task'],
    'command':COMMAND,
    'environment':environment,
    'scope':'Only current source/artifact/original-intake/local-ZIP identities checked. No training, NN inference, final evaluation, old research rerun, or network retry.',
    'original_report':{'path':str(ORIGINAL),'sha256':ORIGINAL_SHA},
    'original_intake':{'path':str(INTAKE),'sha256':INTAKE_SHA,'received_at':intake['received_at']},
    'live_source_sha256':live_source_sha,
    'repository_source_identities':identities,
    'retained_artifact_identities':artifacts,
    'installed_official_source_identities':installed_originals,
    'local_fixed_zip':{'path':str(zip_path),'bytes':zip_path.stat().st_size,'sha256':zip_sha,'comparison':'exactly matches the fixed descriptor and original execution receipt'},
    'prior_execution_identity':{'path':str(BASE/'execution.json'),'sha256':sha_file(ROOT/BASE/'execution.json'),'fresh_CPU_examples_already_executed':3,'new_CPU_examples_executed':0,'new_final_predictions':0},
    'transport_disposition':{'issue_pointer':'/issues/0','status':'unresolved','retained_observation_at':old_transport['created_at'],'retained_http_status':404,'network_retried':False,'anonymous_download_success_established':False,'resolution':'公開下載必要門檻仍未修復；本機ZIP一致與舊CPU實跑不能替代匿名公開取得。等待真正公開asset可驗後再由本人複查。'},
    'result':'All bounded identity checks passed; original public-transport issue remains unresolved; current verdict revise',
}
receipt_path=BASE/'bounded-identity-current-v1.json'
(ROOT/receipt_path).write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
report=copy.deepcopy(original)
report['created_at']=datetime.now(UTC).isoformat()
report['verdict']='revise'
report['original_report']={'path':str(ORIGINAL),'sha256':ORIGINAL_SHA}
report['original_intake_receipt']={'path':str(INTAKE),'sha256':INTAKE_SHA}
report['current_scope']='同owner在原判真收件之後，只核source21/live切片、33份repository原始來源、原先已跑證據與固定本機C-v3 ZIP的SHA/大小。沿用原判實讀/已跑的有界證據，未重跑舊研究、NN、final或網路；公開transport仍未解。'
report['independence_statement']='原判由本fresh reviewer獨立完成並已真parse/hash/native check_original收件。此current為同owner的有界後續判定，逐項確認原證據身分與範圍；未用其他review pass替代本人證據，未修改original。'
report['review_scope']='只審R.4實質主張及直接必要證據；current僅核固定本機包、來源與原已跑證據identity。無圖。未改教材、Notebook、17產品py、舊證據、DP或active；未訓練、重跑推論/final、重試網路、用Modal、發布或提交。'
report['original_issue_resolutions']=[{'json_pointer':'/issues/0','status':'unresolved','resolution':'仍未解，未宣稱修復。原匿名GET的HTTP404保留；本次僅完成固定本機ZIP/來源/舊執行證據身分核對，沒有重試網路，也沒有新的匿名成功取得證據。公開transport仍是必要發布門檻；等待真正公開asset可驗後，須由本人核對固定bytes/SHA並再次判定。'}]
report['issues'][0]['details']='原HTTP404造成的必要公開取得門檻仍未解。本current僅確認本機固定ZIP與原始證據未漂移；沒有重試網路，未宣稱當前匿名下載成功。'
report['issues'][0]['required_resolution']='等待真正公開的固定asset可匿名取得，再由本人核bytes/SHA；在此之前保持unresolved/revise，不能由本機包資格或舊三CPU例代替公開transport。'
report['sources'].append({'id':'s_current_identity','kind':'execution','title':'同owner current的本機與既有證據身分核對','verified':True,'artifact_id':'a_current_identity'})
report['artifacts'].append({'id':'a_current_identity','kind':'execution','path':str(receipt_path),'sha256':sha_file(ROOT/receipt_path),'description':'原判真收件後的bounded local current：核固定ZIP、來源、原始報告與既有execution證據身分；未重做NN/final/網路。','command':COMMAND,'result':receipt['result'],'environment':environment})
report['artifacts'].append({'id':'a_current_identity_code','kind':'code','path':str(BASE/'save-current-v1.py'),'sha256':sha_file(Path(__file__)),'description':'只做檔案串流hash與保存同owner current，無NN或網路操作。'})
report['checks']['factual_accuracy']['details']='原18項主張與已跑證據身分保持；本current只做本機identity核對。C3模型本機資格與三CPU例保留，但公開下載必要門檻沒有新的成功證據，仍維持revise。'
report['checks']['source_verification']['details']='全部repository來源、七項原artifact、官方wheel原始檔與source21/live切片hash均未漂移；固定本機ZIP大小/SHA吻合。遠端來源不重讀、HTTP404不重試，原公開取得未解不冒作成功。'
report['checks']['limitations']['details']+=' current僅核identity，不重新測量或推廣任何模型能力；原HTTP404未重試，公開資產待真驗。'
assert report['claims'][15]['id']=='c16' and report['claims'][15]['status']=='unresolved'
assert datetime.fromisoformat(report['created_at']) > datetime.fromisoformat(intake['received_at'])
assert sha_file(ROOT/ORIGINAL)==ORIGINAL_SHA
(ROOT/CURRENT).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(CURRENT),'sha256':sha_file(ROOT/CURRENT),'verdict':report['verdict'],'repository_sources_checked':len(identities),'artifacts_checked':len(artifacts),'scope':receipt['scope']},ensure_ascii=False))
