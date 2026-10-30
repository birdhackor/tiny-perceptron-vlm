"""必要問題全收束後，本 original owner 新建唯一 operative permit；沒有預約或派工。"""

import base64,copy,hashlib,json,sys
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path('/workspace/work/omni-models-v1');sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
from scripts.omni_revision import research_execution as runner,research_recovery as recovery
from scripts.omni_revision.finance import digest

def read(p):return json.loads(Path(p).read_bytes())
def binding(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def save_new(p,value):
 with Path(p).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

manifest=read(HERE/'source-snapshot-manifest.json');files={x['path']:x['sha256'] for x in manifest['files']};assert len(files)==95
for name,expected in files.items():assert binding(ROOT/name)['sha256']==expected
assert '92 passed' in (HERE/'independent-targeted-pytest.log').read_text()
sdk=read(HERE/'real-sdk-probe-results.json');finish=read(HERE/'current-finish-results.json');reused=read(HERE/'current-recovery-results.json');admission=read(HERE/'source-admission-results.json')
assert all(x['all_assertions_passed'] for x in (sdk,finish,reused,admission))
assert len(admission['private_individual_source_hash_drift_rejected'])==96 and reused['actual_readonly_integration']['nn_forward_attempts']==[]
old=ROOT/recovery.ORIGINAL_NAME;assert binding(old)['sha256']==recovery.ORIGINAL_SHA
mr07=ROOT/'docs/course-revision-20261029-omni/infrastructure/mature-research-runtime-review-original-mr07-v1.json';assert binding(mr07)['sha256']=='b8a77303c36951d71b5689e551b7147ee1323b42a5fc0c7615b86333d1ccc205'
assert not (ROOT/recovery.PERMIT_NAME).exists()
ledger=json.loads(base64.b64decode(read(HERE/'provider-ledger.json')['content']));row=next(x for x in ledger['attempts'] if x['attempt_id']==recovery.IDENTITY['attempt'])
assert digest(row)==recovery.IDENTITY['row_sha256'] and digest(row['receipt'])==recovery.IDENTITY['receipt_sha256'] and ledger['reserved_total_usd']=='6.60'
now=datetime.now(UTC).isoformat()
permit={
 'schema':'omni-mature-preparation-recovery-permit-v1','created_at':now,'created_by':'/root/omni_development_execution_review','author_task':'/root','reviewer_task':'/root/omni_development_execution_review','status':'pass_for_exact_preparation_recovery',
 'identity':copy.deepcopy(recovery.IDENTITY),'original_runtime_review':{'path':recovery.ORIGINAL_NAME,'sha256':recovery.ORIGINAL_SHA},
 'additional_mr07_original':{'path':str(mr07.relative_to(ROOT)),'sha256':binding(mr07)['sha256']},
 'maximum_new_attempts':1,'new_bound_usd':'4.30',
 'source_path_proof':{'sdk_version':'1.6.0','first_write_failed_before_any_fs_rpc':True,'sha_marker_not_sent_by_failed_path':True,'provider_historical_NN_telemetry_directly_observed':False,'basis':'Original source92 and exact paid artifact match; real Modal1.6.0 raw internal filesystem deterministically rejects inverted first write before exec/RPC. This is a source-path proof, not direct provider zero-NN telemetry.'},
 'scope':'只允許既有USD60 round內，對精確失敗mature-kev-41004743884-1的一次明示controller準備修復新預約。保留原4.30及原失敗全部lineage；原input/job/worker62/科學條件不變。一般once-only、防重及Qwen原一次規則仍在。',
 'reviewed_existing95_files_sha256':files,
 'requires_source_audit':{'path':'docs/course-revision-20261029-omni/infrastructure/mature-research-audit-v2.json','status':'pass_for_reviewed_scope','exact_source_paths_count':96,'all_sources_and_audit_must_be_committed_at_actual_main_revision':True},
 'required_before_any_new_paid_call':[
  '必須是manual main且完整96來源和新audit逐bytes等於本次checkout；本permit不觸發任何dispatch。',
  '原row/receipt SHA、failed-stopped、全額4.30與完整逐ID停止必須仍精確相同；任何既有repair或第二同job嘗試拒絕，失敗亦不得再次例外。',
  'live原owned draft/body/target/唯一input asset ID/大小/SHA仍匹配，真正重取同434670B ZIP；job73f751...、70input與現行worker62及frozen manifests全核。',
  'fresh strict core/selection/opening與原科學job_config一致；不重挑T/tau、不改模型、pin、資料、題數、GPU、精度、更新或生成上界。',
  'fresh official rates/billing/inventory與原whole-gross60＋全部future bounds守門；先CAS新4.30，不退原6.60任何bound。',
  '新唯一attempt/App/Sandbox/draft；原single create/no retry/deadline/export-before-stop/已知ID先停/未知unresolved與保存界都維持。',
  '新purpose與receipt明示recovery_of/permit SHA/original input/job/row SHA，finish須綁本次attempt/source/audit/config/opening及provider target。',
 ],
 'mr06_mr07_resolution_evidence':{'real_sdk':binding(HERE/'real-sdk-probe-results.json'),'real_sdk_deadlines':binding(HERE/'real-sdk-deadline-results.json'),'foreign_receipt_and_finish':binding(HERE/'current-finish-results.json'),'genuine_old_record_and_live_input_reuse':binding(HERE/'current-recovery-results.json'),'source_guard_fixture96':binding(HERE/'source-admission-results.json')},
 'limitations':['沒有直接觀察原provider歷史NN telemetry；不能以無輸出宣稱實測零NN。','本reviewer沒有新預約／ModalRPC／teacher weights下載／NN prediction／dispatch。','這不是能力、teacher研究結果、教材三審或自家final結果通過判定。'],
}
path=ROOT/recovery.PERMIT_NAME;save_new(path,permit);(HERE/'operative-permit-final-snapshot.json').write_bytes(path.read_bytes())
actual,sha=recovery.checked_permit();assert actual==permit
lineage=recovery.eligibility(ledger,'kev');assert lineage==recovery.lineage(sha)
assert recovery.eligibility(ledger,'qwen25') is None
assert set(runner.source_paths())==set(files)|{recovery.PERMIT_NAME} and len(runner.source_paths())==96
save_new(HERE/'operative-permit-validation-results.json',{'created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_development_execution_review','all_assertions_passed':True,'operative_permit':binding(path),'actual_checked_permit_positive':True,'actual_eligibility_positive_with_original_ledger_and_live_owned_asset_get':True,'lineage':lineage,'qwen_no_prior_attempt_normal_path':True,'live_ledger_writes':0,'release_writes':0,'modal_rpc_calls':0,'dispatches':0,'audit_or_current_commit_admission_not_claimed':True})
print(json.dumps({'operative_permit':binding(path),'source_paths_count':96,'actual_checked_permit_and_eligibility_passed':True,'dispatches':0},ensure_ascii=False))
