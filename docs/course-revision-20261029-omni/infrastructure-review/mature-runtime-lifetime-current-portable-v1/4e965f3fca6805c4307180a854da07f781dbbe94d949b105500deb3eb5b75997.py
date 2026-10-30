"""新MR08 current與audit-v3；只延續source approval，唯一舊permit不改不重簽。"""

import hashlib,json,subprocess,sys
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path('/workspace/work/omni-models-v1');sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
from scripts.omni_revision import research_execution as runner,research_recovery as recovery

REPORT=ROOT/'docs/course-revision-20261029-omni/infrastructure/mature-research-runtime-review-current-mr08-v1.json'
AUDIT=ROOT/'docs/course-revision-20261029-omni/infrastructure/mature-research-audit-v3.json'
INDEX=Path('/workspace/work/omni-mature-runtime-lifetime-portable-evidence-current-v1.json')
def read(p):return json.loads(Path(p).read_bytes())
def binding(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def save(p,d):
 with Path(p).open('x') as f:f.write(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

assert not REPORT.exists() and not AUDIT.exists() and not INDEX.exists()
snap=read(HERE/'source-snapshot-manifest.json');files={x['path']:x['sha256'] for x in snap['files']}
assert len(files)==96 and set(files)==set(runner.source_paths())
for row in snap['files']:assert binding(row['original_path'])['sha256']==binding(row['snapshot_path'])['sha256']==row['sha256']
assert '92 passed' in (HERE/'targeted-pytest.log').read_text()
assert '4 passed' in (HERE/'clean-sdk-pytest.log').read_text() and '4 passed' in (HERE/'controlled-gc-sdk-pytest.log').read_text()
life=read(HERE/'lifetime-probe-results.json');prod=read(HERE/'gc-production-probe-results.json');guard=read(HERE/'source-admission-results.json');fixed=read(HERE/'unchanged-scope-permit-results.json')
assert all(x['all_assertions_passed'] for x in (life,prod,guard,fixed))
assert len(guard['private_individual_source_hash_drift_rejected'])==96 and prod['gc_stress_calls']==10
permit=ROOT/recovery.PERMIT_NAME;permit_ref=binding(permit);assert permit_ref['sha256']=='929b40a464f2cedea3021e33db3fcdabc6570552aa361dfc969ce17fd0be2fd8'
assert fixed['permit_unconsumed'] and fixed['retained_total_usd']=='6.60'
old_digests={
 'mature-research-runtime-review-original-mr08-v1.json':'1c178bf2d7b751638a62db31ab68d1ba6c9bddd4b15ffcd24c62a674262025a2',
 'mature-research-runtime-review-current-v1.json':'85f25f947f453786a81f864e286a3dd8672cfad0ddf721b16ee283de9c4f7405',
 'mature-research-audit-v2.json':'da64a8272155e03e01d1b4921669205b1c9773880c291159de5c3007cdfa4d27',
 'mature-research-runtime-review-original-v1.json':'d947cf06c95c28c60f34ab02f70a0bb4a3ec6f5d3348e6af2b08945e68082d40',
 'mature-research-runtime-review-original-mr07-v1.json':'b8a77303c36951d71b5689e551b7147ee1323b42a5fc0c7615b86333d1ccc205',
}
old=[]
for name,sha in old_digests.items():
 b=binding(ROOT/'docs/course-revision-20261029-omni/infrastructure'/name);assert b['sha256']==sha;old.append(b)
now=datetime.now(UTC).isoformat()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();assert head==snap['head']
save(HERE/'final-state-recheck.json',{'created_at':now,'reviewer_task':'/root/omni_development_execution_review','all96_current_match_frozen_snapshot':True,'changed_only':snap['changed_from_v2'],'old_records_unchanged':old,'unchanged_single_permit':permit_ref,'permit_unused':True,'ledger_blob':fixed['ledger_blob'],'retained_total_usd':'6.60','actual_head':head,'actual_corrected_source_still_uncommitted':True,'real_paid_calls':0,'real_ledger_or_release_writes':0})
names=['source-snapshot-manifest.json','targeted-pytest.log','clean-sdk-pytest.log','controlled-gc-sdk-pytest.log','lifetime_gc_plugin.py','exact-original-pytest-gc-trace.json','probe_lifetime_controls.py','lifetime-probe-results.json','lifetime-controls.log','probe_gc_stressed_production.py','gc-production-probe-results.json','gc-production.log','probe_current_source_admission.py','source-admission-results.json','source-admission.log','source-admission-fixture-first-attempt.log','probe_unchanged_scope_and_permit.py','unchanged-scope-permit-results.json','unchanged-scope-permit.log','final-state-recheck.json','seal_current_mr08.py']
portable=[binding(HERE/name) for name in names]
save(INDEX,{'schema':'omni-mature-runtime-lifetime-current-portable-evidence-v1','created_at':now,'reviewer_task':'/root/omni_development_execution_review','report_path':str(REPORT),'files':portable,'file_count':len(portable),'total_bytes':sum(x['bytes'] for x in portable),'excluded':['full96 source snapshot and3MB data','SDK dependency fullsource snapshots','raw provider ledger/owned release responses','private payload/result/commit-admission fixture files','all old original/current/index/permit bytes unchanged'],'location_map_instruction':'移轉小型自製證據時保留原path/SHA並建location-map；fullsources/3MBdata/第三方全文/大型ZIP不進普通Git。'})
extension={
 'schema':'same-owner-source-approval-extension-v1','created_at':now,'reviewer_task':'/root/omni_development_execution_review',
 'unchanged_sole_operative_permit':permit_ref,'new_permit_issued':False,'maximum_new_attempts_remains':1,'new_bound_usd_remains':'4.30','permit_consumed':False,
 'historical_source_approval':{'path':'docs/course-revision-20261029-omni/infrastructure/mature-research-audit-v2.json','sha256':'da64a8272155e03e01d1b4921669205b1c9773880c291159de5c3007cdfa4d27'},
 'effective_current_source_approval':{'path':'docs/course-revision-20261029-omni/infrastructure/mature-research-audit-v3.json','source_paths_count':96,'exact_files_field':'files in this audit'},
 'interpretation':'本人是929b原permit與d947原判同一reviewer。本新決定明確以本audit-v3接續requires_source_audit的來源核准，僅限本次兩檔差異；permit內舊95 map與v2路徑保留作歷史核准，不改寫為新source已審。新的完整96 map及commit/fresh守門由v3承擔。',
 'approved_delta_only':{'tests/test_omni_mature_modal_sdk.py':'strong sandbox/fs across await、GC before exercise、owner afterward；保留inverse、write/read、TimeoutError及cancel斷言','scripts/omni_revision/research_execution.py':'AUDIT常數只由v2改v3檔名'},
 'unchanged_authority':'同一原失敗row/receipt/input/job/worker62、同一identity/oldbound與一筆新4.30；不是額外attempt/token，不重置或消費permit。',
 'required_before_paid':'仍須完整96＋本v3 audit commit/main exact、manual gate、真SDK前置、fresh官方rates/billing/inventory、strict core、原ownedinput重用與CAS；無自動重試。',
}
limitations=['乾淨本機Python3.13.5/Modal1.6.0正控制與GC控制均過；實際Actions3.13.17成功仍待下一次exactmerged source，沒有冒稱已驗。','GC-controlled pytest只用原external trace安排GC，沒有替換SDK方法、吞ReferenceError、改timeout或刪case。','source96 positive與逐檔drift矩陣是明示private git fixture；真current來源未commit，實際paid admission須仍拒。','private source-admission fixture首拷沿用舊ee SHA而被正確拒，已保留first-attempt log，僅改取本次snapshot HEAD36a。','exec/RPC及worker result為private fixtures，未進行NN推論、ModalRPC、teacher下載或新reservation。','本次只核最小lifetime修正與source approval延續；不重評實際成熟結果、能力、自家final或教材三審。']
proofs={name:binding(HERE/name) for name in ('lifetime-probe-results.json','gc-production-probe-results.json','source-admission-results.json','unchanged-scope-permit-results.json','final-state-recheck.json')}
scope={'source_paths_count':96,'worker_files_count':62,'same_source_key_set_as_v2':True,'changed_files':snap['changed_from_v2'],'unchanged_science_sources':{'evaluation':57,'training':25,'implementation':8,'worker':62},'maximum_new_paid_attempts_under_same_permit':1,'new_bound_usd':'4.30','retained_round_bounds_usd':'6.60','single_permit_sha256':permit_ref['sha256'],'permit_unused':True,'reviewer_paid_calls':0,'reviewer_dispatches':0,'reviewer_ledger_or_release_writes':0,'reviewer_nn_predictions':0}
save(REPORT,{'schema':'omni-mature-runtime-mr08-current-review-v1','created_at':now,'author_task':'/root','reviewer_task':'/root/omni_development_execution_review','status':'pass_for_reviewed_scope','verdict':'MR08及inverse owner建議收束：原GC負例仍真實，修正版一般/GC-control均4P，必要timeout/取消與production維持；只接續同一未用permit的最小來源核准。','necessary_findings':[],'optional_findings':[],'resolved':[{'id':'MR08','evidence':'strong owner survives GC, write/read TimeoutError＋cancel preserved; exact original GC trace now4P'},{'id':'O-lifetime-01','evidence':'inverse nowretainsstrongowner beforeGC andafterawait; deliberately reversedarguments仍TypeError且exec0'}],'scope':scope,'files':files,'source_snapshot':binding(HERE/'source-snapshot-manifest.json'),'author_freeze_sha256':snap['author_freeze_sha256'],'preserved_prior_judgments':old,'source_approval_extension':extension,'independent_tests':{'targeted_passed':92,'clean_sdk_passed':4,'gc_controlled_sdk_passed':4,'lifetime_cases':5,'production_gc_points':10,'source_drift_cases':96,'actual_checked_permit_and_live_original_asset_eligibility':True},'evidence':proofs,'portable_evidence':binding(INDEX),'limitations':limitations})
save(AUDIT,{'schema':'omni-mature-research-audit-v3','created_at_utc':datetime.now(UTC).isoformat(),'author_task':'/root','reviewer_task':'/root/omni_development_execution_review','status':'pass_for_reviewed_scope','verdict':'完整96 current來源通過；same-owner明確延續唯一未用929b permit的source approval，沒有新permit/新增attempt/新增預算。','necessary_findings':[],'optional_findings':[],'files':files,'source_paths_count':96,'worker_files_count':62,'scope':scope,'current_review':binding(REPORT),'source_snapshot':binding(HERE/'source-snapshot-manifest.json'),'source_approval_extension':extension,'previous_judgments':old,'independent_evidence':proofs,'portable_evidence':binding(INDEX),'limitations':limitations})
assert read(AUDIT)['files']=={name:binding(ROOT/name)['sha256'] for name in runner.source_paths()}
assert binding(permit)==permit_ref
print(json.dumps({'current_report':binding(REPORT),'audit_v3':binding(AUDIT),'unchanged_permit':permit_ref,'portable_index':binding(INDEX),'portable_files':len(portable),'portable_bytes':sum(x['bytes'] for x in portable),'files_exact':len(files),'necessary_findings':0},ensure_ascii=False))
