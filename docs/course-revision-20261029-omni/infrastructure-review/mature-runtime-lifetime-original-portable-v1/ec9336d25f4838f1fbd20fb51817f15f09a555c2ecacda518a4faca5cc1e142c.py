"""封存新的MR08 original；不改任何舊original/current/audit/permit或author來源。"""

import hashlib,json,re
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path('/workspace/work/omni-models-v1');HERE=Path(__file__).resolve().parent
REPORT=ROOT/'docs/course-revision-20261029-omni/infrastructure/mature-research-runtime-review-original-mr08-v1.json'
INDEX=Path('/workspace/work/omni-mature-runtime-lifetime-portable-evidence-original-v1.json')
def read(p):return json.loads(Path(p).read_bytes())
def binding(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def save(p,d):
 with Path(p).open('x') as f:f.write(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

assert not REPORT.exists() and not INDEX.exists()
snap=read(HERE/'source-snapshot-manifest.json');assert snap['source_count']==96
for row in snap['files']:assert binding(row['source_path'])['sha256']==row['sha256']==binding(row['snapshot_path'])['sha256']
life=read(HERE/'lifetime-probe-results.json');prod=read(HERE/'gc-production-probe-results.json');assert life['all_assertions_passed'] and prod['all_assertions_passed'] and prod['gc_stress_calls']==10
assert life['real_rpc_calls']==prod['real_rpc_calls']==0
clean=(HERE/'clean-py313-sdk-tests.log').read_text();assert '4 passed' in clean
gc_log=(HERE/'exact-original-pytest-controlled-gc.log').read_text();assert '2 failed, 2 passed' in gc_log and 'ReferenceError' in gc_log
trace=read(HERE/'exact-original-pytest-gc-trace.json');assert {x['operation'] for x in trace['events']}=={'read','write'}
jobs=read(HERE/'provider-jobs.json')['structuredContent']['jobs'];assert len(jobs)==1 and jobs[0]['id']==123912397170
steps={x['name']:x['conclusion'] for x in jobs[0]['steps']};assert steps['Verify the real Modal filesystem contract offline before reservation']=='failure'
for name in ('Prepare and reserve once, infer, export and stop the exact owned resources','Check all evidence size before artifact upload','Retain complete or partial original evidence for seven days','Verify provider artifact limits and finish the exact reservation without refund'):assert steps[name]=='skipped'
lr=read(HERE/'provider-ledger.json')['structuredContent'];ledger=json.loads(lr['content']);assert lr['sha']=='21c26a416535f9b863340ac1bf2ae2669d389b30' and ledger['reserved_total_usd']=='6.60' and not any('41009648132' in x['attempt_id'] for x in ledger['attempts'])
old={
 'mature-research-runtime-review-original-v1.json':'d947cf06c95c28c60f34ab02f70a0bb4a3ec6f5d3348e6af2b08945e68082d40',
 'mature-research-runtime-review-original-mr07-v1.json':'b8a77303c36951d71b5689e551b7147ee1323b42a5fc0c7615b86333d1ccc205',
 'mature-research-runtime-review-current-v1.json':'85f25f947f453786a81f864e286a3dd8672cfad0ddf721b16ee283de9c4f7405',
 'mature-research-audit-v2.json':'da64a8272155e03e01d1b4921669205b1c9773880c291159de5c3007cdfa4d27',
 'mature-research-preparation-recovery-permit-v1.json':'929b40a464f2cedea3021e33db3fcdabc6570552aa361dfc969ce17fd0be2fd8',
}
old_refs=[]
for name,sha in old.items():
 b=binding(ROOT/'docs/course-revision-20261029-omni/infrastructure'/name);assert b['sha256']==sha;old_refs.append(b)
closed=datetime.now(UTC).isoformat()
save(HERE/'controlled-gc-pytest-excerpt.json',{'created_at':closed,'raw_failure_log':binding(HERE/'exact-original-pytest-controlled-gc.log'),'short_excerpt':[s for s in gc_log.splitlines() if s.startswith('FAILED ') or s.startswith('E ') and 'ReferenceError' in s or '2 failed, 2 passed' in s],'gc_control_trace':trace,'original_test_source_unchanged':True})
save(HERE/'provider-precas-and-preservation-results.json',{'created_at':closed,'reviewer_task':'/root/omni_development_execution_review','all_assertions_passed':True,'job_id':jobs[0]['id'],'job_conclusion':jobs[0]['conclusion'],'step_conclusions':steps,'ledger_blob':lr['sha'],'retained_bounds_usd':ledger['reserved_total_usd'],'no_new_attempt_row':True,'permit_consumed':False,'original96_source_bytes_unchanged':True,'preserved_records':old_refs})
portable_names=['source-snapshot-manifest.json','native-error-excerpt.json','clean-py313-sdk-tests.log','clean-py31317-sdk-tests.log','python31317-availability.log','probe_real_sdk_lifetime.py','lifetime-probe-results.json','lifetime-probe.log','lifetime-probe-first-assertion.log','probe_gc_stressed_production.py','gc-production-probe-results.json','gc-production-probe.log','lifetime_gc_plugin.py','exact-original-pytest-gc-trace.json','controlled-gc-pytest-excerpt.json','provider-precas-and-preservation-results.json','seal_mr08_original.py']
portable=[binding(HERE/name) for name in portable_names]
save(INDEX,{'schema':'omni-mature-runtime-lifetime-original-portable-evidence-v1','created_at':closed,'reviewer_task':'/root/omni_development_execution_review','report_path':str(REPORT),'files':portable,'file_count':len(portable),'total_bytes':sum(x['bytes'] for x in portable),'excluded':['full96 source snapshots and3MB data','third-party SDK full source/dependency snapshots','full native log and raw provider responses','full controlled-pytest failure log; only short supporting excerpt plus raw SHA retained','private payload/result files'],'location_map_instruction':'保留原path/SHA，移轉小型自製證據時附location-map；不得把fullsnapshots/第三方全文/原native全文加普通Git。'})
save(REPORT,{
 'schema':'omni-mature-research-runtime-original-mr08-v1','created_at':closed,'author_task':'/root','reviewer_task':'/root/omni_development_execution_review','status':'changes_required',
 'verdict':'MR08成立：pre-CAS真SDK timeout test的filesystem失去強Sandbox owner，GC可在進exec前造成ReferenceError。原source96與所有舊audit/permit保留。',
 'scope':{'run_id':41009648132,'job_id':123912397170,'source_revision':snap['head'],'source_count':96,'science57_training25_implementation8_worker62_changed':False,'reviewer_modal_rpc_calls':0,'reviewer_ledger_or_release_writes':0,'reviewer_dispatches':0,'reviewer_nn_predictions':0},
 'source_snapshot':binding(HERE/'source-snapshot-manifest.json'),'files':{x['path']:x['sha256'] for x in snap['files']},'preserved_prior_judgments_and_single_permit':old_refs,
 'necessary_findings':[{
  'id':'MR08','severity':'blocking_prepaid_gate','location':'tests/test_omni_mature_modal_sdk.py:187',
  'summary':'fs=make().filesystem只留下filesystem；Modal1.6.0 filesystem對container是weakref.proxy，暫時Sandbox可被GC回收，timeout/cancellation測試因此尚未進fakeexec就ReferenceError。',
  'evidence':{'native_actual':'read param 1F3P / ReferenceError at SDK sandbox_fs.py307','clean_local_unmodified':'Python3.13.5 / Modal1.6.0 original4P；不冒稱無控制時亦重現CI1F3P。','exact_original_test_with_gc_scheduled_after_fs_assignment':'同一未改test與SDK，external trace只安排gc.collect，write/read均ReferenceError，2F2P。','real_sdk_lifetime_controls':'暫時owner經GC後write/read皆ReferenceError且exec0；強owner經GC後write/read皆TimeoutError並收到取消。','production_control':'真原run_staged_job在10次hard_wait前各強制GC，仍2write2read、單次模擬create、export-before-stop與逐ID停止通過。'},
  'required_fix':['timeout probe先保留sandbox=make()，再fs=sandbox.filesystem，持有到operation與取消斷言完成；保持write/read兩case、TimeoutError和取消斷言。','保留原真SDK signature/inverse/完整controller回歸；不要catch ReferenceError當成功、延長timeout、跳過read、關GC、fake掉filesystem或換SDK以掩蓋。','修正後本人用相同未改GC控制重跑original test file；需normal4P及GC壓力4P，並重新凍結完整96與新audit版本。'],
  'production_assessment':'run_staged_job有明確strong sandbox local，自_new_hydrated保留到finally結束；本次沒有已證production transport缺陷，也不應據此改staged/模型/worker或放寬paid界線。',
 }],
 'optional_findings':[{'id':'O-lifetime-01','location':'tests/test_omni_mature_modal_sdk.py:79','summary':'inverse case亦使用臨時make()，但參數驗證發生在weakproxy deref前，故owner已回收仍先TypeError；其pass不能證lifetime。','suggestion':'建議inverse也明留強sandbox，與production一致；保留故意反參數與exec0斷言。'}],
 'financial_and_execution_observations':{'fresh_github_job_steps':steps,'fresh_ledger_blob':lr['sha'],'retained_total_usd':'6.60','no_new_attempt_row_for_failed_preCAS_run':True,'no_new_paid_reservation_observed':True,'single_permit_unused':True,'original4.30_not_refunded':True,'no_research_or_finish_step_executed':True},
 'runtime':{'tested_clean_python':life['python'],'native_python_patch':'3.13.17','modal':life['modal_version'],'synchronicity':life['synchronicity_version'],'sdk_source':life['sdk_filesystem_source'],'exact_native_patch_replay_available':False,'python31317_install_attempt':'local uv has no interpreter/download definition for3.13.17; availability logs preserved'},
 'controlled_cases':life['cases'],'independent_evidence':{'lifetime':binding(HERE/'lifetime-probe-results.json'),'exact_original_pytest_controlled_gc':binding(HERE/'controlled-gc-pytest-excerpt.json'),'production_gc':binding(HERE/'gc-production-probe-results.json'),'provider_preCAS':binding(HERE/'provider-precas-and-preservation-results.json'),'native_excerpt':binding(HERE/'native-error-excerpt.json')},
 'future_source_review_requirement':'本pre-CAS失敗未消耗唯一paid repair permit，保留929b原bytes。修正source後仍須同owner新current/audit明確接續source approval；不能把舊audit-v2或permit內歷史95source map說成新source已審。新audit需明示只承接同一未用permit與既定一筆4.30，不能新增第二permit或增加次數／科學範圍。',
 'portable_evidence':binding(INDEX),
 'limitations':['原native41807B/SHA50e2...已自核；root最初41805是JS codeunitcount，沒有改原log來配數字。','乾淨本機可用3.13.5；3.13.17不可用，沒有聲稱精確native patch的無控制重跑。SDK弱參照與GC控制的因果重現另明示。','所有exec/RPC及worker result都是privatefixtures；不是真GPU/teacher正控制。','本reviewer lifetime probe初次過早要求cancel後立即GC收回owner而AssertionError；保留首log，僅加入3個event-loop turns讓cancel callback釋放，再驗owner回收，沒有改author測試／timeout／SDK。','先前92P與audit-v2未含這項強制GC生命週期控制；新原判如實記錄漏掉的fixture契約，不覆寫歷史。','本判定不變更operative permit、舊報告、實作或科學資料，也不授權新dispatch。'],
})
print(json.dumps({'original_report':binding(REPORT),'portable_index':binding(INDEX),'portable_files':len(portable),'portable_bytes':sum(x['bytes'] for x in portable),'necessary_findings':['MR08']},ensure_ascii=False))
