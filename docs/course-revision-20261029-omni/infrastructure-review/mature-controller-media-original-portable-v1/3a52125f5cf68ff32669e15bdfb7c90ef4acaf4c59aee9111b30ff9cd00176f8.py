"""封存MR09新original；保留舊scope、已用permit與所有已封判定。"""
import hashlib,json
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path('/workspace/work/omni-models-v1');HERE=Path(__file__).resolve().parent
REPORT=ROOT/'docs/course-revision-20261029-omni/infrastructure/mature-research-runtime-review-original-mr09-v1.json'
INDEX=Path('/workspace/work/omni-mature-controller-media-portable-evidence-original-v1.json')
def read(p):return json.loads(Path(p).read_bytes())
def binding(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def save(p,d):
 with Path(p).open('x') as f:f.write(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
assert not REPORT.exists() and not INDEX.exists()
snap=read(HERE/'source-snapshot-manifest.json');assert snap['source_count']==96
for row in snap['files']:assert binding(row['original_path'])['sha256']==row['sha256']==binding(row['snapshot_path'])['sha256']
probe=read(HERE/'media-prerequisite-probe-results.json');assert probe['all_assertions_passed']
jobs=read(HERE/'provider-jobs.json')['structuredContent']['jobs'];assert len(jobs)==1 and jobs[0]['id']==123920392038
steps={s['name']:s['conclusion'] for s in jobs[0]['steps']};assert steps['Verify the real Modal filesystem contract offline before reservation']=='success' and steps['Prepare and reserve once, infer, export and stop the exact owned resources']=='failure'
for name in ('Check all evidence size before artifact upload','Retain complete or partial original evidence for seven days','Verify provider artifact limits and finish the exact reservation without refund'):assert steps[name]=='skipped'
lr=read(HERE/'provider-ledger.json')['structuredContent'];ledger=json.loads(lr['content']);assert ledger['reserved_total_usd']=='10.90'
assert not any(x['attempt_id'].startswith('mature-qwen25-') for x in ledger['attempts'])
kev=[x for x in ledger['attempts'] if x['attempt_id'].startswith('mature-kev-')];assert len(kev)==2 and {x['reserved_usd'] for x in kev}=={'4.30'}
old_digests={
 'mature-research-audit-v2.json':'da64a8272155e03e01d1b4921669205b1c9773880c291159de5c3007cdfa4d27',
 'mature-research-audit-v3.json':'53a5f325e6b4800f0f12be56df31551423a9bbbbf6a070753f4e62f1d5559a61',
 'mature-research-runtime-review-original-mr08-v1.json':'1c178bf2d7b751638a62db31ab68d1ba6c9bddd4b15ffcd24c62a674262025a2',
 'mature-research-runtime-review-current-mr08-v1.json':'ff805e800214fd050a8239e94c1f35b5e6a19e67bd07d6113dca7c0331175859',
 'mature-research-preparation-recovery-permit-v1.json':'929b40a464f2cedea3021e33db3fcdabc6570552aa361dfc969ce17fd0be2fd8',
}
old=[]
for name,sha in old_digests.items():
 b=binding(ROOT/'docs/course-revision-20261029-omni/infrastructure'/name);assert b['sha256']==sha;old.append(b)
now=datetime.now(UTC).isoformat()
save(HERE/'provider-and-preservation-results.json',{'created_at':now,'reviewer_task':'/root/omni_development_execution_review','all_assertions_passed':True,'job_id':123920392038,'steps':steps,'ledger_blob':lr['sha'],'retained_bounds_usd':'10.90','qwen_rows':0,'kev_rows':[{'attempt_id':x['attempt_id'],'status':x['status'],'reserved_usd':x['reserved_usd']} for x in kev],'sole_kev_permit_used':True,'kev_recovery_guard_closed':True,'source96_unchanged':True,'preserved_prior_records':old,'actual_kev_result_or_capability_reviewed':False})
portable_names=['source-snapshot-manifest.json','native-error-excerpt.json','probe_missing_media_prerequisite.py','media-prerequisite-probe-results.json','media-prerequisite-probe.log','provider-and-preservation-results.json','seal_mr09_original.py']
portable=[binding(HERE/n) for n in portable_names]
save(INDEX,{'schema':'omni-mature-controller-media-original-portable-evidence-v1','created_at':now,'reviewer_task':'/root/omni_development_execution_review','report_path':str(REPORT),'files':portable,'file_count':len(portable),'total_bytes':sum(x['bytes'] for x in portable),'excluded':['full96 source snapshots and3MB data','raw native log and provider responses','all generated WAV/PNG/text/cases material; only fingerprints/counts/PCM metadata retained','all prior evidence/report/audit/used permit bytes unchanged'],'location_map_instruction':'保留小型自製證據原path/SHA，搬移用location-map；fullsnapshots/第三方全文/native全文/生成媒體不進普通Git。'})
save(REPORT,{
 'schema':'omni-mature-research-runtime-original-mr09-v1','created_at':now,'author_task':'/root','reviewer_task':'/root/omni_development_execution_review','status':'changes_required',
 'verdict':'MR09成立：clean Actions controller未宣告/驗證ffmpeg flite/slt前置，Qwen在prepare第一個FFmpeg呼叫即ENOENT，尚未reserve/CAS。不得以本機已有binary當作Actions能力證明。',
 'scope':{'run_id':41012209461,'job_id':123920392038,'source_revision':snap['head'],'source_count':96,'worker_scientific_model_sources_changed':False,'reviewer_paid_calls':0,'reviewer_modal_rpc_calls':0,'reviewer_ledger_or_release_writes':0,'reviewer_nn_predictions':0,'actual_kev_verification_excluded':True},
 'source_snapshot':binding(HERE/'source-snapshot-manifest.json'),'files':{x['path']:x['sha256'] for x in snap['files']},'preserved_prior_judgments_and_used_sole_permit':old,
 'necessary_findings':[{
  'id':'MR09','severity':'blocking_controller_preCAS_prerequisite','locations':['.github/workflows/omni-mature-research.yml:44','scripts/omni_revision/research_execution.py:537','scripts/omni_revision/research_execution.py:264','scripts/omni_revision/research_cases.py:82'],
  'summary':'workflow只安裝Python依賴與跑SDK契約，沒有提供可執行ffmpeg及flite/slt能力；原Qwen材料生成器直接要求此OS工具。',
  'actual_native_phase':'run → prepare → research_cases.build → subprocess.run first ffmpeg; FileNotFoundError executable ffmpeg',
  'source_order':'input prepare先於Workspace/rates/inventory、reserve/CAS、Releasecreate、Modalexecutor；首次合成即失敗，來源中CAS分支不可達。',
  'required_fix':['只修controller workflow：在prepare/reservation之前，明確安裝/提供FFmpeg，安裝或驗證失敗必須停在pre-CAS。','實跑FFmpeg lavfi flite、voice=slt轉出短WAV，驗PCM16/mono/16kHz且有限非空duration；只which、version或列filter不足以證明voice可用。Flite在此是FFmpeg filter/library，並非必須另找獨立flite executable。','保留原research_cases/24audio/24AVpairs/8speech、模型、worker、精度、GPU/caps、來源pins與所有科學條件；不用其他TTS/少題/預存答案繞過。','controller AUDIT改新不可變v4，修正後另做獨立current/full96 source audit；不改v3與全部原件。'],
  'required_rechecks':['本次真正OS missing-executable負例仍須在prepare前置或prepare中拒絕，任何reserve/writer/release/Modal 0。','安裝後實際flite/slt合成正控制；無filter/voice、非PCM16或空輸出應失敗。','workflow順序與source guard exactbytes重核；實際clean Actions安裝正例只能等未來真run，不拿本機binary冒充。'],
 }],
 'optional_findings':[],
 'independent_reproduction':{'probe':binding(HERE/'media-prerequisite-probe-results.json'),'negative':'真run/prepare/build；ffmpeg子process僅令PATH空，作真OS lookup而非手寫FileNotFoundError；只有ledger_read_fixture，其他paid/write守門皆未觸及。','local_generator_positive':{'actual_ffmpeg_version':probe['local_actual_generator_positive']['ffmpeg_version'],'audio_pairs':24,'av_pairs':24,'speech_prompts':8,'audio_pcm_contract':'16kHz / mono / signed16-bit WAV / 0<duration<=8s','NN_or_teacher_outputs_used':False,'clean_actions_install_capability_proven':False}},
 'provider_and_finance':{'evidence':binding(HERE/'provider-and-preservation-results.json'),'sdk_step_actual_success':True,'research_step_actual_failure':True,'bound_upload_finish_skipped':True,'fresh_ledger_blob':lr['sha'],'retained_bounds_usd':'10.90','no_qwen_reservation_row':True,'qwen_general_once_remains_unconsumed':True,'sole929b_kev_repair_permit_used':True,'no_kev_rerun_authorized':True,'actual_kev_completed_status_is_not_ability_verdict':True},
 'future_scope_constraint':'本次pre-CAS Qwen失敗不需也不准新repair permit；Qwen仍只剩原一般一次paid attempt。929b已由Kev41011677439使用，未來audit-v4只能延續既有程式來源核准，不重新開放Kev、不重設permit次數、不退款10.90或改舊receipt。',
 'evidence':{'native_short_excerpt':binding(HERE/'native-error-excerpt.json'),'local_probe_log':binding(HERE/'media-prerequisite-probe.log')},'portable_evidence':binding(INDEX),
 'limitations':['本機FFmpeg7.1.5實跑原generator成功，不代表Ubuntu Actions future apt安裝/編譯選項/voice可用；這一能力仍未驗證。','realrun負例只對ffmpeg lookup控制PATH；ledger與strictcore是明示readonly fixtures，不能稱重跑了freshActions fullcore wrapper。','所有生成材料在本人外部review目錄；不放正式outputroot，不做NN推論或teacher weights下載。','Kev實際輸出由另一owner核對，本判定只讀帳本operational status/permit consumption，不代簽其結果或能力。','原native全文有bindings，僅留原SHA與必要trace摘錄，不放全文或環境至context/ordinaryGit。'],
})
print(json.dumps({'original_report':binding(REPORT),'portable_index':binding(INDEX),'portable_files':len(portable),'portable_bytes':sum(x['bytes'] for x in portable),'necessary_findings':['MR09']},ensure_ascii=False))
