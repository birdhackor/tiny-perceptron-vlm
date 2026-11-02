"""封存Qwen原付費attempt真實失敗證據；不是新source/重試permit或能力通過。"""
import hashlib,json,subprocess
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path('/workspace/work/omni-book-integrated-v1');HERE=Path(__file__).resolve().parent
REPORT=ROOT/'docs/course-revision-20261029-omni/infrastructure/qwen25-actual-execution-verification-v1.json'
INDEX=Path('/workspace/work/omni-qwen25-actual-execution-portable-evidence-v1.json')
def read(p):return json.loads(Path(p).read_bytes())
def binding(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def save(p,d):
 with Path(p).open('x') as f:f.write(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
assert not REPORT.exists() and not INDEX.exists()
d=read(HERE/'actual-failure-verification-results.json');assert d['all_assertions_passed'] and d['reviewer_actions']['NN_forward_or_generation_calls']==0
snap=read(HERE/'input-source-manifest.json');assert snap['source_count']==96
for row in snap['sources']:assert binding(row['source_path'])['sha256']==row['sha256']==binding(row['snapshot_path'])['sha256']
for row in snap['actual_input_artifacts']:assert binding(row['path'])==row
old={'mature-research-audit-v4.json':'a9dff64302eb183cce247765f88b98d16e6ed6b8a8a7ee8257401a770bab37f8','mature-research-preparation-recovery-permit-v1.json':'929b40a464f2cedea3021e33db3fcdabc6570552aa361dfc969ce17fd0be2fd8','mature-research-runtime-review-original-mr09-v1.json':'0ed0e372c88cf5d9c822b5b02736bccbd7f853521e67938323216c2ad610166b'}
old_refs=[]
for name,sha in old.items():
 ref=binding(ROOT/'docs/course-revision-20261029-omni/infrastructure'/name);assert ref['sha256']==sha;old_refs.append(ref)
now=datetime.now(UTC).isoformat()
save(HERE/'final-state-preservation-results.json',{'created_at':now,'reviewer_task':'/root/omni_development_execution_review','all96_source_bytes_match_original_snapshot_and_active_audit':True,'all23_original_input_artifact_fingerprints_unchanged':True,'old_records_unchanged':old_refs,'canonical_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),'reviewer_source_edits_or_commit':False,'reviewer_NN':0,'reviewer_dispatch_or_remote_writes':0,'retry_authorized':False})
names=['input-source-manifest.json','verify_actual_failure.py','actual-failure-verification-results.json','actual-failure-verification.log','input-entry-manifest.json','native-log-comparison.json','native-relevant-excerpt.json','verification-first-assertion.log','verification-second-assertion.log','final-state-preservation-results.json','seal_execution_verification.py']
portable=[binding(HERE/name) for name in names]
save(INDEX,{'schema':'omni-qwen25-actual-execution-portable-evidence-v1','created_at':now,'reviewer_task':'/root/omni_development_execution_review','report_path':str(REPORT),'files':portable,'file_count':len(portable),'total_bytes':sum(x['bytes'] for x in portable),'excluded':['full96 source snapshots including3MB data','all original/provider input orActionsZIPs and generatedPNG/WAV data','full original/native decoded job logs including preserved extra-LF copy','raw provider run/job/ledger/release/reservation responses','liveModal stderr/full diagnostic text; only fingerprint/TokenMissing recognition kept'],'location_map_instruction':'移轉小型自製probes/results/SHA manifests時保留原path/SHA並建location-map；不複製大ZIP/log/完整source或第三方全文進Git。'})
save(REPORT,{
 'schema':'omni-qwen25-actual-execution-verification-v1','created_at':now,'author_task':'/root','reviewer_task':'/root/omni_development_execution_review',
 'status':'verified_failed_stopped_for_reviewed_scope','evidence_integrity_verdict':'pass_for_bounded_failure_evidence','execution_outcome':'failed-stopped while waiting_for_task_id at shared480s startup deadline','capability_verdict':'not_assessed_no_model_results',
 'necessary_findings':[],'optional_findings':[],
 'scope':{'kind':'actual original paid attempt / provenance / bounded failure and cleanup verification','canonical_root':str(ROOT),'run_id':41015365689,'job_id':123930385331,'attempt_id':'mature-qwen25-41015365689-1','source_revision':'99baece1f20e5758dfdae9cdccdadb2d600f6029','active_audit_sha256':'a9dff64302eb183cce247765f88b98d16e6ed6b8a8a7ee8257401a770bab37f8','reviewer_NN_forward_or_generation_calls':0,'reviewer_paid_calls':0,'reviewer_dispatches':0,'reviewer_ledger_or_release_writes':0,'reviewer_model_recipe_source_changes':0,'reviewer_commit':False,'excluded':['textbook reader/correctness/coherence three-round review','Qwen capability/scientific result acceptance','actual Kev results/capability verification','new Qwen/Kev retry or permit']},
 'provider_observations':{'run':d['run'],'job_id':d['job_id'],'steps':d['steps'],'actual_Actions_ffmpeg_flite_input_gate_succeeded':True,'actual_Actions_true_Modal_SDK_gate_succeeded':True,'research_step_failed':True,'bound_artifact_finish_steps_succeeded':True,'fresh_reads':'native GitHub connector run/jobs/decodedlog/artifactmeta/ledger/reservation Gitobjects/release; draft metadata also independently crosschecked with authenticated gh GET'},
 'source_and_input_provenance':{'scope_counts':d['source_scopes'],'files_sha256':{x['path']:x['sha256'] for x in d['source96']},'all96_current_and_paid_commit_bytes_equal_active_audit':True,'real_checked_sources_and_receipt_identity_passed':True,'audit_reference':d['active_audit'],'six_metadata':d['payload']['six_metadata'],'input_payload':{k:v for k,v in d['payload'].items() if k!='six_metadata'},'source_and_original_input_manifest':binding(HERE/'input-source-manifest.json'),'all_input_entry_manifest':binding(HERE/'input-entry-manifest.json')},
 'prepared_inputs_only':{'audio_text_pairs':24,'AV_pairs':24,'speech_output_prompts':8,'planned_max_Qwen_generate_calls':104,'planned_max_Whisper_proxy_calls':8,'actual_FFmpeg_input_version':d['prepared_cases']['actual_ffmpeg_input_preparation_version'],'verified_audio_contract':'24 actual inputWAVs: monoPCM16/16kHz/nonzero/0<duration<=8s; transcript/gold/prompt exact frozen generator','verified_AV_contract':'24 actual inputpairs:8RGB112x112 frames@2fps/4s, image colors and pairedbeep references match sourcepattern;2monoPCM16/16kHz/4s inputWAVs each','these_are_model_inputs_not_model_speech_outputs':True,'model_processor_or_inference_measurement_completed':False,'executed_per_case_count_observed':None,'raw_worker_model_outputs_or_logits_observed':False},
 'failure_phase_and_limits':d['failure'],
 'observed_vs_source_derived':{'observed':['Native paid_call_started plus knownApp/Sandbox IDs; terminal phase waiting_for_task_id andTimeoutError.','No result.zip/result-summary/stdout/worker result in the18entry artifact or owned draft.','Runtimefinish06:48:35.262790; historical SDK SandboxWait3/Appstopped/tasks0 and successful cleanup checks.'],
  'source_derived':['Fixed staged controller sets uploading_fixed_input_payload only after _get_task_id completes; this branch and SHA-marker send were not reached in the retained terminal path.','The shared480deadline begins before SandboxCreate; absent per-subphase timestamps cannot divide RPCcreate andtask-ID wait duration.'],
  'not_established':['provider scheduling/bootstrap/CUDA cause','GPU OOM','actual GPUallocation/type or modeldtype','direct provider historical neural-execution telemetry','modelaccuracy/latency/TTFA/ASR/speechquality'],
  'source_derived_not_uploading_is_not_a_fabricated_model_metric':True},
 'cost_reservation_and_durability':d['finance'],
 'retention_and_ownership':{'official_Actions_artifact':d['artifact'],'provided_originalZIP':binding('/workspace/work/omni-qwen25-original-attempt-41015365689/original-provider-artifact.zip'),'all18_entries_sha_size_equal_provided_extraction':True,'all18_entry_bindings':d['artifact18_entries'],'owned_draft':d['draft_release'],'only_asset':d['only_release_asset'],'no_result_asset':True},
 'cleanup':d['cleanup'],
 'future_attempt_authority':{'ordinary_Qwen_once_consumed':True,'failed_attempt3_10_full_bound_retained':True,'all_round14_00_not_refunded':True,'sole929b_Kev_permit_already_consumed':True,'current_once_guards':d['once_guards'],'Qwen_retry_authorized':False,'Kev_retry_authorized':False,'new_permit_issued':False},
 'native_log_integrity':d['native_log_comparison'],
 'author_summary_crosscheck':d['author_summary_consistency'],
 'evidence':{'full_independent_results':binding(HERE/'actual-failure-verification-results.json'),'probe_log':binding(HERE/'actual-failure-verification.log'),'safe_native_excerpt':binding(HERE/'native-relevant-excerpt.json'),'final_preservation_check':binding(HERE/'final-state-preservation-results.json')},
 'portable_evidence':binding(INDEX),'preserved_old_records':old_refs,
 'limitations':d['limits']+[
  'Fresh live Modal app query failed TokenMissing; target stop proof is explicitly contemporaneous controller/officialApp inventory from original provider artifact plus durable ledger, not a fresh liveModal state claim.',
  'Official connector decoded native log is95189B/SHA66a0d1d1882de9e3ed254fa5b994169e64f3c2aa04d04afd1112935a4836531e. Provided preserved95190B/SHAe5b5fab5cecd30875e6798b4b30fafbcc7fd49fea42562880fae15fd544328a3 has exactlyone extra finalLF; both preserved, neither rewritten to makehashesmatch.',
  'USD3.10 is a retained reservation ceiling, not an observed invoice. The historical basegross guard18.91075921 and doubledfuture protection36.91075921 are conservative guards, not spending.',
  'Local verification harness first corrected an overstrict equality expectation for later-added cleanup-check fields; second corrected its independent future-guard formula to the documented doubled-bound formula. Both diagnostic logs remain; no author source/data/report was changed.',
  'Author qwen summary was only crosschecked for observed fields against original evidence; it did not serve as the authority and its independentpending field was not edited.',
 ]})
print(json.dumps({'report':binding(REPORT),'portable_index':binding(INDEX),'portable_files':len(portable),'portable_bytes':sum(x['bytes'] for x in portable),'execution_outcome':'failed-stopped','reviewer_NN':0,'necessary_findings':0,'retry_authorized':False},ensure_ascii=False))
