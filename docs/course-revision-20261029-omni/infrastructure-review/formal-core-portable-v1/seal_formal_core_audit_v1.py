"""Independent reviewed-scope pass after original/current counterexamples close."""
import datetime,hashlib,json,pathlib
B=pathlib.Path('/workspace/work/omni-formal-core-review-v1');R=pathlib.Path('/workspace/work/omni-models-v1');S=B/'current-v2-source';D=R/'docs/course-revision-20261029-omni/infrastructure';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):
 p=pathlib.Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
recipe=json.loads((S/'docs/omni-products/v1/formal-recipes.json').read_bytes());metrics=json.loads((S/'docs/omni-products/v1/metrics.json').read_bytes());old=json.loads((B/'original-source/docs/omni-products/v1/formal-recipes.json').read_bytes());v1=json.loads((B/'current-source/docs/omni-products/v1/formal-recipes.json').read_bytes())
required=set(recipe['execution_sources_sha256'])|set(metrics['measurement_sources_sha256'])|{'tiny_perceptron/omni_products/'+n for n in recipe['implementation_sha256']}|{'docs/omni-products/v1/'+n+'.json' for n in ('data-definition','metrics','formal-recipes')}
assert len(recipe['execution_sources_sha256'])==25 and len(recipe['implementation_sha256'])==8 and len(required)==36
assert old['jobs']==v1['jobs']==recipe['jobs']
assert old['data_definition_sha256']==recipe['data_definition_sha256']=='66a37645b983ceba49c283834eb273a8381015bba735130ae703f0c0d4d7f73e'
assert old['metrics_sha256']==recipe['metrics_sha256']=='cb291a08265a96526477bf8e0350243c8ddf9564600ba691d7a7f6aee3f6b336'
assert old['candidate_selection']==recipe['candidate_selection']
assert v1['implementation_sha256']==recipe['implementation_sha256']
assert sha(S/'docs/omni-products/v1/formal-recipes.json')=='f6215bb75618badb890831e7851d678df7bddf6d0a656a9c3c7cf49d613ebb87'
assert all((R/n).read_bytes()==(S/n).read_bytes() for n in required)
assert not (R/'outputs/omni-formal-core-v1').exists()
immutable={'formal-core-review-original-v1.json':'5c6258033afd295f2636e44905d3c6a3dd300963f654ad56063b582b0c13accc','formal-core-review-current-v1.json':'c881250d6517e916255ad851e6cad5ea853cc9083582ca8286741bc0460383b5','formal-recipes-pre-fix-v1.json':'973384ca85169eafd9aa9ef1085b8ba3c0a16d40d5ba97dfa2fcbfd4267a3c8e','formal-recipes-pre-fix-v2.json':'754ac72f01ce61e48757c23655bcb4531072181f2d68c05676f62278748609db','experiment-execution-audit-v1.json':'c430389a891fc6b51deca8ee857773fd5565f05d6a0ee4b6bcc37ef960eedb19','experiment-execution-audit-v2.json':'3563c244963b97f1f79023368145c80da379a29c21fdb59a81410c7fe1e166af'}
assert all(sha(D/n)==h for n,h in immutable.items())
closure=json.loads((B/'current-v2-source-closure-result.json').read_bytes());gate=json.loads((B/'current-v2-gate-selection-result.json').read_bytes());boundary=json.loads((B/'current-v2-cpu-boundaries-result.json').read_bytes())
assert closure['all_required_negative_controls_rejected'] is True and gate['six_way_tie']['rule_followed'] is True
assert gate['codec_gate']['talker_trainer_reached'] is False and gate['codec_gate']['partial_with_formal_looking_receipts_rejected'] is True
assert boundary['launch_interruption']['all_training_processes_exited_flag'] is False and boundary['launch_interruption']['review_fixture_child_cleanup_observed'] is True
assert next(x for x in boundary['finish_cases'] if x['case']=='lost_start_false_flag')['status']=='unresolved'
assert '53 passed in 4.53s' in (B/'current-v2-pytest.log').read_text()
evidence_names=('current-v2-pytest.log','current-v2-source-closure-result.json','current-v2-gate-selection-result.json','current-v2-cpu-boundaries-result.json','current-v2-protocol-result.json','current-resume-provenance-result.json','current-runtime-limits-result.json','current-checked-job-result.json','current-pricing-result.json','original-source-guard-controls-result.json')
scripts=('probe_current_v2_source_closure.py','probe_current_v2_gate_selection.py','probe_current_v2_cpu_boundaries.py','probe_protocol.py','probe_resume_provenance.py','probe_runtime_limits.py','probe_checked_job.py','probe_source_guard_controls.py','probe_current_pricing.py','finalize_current_pricing.py')
rechecked=[
 {'id':'F01','status':'resolved','verification':'actual tiny two-update development codec、缺正式origin/result及有formal-looking partial收據皆先拒絕，未進Talker trainer；同配方成功complete6000/config/source/data/metrics及選定SHA是targets前置。','evidence':['current-v2-gate-selection-result.json']},
 {'id':'F02','status':'resolved','verification':'checkpoint metadata綁config/revision/implementation/runtime，expected_metadata在model/optimizer restore前比較。原fresh-process跨來源2步→續算反例拒絕；本輪trainer與八個implementation SHA逐bytes不變，53測試再次包含before-mutation拒絕控制。','evidence':['current-resume-provenance-result.json','current-v2-pytest.log']},
 {'id':'F03','status':'resolved','verification':'缺POSIX alarm在任何output/origin/lock前拒絕，finally只取消已安裝alarm；本輪再次跑formal portability測試，變更僅新增measurements gate。','evidence':['current-checked-job-result.json','current-runtime-limits-result.json','current-v2-pytest.log']},
 {'id':'F04','status':'resolved','verification':'latest保存原best checkpoint bytes且best不遞迴內嵌。原同來源codec4步對2+2新目錄參數一致且best可load；本輪相同trainer再次跑無dev改善的新目錄續算測試。','evidence':['current-runtime-limits-result.json','current-v2-pytest.log']},
 {'id':'F05','status':'resolved','verification':'checked_job實核metrics.measurement_sources及audit map；CPU git bytes範圍含measurement。25執行來源另固定legacy finance、reliability與三個實際package init。七個fresh processes獨立驗：穩定來源通過；reliability/legacy finance/package init漂移、缺measurement audit、git bytes錯與HEAD錯皆先拒絕。','evidence':['current-v2-source-closure-result.json']},
 {'id':'F06','status':'resolved','verification':'六A key依dev_score、-best.update、-job_order；原同分1000對500 fixture現選av-time-rope-s29 update500。單job相同dev仍保留較早checkpoint。','evidence':['current-v2-gate-selection-result.json']},
 {'id':'F07','status':'resolved','verification':'Popen前即False且建立在try內。OS真建立child但回應中斷的原反例，child活時旗標False，finish保持unresolved；reviewer隨即kill+wait並觀測退出。已取得handle的deadline會kill/wait；未知wait仍False。','evidence':['current-v2-cpu-boundaries-result.json','current-v2-pytest.log']},
 {'id':'O01','status':'resolved','verification':'metrics的CONTRACTS repo路徑已有原bytes，SHA7eff498f798641967478fe364e2fa89a262ed1fec96fc2cf6a9887c051fe8388。','evidence':['current-v2-protocol-result.json']}]
record={
 'schema':'omni-formal-core-audit-v1','created_at_utc':datetime.datetime.now(datetime.UTC).isoformat(),'author_task':'/root','reviewer_task':'/root/omni_development_execution_review','status':'pass_for_reviewed_scope','verdict':'pass','necessary_findings':[],'optional_findings':[],
 'scope':{'description':'僅十個已凍結、fresh random initialization的正式CPU核心工作與前final的codec/候選/T/tau選擇及其有界保存控制。','job_ids':recipe['job_order'],'maximum_total_core_attempts':10,'maximum_attempts_per_job':1,'execution_host':'single standard GitHub-hosted ubuntu-latest CPU runner','workflow_timeout_minutes':300,'proposed_artifact_reservation_usd':'0.10','USD60_round_unchanged':True,'future_core_research_final_protection_unchanged':True,'no_Modal_compute':True,'final_opened':False,'reviewer_ran_formal_training':False,'reviewer_new_reservations':0,'reviewer_paid_dispatches':0,'reviewer_final_model_predictions':0,'excluded':['實際6000/1000-update正式訓練成功或產物已存在','任何模型能力／promotion或final gate通過','mature inference/research workers及其新付費source','教材三輪獨立審閱完成','core_final/core_evaluation/qwen_inputs/kev_research/staged_execution等未審工程草稿']},
 'formal_recipe_sha256':sha(S/'docs/omni-products/v1/formal-recipes.json'),'reviewed_files_sha256':{n:sha(S/n) for n in sorted(required)},'reviewed_files_count':len(required),'scope_set':{'execution_sources':25,'implementation_sources':8,'frozen_json':3,'measurement_sources_included':list(metrics['measurement_sources_sha256'])},
 'source_snapshot':ref(B/'current-v2-source-manifest.json'),'previous_judgments':{n:ref(D/n) for n in immutable},'recheck_findings':rechecked,
 'execution_evidence':{n:ref(B/n) for n in evidence_names},'probe_sources':{n:ref(B/n) for n in scripts},
 'verification':{
  'tests':'UV_OFFLINE=1 uv run --extra cpu python -m pytest tests/test_omni_formal_protocol.py tests/test_omni_cpu_execution.py tests/test_omni_training_runtime.py tests/test_omni_products.py tests/test_omni_products_protocol_fixes.py tests/test_omni_inference.py -q；本人53 passed in4.53s。',
  'source_freeze':'寫audit前36個required來源與01:29:40.113327 UTC snapshot逐bytes相同。實際core入口要再核reviewed SHA、exact Torch、fresh checkout GITHUB_SHA及git show bytes；來源尚待作者提交，不在本次偽造任何commit或HEAD通過。',
  'data_to_metrics_to_recipe':'data00:48:35.879377 -> metrics00:49:23.164142 -> current recipe；data66a37645...、metricscb291a08...及十job全部updates/batch/LR/seed/caps不變。',
  'six_A_fairness':'三seed11/29/47每對additive/time-rope初值SHA相同、119693可學參數與8個per-head頻率一致，additive重複4份。配對兩題皆正確為主要dev分，缺線索分僅同分決選，checkpoint與job順序現都按凍結規則。',
  'C_control':'compact3612為augmented10836原variant0精確子集，不補unknown；同架構/seed/batch/updates與實際捕捉route half。C augmented固定primary、compact只作data control。',
  'sound_targets_and_provenance':'formal codec完整6000步且同recipe/source的成功origin/result與匯出SHA先freeze，才准Talker targets。codec只用sound train；Talker optimizer不含codec，dev使用自由自回歸。舊plumbing漂移的AV/C development不當正式初值或來源證明。',
  'once_and_storage':'main manual workflow、相同global concurrency group、ledger purpose cpu-core-formal跨run連失敗都拒絕重跑；fresh唯讀inventory/billing後先CAS0.10，才進core；不退款或釋回既有2.10。',
  'credentials':'Modal keys只在reserve，checkout不保存credentials；actual child測試核GH_TOKEN/GITHUB_TOKEN/MODAL_TOKEN_ID/MODAL_TOKEN_SECRET皆不在子環境。正式CPU路徑不建Modal App/Sandbox/Image/Secret/Volume。',
  'stop_and_finish':'已知child每次poll/kill/wait觀測；start回應或wait未知保持False/unresolved。finish需owned draft exact資產ID/name/size、先前真bytes/SHA回讀、Actions artifact expiry/size與process exited均核過才terminal。',
  'export_bounds':'每job完整權重/optimizer/RNG/checkpoint archive≤128MiB，逐完成先owned draft上傳並下載驗SHA再下一job；最後package保留全十份及selection。Actions control≤16MiB、zip合計≤128MiB、aggregate≤256MiB，upload前真check、retention7日。',
  'cost_evidence':'本人01:26:24 UTC真官方pricing與公開repo讀取：standard public GitHub runner免費；Actions artifact .25USD/decimalGB-month，production parser真通過；.02/GiB/day保守ceiling×256MiB×7天向上.04，另.06cleanup，保留.10。Reserve會再讀當前官方來源，價格超ceiling拒絕。Release官方短摘錄無總size/bandwidth限制，單檔<2GiB；本scope實施更小128MiB。',
  'current_closure':'F01–F07 necessary全部由本人原反例/guard控制回核收束；不把作者pytest或既有protocol review當成獨立審核替代。'},
 'limitations':['這是source/protocol/成本與停止保存邊界的reviewed-scope pass，沒有正式模型已成功或能力已通過的主張。','微型模型證據僅2–4 updates的train/dev；沒有本次6000/1000正式run、production Actions保存或新權重SHA可簽。','GitHub/ledger/HEAD/finish的失敗路徑為清楚標記的私有離線fixtures；唯一真CPU child均由reviewer kill+wait清除。','真網路只讀官方pricing/docs與公開repo metadata，原始第三方內容留Git之外；引用以短摘錄、URL、path與SHA保存。','將來首跑仍須actual main commit/source/audit bytes、fresh billing/inventory、CAS與workflow各guard通過；本audit不提供自動重跑、final或mature worker授權。']}
p=D/'formal-core-audit-v1.json'
with p.open('x') as f:f.write(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(p),'sha256':sha(p),'created_at_utc':record['created_at_utc'],'status':record['status'],'reviewed_files_count':len(required)},ensure_ascii=False,indent=2))
