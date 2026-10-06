"""Assemble this reviewer's completed independent report, never read older reports."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return str(Path(p).relative_to(ROOT))
env={'python':'3.13.5','torch':'2.14.1+cpu','device':'cpu','safetensors':'0.8.0','huggingface_hub':'1.33.0'}
audit=json.loads((OUT/'audit-result.json').read_bytes())
artifacts=[]
def artifact(ident,kind,name,description,command=None,result=None):
    p=OUT/name;d={'id':ident,'kind':kind,'path':rel(p),'sha256':sha(p),'description':description}
    if kind=='execution':d.update(command=command,result=result,environment=env)
    artifacts.append(d);return ident
for name,description,result in [
 ('audit','人工判準、固定LFS材料、15段收據、原CPU記錄、最後測試完成及parser/錯種checkpoint核對',
  'exit0；8962個archive members均符合固定SHA，15段completed；8次chat+1append原始stdout SHA一致；兩版final各3734；4個training argv可解析；3個錯種checkpoint拒絕'),
 ('public-actual','fixed HF979cdf... 的兩個public validation情境真正CPU推論',
  'exit0；text生成一次EOS；tool真正執行計算器，第二次同核生成結果是416並EOS；兩者與原CPU答案一致'),
 ('contract-audit','小型CPU隨機初始化、train-only字表、公開下載身分拒絕與AST載入契約',
  'exit0；同seed所有張量相同，異seed68張量改變且涵蓋四個神經模組；test-only符號為UNK；main revision及../prefix在下載前被拒絕'),
 ('history-audit','歷史RAG/tools/reasoning原輸入、原生成及分母的限定pointer查核',
  'exit0；RAG84個保存sample/96test、tools20sample/20test、reasoning24sample/24test，原test有效token696/1289/51；沒有重評品質'),
 ('link-audit','把真正取得的public權重識別連回completed native及固定final記錄',
  'exit0；public model SHA26102d...及manifest SHAf27856...等於native原outer receipt及final manifest；completed4000、selected1000、native parent selected3000')]:
    script={'public-actual':'run-public','audit':'audit','contract-audit':'contract-audit','history-audit':'history-audit','link-audit':'link-audit'}[name]
    command=f'/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-T.11/{script}.py > docs/technical-reviews/artifacts/p6-T.11/{name if name != "audit" else "audit"}.stdout.txt 2> docs/technical-reviews/artifacts/p6-T.11/{name if name != "audit" else "audit"}.stderr.txt'
    # public-actual driver stores public-actual-result.json; the others use the same pattern.
    artifact('a_'+script.replace('-','_'),'execution',name+'-result.json',description,command,result)
    artifact('code_'+script.replace('-','_'),'code',script+'.py','本人實際CPU查核程式；未修改原trainer')
    artifact('stdout_'+script.replace('-','_'),'source_snapshot',name+'.stdout.txt','當次實際stdout；與對應JSON及命令配對')
    artifact('stderr_'+script.replace('-','_'),'source_snapshot',name+'.stderr.txt','當次實際stderr；成功執行時為空')
artifact('a_text_stdout','source_snapshot','text.actual.stdout.json','真正public CPU text的完整模型輸出、generated ids、EOS及模型識別')
artifact('a_tool_stdout','source_snapshot','tool_call.actual.stdout.json','真正public CPU tool的完整請求、實際工具結果、第二次prompt與生成')
artifact('a_pointer','source_snapshot','lfs-pointer.txt','git show固定資料commit返回的原LFS pointer')
artifact('a_public_manifest','source_snapshot','public-inference-manifest.original.json','fixed HF979cdf...下載之原manifest完整副本；selected_step等具名pointer查核')
artifact('a_inspection','source_snapshot','inspection-record.json','本人實際read ranges、原小節SHA及各次失敗/修正紀錄；未讀舊審閱')
artifact('a_section','source_snapshot','section-original.md','首次讀取之原始UTF-8 T.11 bytes')
artifact('a_ast','source_snapshot','ast-map.json','先以AST定位必要原方法；避免作者額外結果解釋常數')
artifact('a_topkeys','source_snapshot','raw-topkeys.json','讀原始結果前先列top-level key/type及原檔SHA')
raw_groups={'training':[],'cpu':[],'final':[],'manifest':[],'historical':[]}
for i,item in enumerate(audit['preserved_original_sha256']):
    p=ROOT/item['copy']
    if '/originals/' not in item['copy']:continue
    ident=f'raw_{i}'
    artifacts.append({'id':ident,'kind':'source_snapshot','path':item['copy'],'sha256':item['sha256'],
       'description':'未刪改原件副本，已親核原件/副本SHA一致；真正讀取pointer見audit-result.json：'+item['original']})
    group='training' if 'training-raw' in item['original'] else 'cpu' if 'public-cpu-raw' in item['original'] else 'final' if 'public-raw/' in item['original'] else 'manifest'
    raw_groups[group].append(ident)
history=json.loads((OUT/'history-audit-result.json').read_bytes())
for x in history['rows']:
    ident='raw_history_'+x['experiment'];raw_groups['historical'].append(ident)
    artifacts.append({'id':ident,'kind':'source_snapshot','path':x['copy'],'sha256':x['sha256'],
       'description':'未刪改原歷史測量；只讀具名原輸入/生成/分母pointer，未讀limits及作者評語'})
sources=[]
def code(ident,path,note):
    p=OUT/'code'/path
    sources.append({'id':ident,'kind':'repository_code','title':path+' 原方法快照','path':rel(p),'sha256':sha(p),
                    'version':'selected local Git HEAD '+audit['source_identity']+'; full-file SHA frozen at inspection',
                    'verified':True,'inspection_note':note})
code('s_wrapper','scripts/selftrained/train_local_stage.py','AST後直接讀36-115及118-251：fixed declaration、source SHA、own best/latest sidecars、subprocess、完整步數判準。')
code('s_train','scripts/selftrained/train.py','讀export235-281、weighted/native454-516、parser631-686、main689-875及951-1004；確認random/new-stage/resume與safe export；未讀額外作者解釋常數。')
code('s_transport','scripts/selftrained/hf_transport.py','AST後精讀229-300 verify_file/unpack/materialize：full Git commit+LFS pointer+object hash+bounded extraction。')
code('s_inference','tiny_perceptron/selftrained/inference.py','AST後精讀74-136及215-328；四個safe檔、token=False、fixed 40hex、manifest/payload checks、同模型跨tool hop。')
code('s_tool','tiny_perceptron/selftrained/tools.py','AST後讀122-170 run_tool_loop；明確解析、真正計算、tool role訊息與第二次generation。')
code('s_model','tiny_perceptron/selftrained/model.py','AST後讀LM與三個encoder constructor及LimitedAssistant295-302；nn modules新建，無成熟模型載入。')
code('s_base_model','tiny_perceptron/model.py','AST定位後讀Block.__init__32-41與TinyLM.__init__54-66；Embedding/Linear/Block新建，無pretrained checkpoint。')
code('s_dataset','tiny_perceptron/selftrained/dataset.py','AST後讀train_tokenizer84-91：只收train文本；固定ASCII及已知OCR字元是方法規約。')
code('s_native_gate','scripts/selftrained/modal_runner.py','只讀AST定位後680-854 native_target_source_options_gate/native_joint_source_gate；函式只核對本機paths/sidecars，不需要Modal API或作者Volume；名稱/docstring不是額外外部依賴。')
code('s_mature','tiny_perceptron/natural_assistant.py','AST後讀load_core263-306、load_asr687-697：Qwen3VL及Whisper明確from_pretrained；train分支可選LoRA；未載入大型底座或執行GPU。')
code('s_extras','pyproject.toml','27-30與52-74：cpu、cu126、cu130為互斥Torch extra，selftrained固定safetensors0.8.0；GPU依路線另選，CPU已實核。')
for ident,art in [('s_audit','a_audit'),('s_public','a_run_public'),('s_contract','a_contract_audit'),('s_history','a_history_audit'),('s_link','a_link_audit')]:
    sources.append({'id':ident,'kind':'execution','title':'本人'+ident+'有界查核','verified':True,'artifact_id':art})
official=[
 ('s_hash','python-hashlib.rst','Python hashlib','https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/hashlib.rst','CPython v3.13.5','Python官方語言文件原碼','Hash algorithms第38-48行：對bytes計算digest；不檢查答案語義或模型能力。'),
 ('s_git','git-commit.adoc','git-commit','https://raw.githubusercontent.com/git/git/v2.50.1/Documentation/git-commit.adoc','Git v2.50.1','Git官方實作倉庫的命令文件','DESCRIPTION第22-28行：commit保存index內容及log metadata；辨識程式snapshot，無品質保證。'),
 ('s_lfs','git-lfs-spec.md','Git LFS pointer specification','https://raw.githubusercontent.com/git-lfs/git-lfs/v3.7.0/docs/spec.md','Git LFS v3.7.0','Git LFS官方實作倉庫的pointer規格','第30-47行：version、oid sha256、size bytes；已核對本資料pointer原bytes。'),
 ('s_cv','sklearn-cross-validation.rst','Cross-validation: evaluating estimator performance','https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst','scikit-learn 1.7.2','scikit-learn官方評測方法文件原碼','第60-82行：用test調hyperparameter會leak；validation選設定後final用留出test；不是保證未受污染的稽核。'),
 ('s_function','openai-function-calling.ipynb','How to call functions with chat models','https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/How_to_call_functions_with_chat_models.ipynb','main snapshot accessed2026-10-06 SHA256a2411d7927ecbc42e03548ce1423e80003d2f9bbc4b6d0eb638f1f0322d8e946','OpenAI官方API cookbook原notebook','本人讀markdown cells0及35：模型只生成arguments；應用真正執行function、把result append至messages、再請模型生成回答。')]
for ident,name,title,url,version,authority,note in official:
    p=OUT/'official'/name
    sources.append({'id':ident,'kind':'official_docs','title':title,'url':url,'version':version,'verified':True,
                    'checked_original':True,'accessed_on':'2026-10-06','authority_reason':authority,'inspection_note':note})
    artifact('official_'+ident,'source_snapshot','official/'+name,'本人取得及閱讀的官方原文件完整快照；實讀範圍見sources')

claims=[]
def e(source,locator,supports):return {'source_id':source,'locator':locator,'supports':supports}
def claim(ident,kind,statement,location,evidence,arts,scope,expected=None,observed=None,details=None,denominators=None):
    c={'id':ident,'kind':kind,'statement':statement,'location':location,'status':'verified','evidence':evidence,'artifact_ids':arts,'scope':scope}
    if kind!='concept':
        c['verification']={'method':'executed','expected':expected,'observed':observed,'details':details}
        if kind=='numeric':c['verification']['tolerance']='整數與布林精確相等；比率4/5與3/5按Python浮點值0.8/0.6表示。'
        if denominators:c['verification']['denominators']=denominators
    claims.append(c)

claim('c1','concept','有限量化比較應保存起點版本、資料切分/指紋、轉換與兩版評估命令、生成條件、逐題輸出與量測範圍；這是讓他人核對比較的方法建議。','首段與六列紀錄表',
 [e('s_git','DESCRIPTION22-28','版本綁定原內容'),e('s_hash','Hash algorithms38-48','bytes指紋'),e('s_cv','60-82','切分與選設定條件'),e('s_function','cells0,35','輸出與真正操作結果分別記錄')],['a_inspection'],
 '這是一份可追查配方的建議；不聲稱記錄完整便能重現所有硬體浮點結果，也不聲稱已實跑量化。未量欄位標未量測是紀錄規則。')
claim('c2','concept','SHA-256辨識檔案bytes、Git commit辨識程式snapshot；兩者不會自動證明資料答案正確或模型有能力。','SHA-256與Git commit段',
 [e('s_hash','38-48','hash輸入是bytes且輸出digest'),e('s_git','22-28','commit內容為index與metadata')],['official_s_hash','official_s_git','a_pointer'],
 '內容識別與語義/能力驗收分開；不把SHA視為下載地址或絕對無碰撞的數學保證。')
claim('c3','numeric','手寫例中80000bytes變60000bytes符合檔案較小，但4/5變3/5不符合答對數不減。','4-bit手寫判準例',
 [e('s_audit','/numeric','本人直接代入80000/60000、4/3、共同分母5比較')],['a_audit'],
 '純人工判準，沒有真模型、真量化、時間或RAM測量。','smaller=True且preserved=False','60000<80000=True；3>=4=False；0.8→0.6','audit.py整數直接比較，保存原參數。')
claim('c4','numeric','兩版同為4/5仍可錯不同題；先刪難題會改变分母，不能把其比例直接當作全檔同条件結果。','保留原逐題結果段',
 [e('s_audit','/paired_counterexample','本人兩個五題布林向量計數及刪首題變體')],['a_audit'],
 '示範相同摘要掩蓋不同錯題，以及改題集合改分母；並未假定任何實際模型錯題。','兩向量均4/5、錯題ID不同；刪難題成4/4','same_score=True；different_error_ids=True；full0.8 vs drop1.0','a=[T,T,T,T,F], b=[F,T,T,T,T]；逐題對應及sum/len。')
claim('c5','concept','validation可選設定，最後test保留至選定後作final evaluation。','驗證題與最後題一句',
 [e('s_cv','60-82','正式官方說明test調參洩漏與validation/final分工')],['official_s_cv'],
 '是避免test調參的評測方法，不等於本文件自行证明每次實驗完全無資料洩漏。')
claim('c6','empirical','連結的v2主線確實完成15個訓練階段，保有從隨機初始化與各階段自身前段selected best的承接記錄。','details第19章/v2已完成訓練與父權重說明',
 [e('s_audit','/training；各original raw/execution /status /returncode與train-receipt /steps /origin /stage_history','親核15份原completed執行与步數及parent SHA'),e('s_model','constructor133-150,188-201,217-231,247-262,295-302','新建所有神經模組'),e('s_base_model','32-41,54-66','LM基礎同樣新建'),e('s_contract','/random_init, /tokenizer_train_only','小型CPU四模組seed變體及train-only詞表')],
 ['a_audit','a_contract_audit']+raw_groups['training'],
 '核對既有production completion與lineage，不是重新訓練。MoE8段、Dense7段；native completed4000承接weighted selected3000，不是latest10000改objective resume；沒有據此判能力達標。',
 '15stage completed、returncode0、目標steps完成、test未選版','15/15符合；各stage steps1000/8000/1200/3000/1000/6000/10000，另MoEnative4000','audit按具名raw pointers核對，原件完整SHA/副本均保留；神經初始化只跑width16構造，未更新參數。',
 {'training_stages':15,'architectures':2,'moe_stages':8,'dense_stages':7,'production_seed':20261006,'stage_train_validation_counts':'audit-result.json /training逐stage列原train_records/validation_records；joint28876/2435','new_cpu_training_steps':0})
claim('c7','software','本機自訓入口以固定Git LFS資料逐段呼叫原trainer，可在CPU使用；不需要作者私人雲端儲存，GPU是另選的硬體路線。','details本機訓練指引連結',
 [e('s_wrapper','36-49,88-115,118-251','核對宣告檔、原source、管理argv、local subprocess輸出'),e('s_transport','229-300','固定LFS物件與bounded解包'),e('s_native_gate','680-854','native來源gate只查local files'),e('s_extras','27-30,52-74','CPU與CUDA不同extra')],
 ['a_audit','a_pointer','a_contract_audit']+raw_groups['manifest'],
 '完整training shell只查契約、未執行全量或GPU。固定archive67862431bytes、12records+8950assets=8962每檔都核對；cache6個notice/provenance檔失配已保留並改查原archive，未改教材或宣稱快取全新。',
 'LFS pointer及actual archive各SHA吻合、全8962檔吻合、四個training argv可解析','以上全符合；CPU環境torch2.14.1+cpu，CUDA不可用；source identity核對成功','git show固定08761...，verified archive原members；trainer parser補入wrapper管理的records/asset-dir僅解析。')
claim('c8','software','新stage接自己的best.pt重建optimizer與stage進度；同stage exact resume接自己的latest.pt恢復optimizer/RNG/sampler；public safetensors只是推論檔，trainer沒有safe training-init載入器。','details本機訓練路線所依賴的own checkpoint/公開權重邊界',
 [e('s_wrapper','52-85,169-182','own best/latest名稱、sidecar完整性、incomplete不能fresh init'),e('s_train','235-281,689-837,849-875','safe只導出model tensors；source torch.load及optimizer/sampler/RNG恢复，fresh zero計數'),e('s_native_gate','727-854','native同目錄genuine sidecars與selected SHA绑定')],
 ['a_audit','a_contract_audit'],
 '核對原碼與有界CPU拒絕分支/argv契約，沒有實跑完整fresh或exact resume。 .pt只適用自己可信輸出；optimizer狀態不存在於public safe。',
 'safe文件拒fresh、best拒resume、latest拒fresh；四個recipes對應fresh/resume','三種錯種檔名全部ValueError；四个documented argv正常解析；原main明確fresh0與restore branches','audit直接调用local_source早期錯種拒絕；親讀真正checkpoint及optimizer/RNG/sampler契約。')
claim('c9','empirical','公開CPU操作確實使用fixed HF979cdf... safe權重，可跑文句與真正工具往返；保存的八個chat及一次history append均有原argv/stdout/results。','details公開CPU操作連結',
 [e('s_audit','/archived_public_cpu','8chat+1append原始returncode及stdout SHA核對'),e('s_public','/commands','本人fixed公開text與tool CPU實跑'),e('s_link','/same_training_export_and_public_cpu','fetch結果model/manifest SHA等於completed native原receipt'),e('s_inference','101-136,215-328','匿名token=False四safe檔、驗證與同model生成')],
 ['a_audit','a_run_public','a_link_audit','a_text_stdout','a_tool_stdout','a_public_manifest']+raw_groups['cpu'],
 '已挑出的validation情境；本人只重跑2個，原8chat證據則查檔。成功僅驗安裝/權重/介面；不是test或未知題成功率，不重做heldout。',
 'fixed revision/manifests，CPU兩例產生原保存答案並正常EOS','text一次生成兩點回答；tool實算416後第二次生成結果是416，兩次EOS；與原answer相同','run-public.py保存每个真实argv/stdout/stderr与模型SHA；link-audit绑定模型26102d...manifestf27856...、selected1000/completed4000。',
 {'archived_chat_calls':8,'archived_history_appends':1,'new_cpu_calls':2,'new_cpu_generated_hops':3,'new_quality_test_questions':0,'max_new_tokens_per_hop':128,'cpu_threads':2})
claim('c10','empirical','v2保存两版各3734題的固定最后測試完成记录，其結果与局部示例分開。','details固定最後測試與能力/未達標頁連結',
 [e('s_audit','/archived_final_completion；原metrics /split /count /expected_count /evaluation_complete及receipt /status /completed_count','核對test completion與全題分母'),e('s_link','/same_manifest_and_frozen_final_test','MoE test原manifest等於公開safe及native export')],
 ['a_audit','a_link_audit']+raw_groups['final'],
 '完成執行与能力通過是兩件事。本節沒有逐項品質數字，故只查固定final已完成及與示例分開；未重評任何heldout或把成功示例當驗收。',
 'MoE/Dense各完整test3734且不是teacher-forced generation','兩版metricscount=expected3734/evaluation_complete true；receiptcomplete且3734；teacher_forcing_used_for_generation false','親核保存原metrics與completion receipt具名pointer及SHA；沒有新生成test回答。',
 {'architectures':2,'test_questions_each':3734,'test_questions_total':7468,'new_heldout_evaluations':0})
claim('c11','software','第20章是從成熟Qwen圖文與Whisper ASR延伸的另一條路，並非第19章自己的random-initialized checkpoint承接。','details成熟模型延伸段與重訓連結',
 [e('s_mature','load_core263-306；load_asr687-697','實際使用Qwen3VL和Whisper from_pretrained；可選LoRA訓練語言參數'),e('s_contract','/ast_contracts','本人AST確認兩載入函式from_pretrained'),e('s_model','295-302及constructor','主線使用新建LimitedAssistant各模組')],
 ['a_contract_audit'],
 '只核對起點/程式路線区别；未在CPU載入大型成熟模型或核對另一章品質、GPU時間數字，也沒有把Whisper能力算作本章从零成果。',
 '成熟路線明確pretrained載入，主線本地随机模块/own checkpoint','AST與原函数契約一致；LoRA為可選train分支，非另一random模型替換主線','親讀公開操作指引前180行作为待核主张；驗證以原load_core/load_asr函式及有界AST audit。')
claim('c12','software','歷史RAG、工具及推理報告保存原始問題、生成輸出和各自的資料/有效token分母。','details完整歷史結果與報告保存句',
 [e('s_history','/rows/*/sample_input_output、/split、/test_denominators、/pointers','本人先topkeys再取指定raw measurement pointers並完整保留原件SHA')],
 ['a_history_audit']+raw_groups['historical'],
 '僅驗report中保存的原样本与分母存在且可回查：RAG84樣本不是96題全檔coverage；tools20/20與reasoning所查budget24/24。沒有沿用原作者勝敗解釋，也沒有重跑GPU。',
 '三個报告均有原输入、原生成及split/test token分母','RAG84、tools20、reasoning24保存樣本；test records96/20/24；effective_tokens696/1289/51','history-audit只讀 /results/split、/test及具名samples；避开limits/review/notes等評語。')
claim('c13','concept','模型只生成工具請求文字還未完成操作；應保存實際工具結果，放回對話後再由模型生成回答。','details最後兩句工具紀錄',
 [e('s_function','markdown cells0,35 step3-4','OpenAI官方明確區分生成arguments、執行function、append結果及第二次API生成'),e('s_tool','122-170','本repo同樣真正執行calculator再generate'),e('s_public','tool_call實际CPU output /tool_trace及/generations/1/prompt_messages','實算416、tool-role结果与同core第二次生成')],
 ['official_s_function','a_run_public','a_tool_stdout'],
 '通用function-calling分工與本repo一次calculator hop；不推斷所有工具模型都可靠，也不宣稱固定final工具判準全達。')
claim('c14','concept','文字、圖解或排版修訂本身不會產生新的模型訓練證據；模型、資料或能力主張改變時，驗收應依改動範圍重新核對。','有限比較與相應驗收段',
 [e('s_git','DESCRIPTION22-28','snapshot記錄內容而非新實驗'),e('s_cv','60-82','選設定與final條件分開；變比較條件需相應評估')],['a_inspection'],
 '這是按影響範圍驗收的工作規則。純文字修改仍需核對文義/證據，若實際更動模型/資料/能力則不能以排版名義免驗；不要求重跑昂貴訓練来证明既有成熟方法。')

raw=(ROOT/'course/training.md').read_bytes();m=re.search(rb'^## T\.11 .*$',raw,re.M);n=re.search(rb'^## ',raw[m.end():],re.M)
section=raw[m.start():m.end()+n.start() if n else len(raw)]
assert sha(OUT/'section-original.md')==hashlib.sha256(section).hexdigest(),'T.11 changed; requires reviewer reinspection'
report={'schema_version':1,'review_stage':'technical','lesson_id':'T.11','source':'course/training.md#T.11',
        'reviewer_task':'/root/p6_fact_t11','reviewer_context':'fresh','source_sha256':hashlib.sha256(section).hexdigest(),
        'figure_sha256':{},'verdict':'pass','claims':claims,'sources':sources,'artifacts':artifacts,'issues':[],
        'checks':{
         'factual_accuracy':{'status':'pass','details':'14項逐項核對；完成訓練、公開推論、自己的完整續接與成熟底座延伸有不同直接證據。','claim_ids':[c['id'] for c in claims]},
         'numeric_verification':{'status':'pass','details':'人工byte/答對數與錯題/分母變體親算；既有15stage與8+1CPU/final分母只查原件，不冒稱重訓。','claim_ids':['c3','c4','c6','c9','c10']},
         'figure_consistency':{'status':'not_applicable','details':'T.11没有圖片或SVG引用；必要其他章上下文圖不是本節核對主張。','claim_ids':[]},
         'source_verification':{'status':'pass','details':'本人閱讀官方版本化原文、AST定位原碼与具名raw pointers；原件SHA与持久副本SHA已核對；未讀舊reader/technical或作者結果審閱摘要。','claim_ids':[c['id'] for c in claims]},
         'limitations':{'status':'pass','details':'CPU成功情境仅验接口；固定test执行完成不等于能力pass；safe不可exact resume；新stage须ownbest、resume须ownlatest；未下載資料、訓練、付費、上傳或heldout重評。','claim_ids':[c['id'] for c in claims]}},
        'inspection_artifact_id':'a_inspection','frozen_input':{'path':rel(OUT/'inputs/course/training.md'),'sha256':sha(OUT/'inputs/course/training.md'),'meaning':'first-read whole Markdown snapshot only; formal source_sha256 is raw section bytes'},
        'execution_outcomes':'原審查工具三次早期失敗已保留脚本和stderr；固定cache失配用原archive核對，兩次本人helper契約假設修正後audit exit0。這些不是教材錯誤或舊結果污染。'}
target=ROOT/'docs/technical-reviews/T.11.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(OUT/'T.11.initial-report.json').write_bytes(target.read_bytes())
print(json.dumps({'path':str(target),'verdict':report['verdict'],'claims':len(claims),'artifacts':len(artifacts),'source_sha256':report['source_sha256']},indent=2))
