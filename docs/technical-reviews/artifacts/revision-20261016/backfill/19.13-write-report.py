"""Serialize this reviewer's frozen-source findings; never edit course text or prior reviews."""
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
BASE = Path('docs/technical-reviews/artifacts/revision-20261016/backfill')
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads(Path(path).read_bytes())
run = read(BASE/'19.13-execution.json')
env = run['environment']

def src(id, path, title, note):
    return {'id': id, 'kind':'repository_code', 'title':title, 'path':path,
            'sha256':sha(path),'version':'Read working-tree bytes on 2026-10-16; exact file SHA binds this inspection',
            'verified':True,'inspection_note':note}
def ev(id, locator, supports):
    return {'source_id':id,'locator':locator,'supports':supports}
def verification(expected, observed, details, denominators=None):
    d={'method':'executed','expected':expected,'observed':observed,'details':details}
    if denominators: d['denominators']=denominators
    return d

def claim(id,kind,statement,location,evidence,scope,status='verified',v=None):
    d={'id':id,'kind':kind,'statement':statement,'location':location,'status':status,'evidence':evidence,
       'artifact_ids':['a_execution'],'scope':scope}
    if v:d['verification']=v
    return d
sources=[
 src('s_frozen','docs/course-revision-20261016/freeze-10/sources/19.13.md','freeze-15 指向的 19.13 原始小節快照','逐字讀取整個 19.13；lesson_slices(current chapter)[19.13] 與快照逐字相同。只以此小節作正文證據，未將整章視為已技術查證。'),
 src('s_model','tiny_perceptron/lesson_products/model.py','B 共享循環解題器與 C 候選評分器','讀 Tokenizer.encode、Block、LoopedSolver、CandidateScorer 和候選驗證：B 同一 shared Block 重用，19 類答案；C 每候選編碼 state/question/meaning，同一 scorer 輸出分數。'),
 src('s_inference','tiny_perceptron/lesson_products/inference.py','B／C 獨立權重載入、公開語法、judge 與 Auto','讀完整檔案，定位 Product.__init__/solve/judge、EFFORT 和 auto_solve。特別核對 judge 的完整回傳字典與 Auto 非 B 服務的 unsupported 分支。'),
 src('s_data','tiny_perceptron/lesson_products/data.py','有限世界、答案及 C 類型契約','讀 MEANINGS/QUESTIONS/CANDIDATES/REQUIRED、answer/render/judgment/c_rows 及 split/audit；三數值域0–6、兩盒覆寫交換與必要欄位規準均由本程式定義。'),
 src('s_train','tiny_perceptron/lesson_products/train.py','本課 B／C 隨機初始化與訓練收據產生','讀完整 train 函式；manual_seed 後直接建 LoopedSolver 或 CandidateScorer、AdamW 更新、只用 policy 選 checkpoint、選定權重輸出並保存初始/選定 digest。此處未啟動 train。'),
 src('s_training','tiny_perceptron/lesson_products/training.py','權重 digest 與分類訓練目標','讀 state_digest、encoded_data、logits_for、step、checkpoint/restore。C 用同一候選 scorer 的分組 cross entropy；B multi 直接監督1–5輪。未執行 step 或更新任何神經權重。'),
 src('s_cli','tiny_perceptron/lesson_products/__main__.py','B／C CLI 的實際子命令與選項','讀完整 main；solve/judge/auto/serve/download，只提供 b/c 權重參數及 b/c 下載產品，help 另親跑確認。'),
 src('s_delivery','tiny_perceptron/lesson_products/delivery.py','固定版本下載與共同 HTTP 的實際入口','讀完整檔案，download_product 固定40字HF revision與SHA驗證；serve 同時載入 B/C，health products=b/c；POST solve/judge/auto，沒有 A 選擇器。'),
 src('s_a_model','tiny_perceptron/selftrained/model.py','A 有限多模態模型的實際模組','核對 SelftrainedConfig、MaskedBlock、SelftrainedLanguageModel、VisionEncoder、OCREncoder、AudioEncoder、LimitedAssistant 建構與 _embeddings/forward 開頭；三類商品 head、12字 OCR 與三類音訊；未以此取代歷史訓練成果證據。'),
 src('s_a_metrics','docs/selftrained/results/public-raw/moe/test/metrics.json','A MoE v2 既有最後測試原始 metrics','核對完整 JSON schema 與 count/thresholds/per_task_final_reply/voice_scope、weight identities；全檔SHA等於公開結果所列 metrics_sha256。僅讀既有 primary receipt，未親跑 A 或重算其全部門檻。'),
 src('s_a_results','docs/selftrained/results/v2-final-public-results.json','A／舊 Dense 的固定公開結果與權重來源','核對 MoE/dense capacity、初始化、safe_weights、selected_source、final_test 指紋、capability_acceptance_passed、data_scope/comparison_limit；未讀其引用的其他審閱結論。'),
 src('s_b_runner','docs/lesson-products/results/initial-b-execution.json','B 實際 L4 訓練 runner 原始收據','核對 job.resume=null、status=completed 及完整 training 子物件與本機 training-receipt 逐結構相同；來源程式SHA與隨機初始化/選定步數記錄。這是既有親跑收據，不是本審閱親跑GPU。'),
 src('s_c_runner','docs/lesson-products/results/initial-c-execution.json','C 實際 L4 訓練 runner 原始收據','核對 job.resume=null、status=completed 及完整 training 子物件與本機 training-receipt 逐結構相同；來源程式SHA與隨機初始化/選定步數記錄。這是既有親跑收據，不是本審閱親跑GPU。'),
 src('s_final','docs/lesson-products/results/final-report-v1.json','B／C 固定最後結果與逐題分數綁定','核對 b/c 的 count/correct、gates、scope、predictions 與 settings/data SHA；親自串流既有逐題檔重算核心 test 的每輪 B 與三類 C count/correct 並比對，沒有重啟正式推論。CE、Brier、延遲與機制對照不在本節主張範圍，未重新計算。'),
 src('s_settings','docs/lesson-products/results/settings-freeze-v1.json','B／C 正式測試前固定產品與設定','核對 products 全部指定檔SHA與本機相同、protocol、data manifest、calibration 與 test_outcomes_observed=false；不以freeze檔本身證明未見test，另有 pretest/started 時間收據。'),
 src('s_public_cpu','docs/lesson-products/results/public-cpu-acceptance-v1.json','既有乾淨 CPU 匿名下載與公開命令原始收據','讀 status/environment/auth/release、cli_runs download/solve/judge/auto 及 HTTP health/requests；B/C固定HF revision與release.json相同。這是既有實際操作收據，本審閱未重新連網下載。'),
 src('s_release','tiny_perceptron/lesson_products/release.json','B／C 固定 HF revision 與產品 manifest 入口','核對 repo、40字 revision、B/C prefix 與 manifest SHA；固定下載入口的本地身分，不獨立保證未來服務可用。'),
 {'id':'s_jev','kind':'official_docs','title':'TypeSafe Introduction — state、typed questions 與 Choice/Score/Noul 契約',
  'url':'https://docs.typesafe.ai/introduction','version':'live HTML retrieved 2026-10-16T03:25:43.489344+00:00; SHA-256 75746b82d1678fc99948b5efdaf9ec43e5a11f0837a8105998f4172375a3a018',
  'verified':True,'checked_original':True,'accessed_on':'2026-10-16',
  'authority_reason':'TypeSafe 為 Jev 開發與服務提供者；其官方文件是公開輸入輸出契約的一手來源。',
  'inspection_note':'由原始 cached HTML 解析 main 並實際讀全文，核對首段 structured values/probability distributions、TypeSafe primitives表與 Atomic questions。僅支持服務型態類比，不將商用速度、內部架構或訓練方法移植為本課主張；小摘錄/原檔SHA見 a_official。'},
 {'id':'s_execution','kind':'execution','title':'19.13 審閱者有限 CPU 查核與既有原始分數重算','verified':True,'artifact_id':'a_execution'},
 {'id':'s_assignment','kind':'derivation','title':'覆寫事件的透明推導','verified':True,'details':'以同一被查詢狀態s為對象：s0=2；設成5是賦值而非加法，因此s1=5；再設成1得到s2=1。採用兩盒實作的等價完整輸入：A=2、B=6、事件A設5,A設1、查詢A；B沒有被查詢，不改變此答案。這只驗此具體覆寫例，不能推出普遍解題準確率。'}
]
claims=[
 claim('c1','empirical','A 是沿用既有 MoE v2 權重與固定有限多模態結果的助理，仍未通過全部原定能力判準。','19.13 第2段／course/chapters/19.md:18',
       [ev('s_a_results','architectures.moe.capacity/config/selected_source/safe_weights/final_test/capability_acceptance_passed','固定MoE架構、權重與歷史驗收false。'),ev('s_a_metrics','count/thresholds/per_task_final_reply.voice_qa、tool_call','原始記錄語音42/90、tool_roundtrip 0/276，仍低於原門檻。')],
       '既有版本、有限文字/圖片/字卡/語音/工具考卷；未重跑A，未聲稱任意照片或所有語音皆可用。Dense沿用為歷史對照的產品分類屬本節導覽。',
       v=verification('既有原始結果身份與公開固定結果吻合，且不是全部能力門檻通過。','原始metrics SHA吻合；3734題；voice_qa 42/90、tool_roundtrip 0/276；capability_acceptance_passed=false。','親跑CPU只讀JSON比對SHA/欄位；生成數字來自既有主實驗收據。',{'test_rows':3734,'voice_qa_rows':90,'tool_call_rows':276,'primary_run':'gha-37448172517-1'})),
 claim('c2','software','A 的圖片/位置回答與 OCR/語音入口限於其有限類別；接受圖片不能推出辨識任意物件。','19.13 表格A列及末段／行26、34',
       [ev('s_a_model','VisionEncoder.head、OCREncoder.head、AudioEncoder.head、LimitedAssistant','實際有限多模態構造。'),ev('s_a_results','data_scope','3商品類、12 OCR字、3語音intent的已驗資料邊界。')],
       '此處核對的是能力邊界聲明；不是證明模型無法表示任何其他特徵，也沒有新做圖片品質實驗。',
       v=verification('本節限定應與原始配置/資料範圍一致。','公開data_scope: vision_classes=3、ocr_glyph_count=12、voice_intents=address/app/card，與模組head符合。','親跑讀取現有data_scope並對照已讀構造；本輪未生成A答案。')),
 claim('c3','software','B 是獨立 Looped Dense 模型，同一共享區塊可按輪數重用後讀出答案類別。','19.13 第3段／行20',
       [ev('s_model','LoopedSolver.__init__/forward lines117–141','單一shared Block、19類readout與loops1–6。'),ev('s_inference','Product.__init__/solve lines73–116','B直接載入自己的model.pt，solve回answer/loops。')],
       '已公開有限兩盒/三數語法；不是一般聊天，也不是將A/MoE答案換介面。更多輪不構成單調品質保證。',
       v=verification('B自行載入；指定5輪即呼叫同一shared Block5次，回答案與實際輪數。','獨立載入LoopedSolver；5輪hook記到5次；例題回answer=1/loops=5。','親跑CPU固定權重推論與forward hook；未重新訓練。')),
 claim('c4','numeric','同一狀態從2依序覆寫為5、1，最後狀態類別應為1。','19.13 第3段與表格B列／行20、27',
       [ev('s_assignment','s0=2 -> s1=5 -> s2=1','透明賦值推導。'),ev('s_data','answer(world)覆寫分支','與正式資料答案規則一致。')],
       '語意最小示意，並非承諾該自然語句可逐字送CLI；實跑先轉成公開兩盒完整語法。僅一題不證一般成功率。',
       v={'method':'executed','expected':'最後狀態=1，輸出記錄指定實際輪數。','observed':'answer(world)=1；同等完整兩盒輸入在1/3/5輪皆answer=1，loops欄各為1/3/5。','tolerance':'整數與輪數精確相等。','details':'A=2，B=6，事件A設5,A設1，查詢A；同一A兩次覆寫。自然語示例本身被parse_state拒絕，故未當作literal API命令。'}),
 claim('c5','concept','Jev 式在此指狀態與有型別問題的機率判斷服務類比，不是已公開或被複製的商用神經訓練配方。','19.13 第4段／行22',
       [ev('s_jev','Introduction 首段與 TypeSafe primitives table','官方state/typed-question structured probability契約，Choice/Score/Noul三類。'),ev('s_model','CandidateScorer與Tokenizer.encode','本課可見自製scorer結構。'),ev('s_train','train無resume建模分支','本課可見的自製訓練入口。')],
       '只類比公開輸出用途；沒有採用商用可靠度、延遲或校準保證，沒有以介面反推Jev未公開內部架構。'),
 claim('c6','software','C 獨立載入自己的權重，以狀態、問題與候選含義評分，回服務選擇、資訊充分或有限等級分布。','19.13 第4段與末段選產品練習／行22、34',
       [ev('s_model','Tokenizer.encode、CandidateScorer.forward','每候選吃state/question/meaning。'),ev('s_data','QUESTIONS/CANDIDATES/REQUIRED/judgment','三種有限問題與欄位標籤。'),ev('s_inference','Product.__init__/judge','獨立C權重且judge不需B。')],
       '固定字表、有限語法及預先列出的含義ID；機率僅在候選集合內，不能當任意客服文字理解或普遍正確機率。',
       v=verification('C可在沒有B依賴的judge路徑回三種問題機率。','C載入CandidateScorer 84545參數；卡片缺申請編號：choice=card、info=no、rubric=grade0。','各呼叫Product(c).judge；沒有呼叫B或工具/外部模型；本次同程序還載B供分開的Auto查核，不是C模型本身需要B。')),
 claim('c7','empirical','B 與 C 已由隨機權重實際完成更新並選出可載入權重，屬本專案自訓模型。','19.13 第3／4／6／7段／行20、22、30、32',
       [ev('s_b_runner','status/job.resume/training','B真實L4收據。'),ev('s_c_runner','status/job.resume/training','C真實L4收據。'),ev('s_train','manual_seed、隨機建模分支、保存收據','隨機初始化到更新後選定權重的程式路徑。'),ev('s_training','state_digest','重新比對tensor狀態digest。')],
       'GPU更新為既有primary receipts；審閱只在CPU重建隨機初值、比對選定權重及best checkpoint，不重訓也不驗精確續訓。完成訓練不等於全部題型通過，B算術仍失敗。',
       v=verification('隨機seed原始digest與已存紀錄吻合，選定權重不同且與best一致，runner已完成。','B seed7421/1000步、C seed9041/800步；兩者初始digest重建吻合，selected digest不同且等於收據/best；全部frozen product檔SHA吻合。','親跑CPU重建模型和digest；與L4完成收據training子物件逐結構比對。沒有執行 optimizer.step。',{'b_steps':1000,'c_steps':800,'b_seed':7421,'c_seed':9041,'gpu_runs':2,'b_parameters':170003,'c_parameters':84545})),
 claim('c8','software','本輪 effort 只映射 B 的輪數；Auto 是 C 兩次判斷與 B 解題的組合流程，不新增第四個神經模型。','19.13 第6段／行30',
       [ev('s_inference','EFFORT與auto_solve lines164–195','low/medium/high=1/3/5；先C.choice/info，合格才B.solve。'),ev('s_cli','main子命令及--effort','公開Auto入口與模型入口分開。')],
       '固定產品和規則版本；A未接入此effort參數。loop工作量不是實測加速或品質保證。',
       v=verification('medium對應B三輪，Auto只使用既有B/C且資訊不足可不叫B。','完整例medium: requested_loops=b_loops=3、answer=1；缺A初值例status=clarification_needed、b_loops=0；無額外神經權重載入。','親跑CPU auto_solve兩例，對照程式分支與既有產品模型種類。')),
 claim('c9','empirical','B/C 本輪有已凍結的實際留出題結果，結果同時保留通過與失敗。','19.13 第7段／行32',
       [ev('s_final','b/c/gates/predictions/scope','既有正式test逐題分數與通過失敗旗標。'),ev('s_settings','products/protocol/test_outcomes_observed','正式設定身分與有限任務範圍。')],
       '只證明被列出的有限固定語法考卷；不由C跨家族同標籤重複觀測推論全新文字泛化。未重算CE/Brier/延遲與機制對照，也未重啟正式評估。19.15–19.20的各細節仍由各節審閱負責。',
       v=verification('既有逐題檔SHA與報告一致，重算核心B每輪/C三類的count/correct與報告相符，保留失敗。','原始分數SHA吻合；B五輪boxes2928/2956，sum2/42；C choice11797/11797、info/rubric各10890/10890；gates保留b_secondary_target=false。','親跑串流讀既有predictions.jsonl，僅取split=test並逐row比較預測/gold後計數；未重新生成預測。',{'b_boxes_rows':2956,'b_boxes_families':1478,'b_sum_rows':42,'b_sum_families':10,'c_choice_rows':11797,'c_info_rows':10890,'c_rubric_rows':10890,'reported_b_depths':6})),
 claim('c10','software','表格將 C 的可檢查輸出列為 yes/no 機率，以及澄清建議與其依據。','19.13 表格C列／行28',
       [ev('s_inference','Product.judge完整return字典 lines140–149；auto_solve lines174–177','judge不回澄清建議或具體缺欄依據；Auto在非B任務先unsupported。'),ev('s_delivery','POST /judge 直接轉發c.judge','沒有外層補成澄清/依據輸出。')],
       '指出已交付C輸出契約不符；不否認外層程式能讀no後人工訂策略。meanings[no]只是一段固定類別說明，不是定位申請編號缺失的實際依據。',status='contradicted',
       v=verification('按表格，要可觀察到C的澄清建議與該建議的具體依據。','卡片缺編號judge只回question/selected/probabilities/meanings/temperature/concentration/concentration_scope/seconds；無建議/依據欄。既有Auto對同卡片回unsupported，不是澄清建議。','親跑固定C info/choice/rubric與Auto，逐鍵檢查回傳；原始JSON封存。')),
 claim('c11','software','共同介面會讓人選擇 A、B、C 成品。','19.13 第6段／行30',
       [ev('s_cli','main子命令與download --product b/c','可見共同CLI只涵蓋B/C/Auto。'),ev('s_delivery','serve/health/POST route','HTTP產品清單只有b/c；只有solve/judge/auto。'),ev('s_public_cpu','http_health與cli_runs','現有公開操作實錄也只呈現B/C/Auto入口。')],
       '此句是將三成品共同界面作交付預告；目前必要直接來源不能定位到含A的共同成品選擇器。若只是用途對照表或分開入口，需改文使讀者能對上現有交付。未以搜尋不到推論全repository/外部服務永遠不存在這功能。',status='unresolved',
       v=verification('應找到可供選A/B/C的共同入口或可重做的交付證據。','共同CLI help只有solve/judge/auto/serve/download；Product只接受b/c；HTTP health products=[b,c]；未定位三成品選擇器。','親跑CLI help並讀完整CLI/HTTP來源、既有公開CPU操作收據；沒有讀作者plan或其他審閱。')),
 claim('c12','software','讀者可直接使用既有固定權重推論，無須為閱讀重訓 B/C；固定下載入口已交付。','19.13 第7段／行32',
       [ev('s_release','repo/revision/products','B/C固定下載revision與產品manifest。'),ev('s_delivery','download_product','固定版本下載並比對SHA/byte大小。'),ev('s_public_cpu','cli_runs 前兩筆download與後續推論','既有匿名下載與無重訓推論實際收據。')],
       '本審閱親跑本機固定B/C推論；匿名下載為既有primary receipt，未重新網路下載，也未檢查GPU/MPS平台或未來服務可用性。A沿用已有公開權重的主張由c1的固定原始結果定位。',
       v=verification('現有權重可獨立推論且固定revision下載入口與既有公開操作一致。','本機B/C manifest全檔驗證後可推論；現有CPU收據匿名download b/c均exit0、5檔驗證，revision與release.json相同。','親跑本機Product讀取/推論；讀既有乾淨CPU下載收據，不把它寫成審閱者新下載。'))
]
artifacts=[
 {'id':'a_execution','kind':'execution','path':str(BASE/'19.13-execution.json'),'sha256':sha(BASE/'19.13-execution.json'),
  'description':'有限CPU實跑輸出、B/C初始與選定tensor身份、C反例、CLI、Auto及既有test原始逐題重算；另保留A既有原始metrics比對。',
  'command':run['command'],'result':'完成；權重/receipt/核心逐題重算一致，B示例=1，C僅回no機率而非澄清建議/依據，現有共同入口未含A。','environment':env},
 {'id':'a_probe','kind':'code','path':str(BASE/'19.13-probe.py'),'sha256':sha(BASE/'19.13-probe.py'),'description':'本審閱只讀CPU查核程式；不呼叫train/optimizer或正式test入口。初次整理時曾因未先排除shift row與相對路徑處理出錯；修好分析程式後成功輸出，沒有改變模型或評測。'},
 {'id':'a_official','kind':'source_snapshot','path':str(BASE/'19.13-official-source-receipt.json'),'sha256':sha(BASE/'19.13-official-source-receipt.json'),'description':'實際讀取官方TypeSafe原HTML的小摘錄、URL/版本/SHA與支持範圍；全文留Git外。'}
]
report={'schema_version':1,'review_stage':'technical','lesson_id':'19.13','source':'course/chapters/19.md#19.13',
        'reviewer_task':'/root/backfill_tech_19_13','reviewer_context':'fresh','author_tasks':['/root'],
        'source_sha256':run['source']['slice_sha256'],'figure_sha256':{},'verdict':'revise',
        'claims':claims,'sources':sources,'artifacts':artifacts,
        'issues':[
            {'id':'19.13-I1','claim_id':'c10','status':'unresolved','necessity':'required','details':'C表格的「建議澄清與其依據」超過已交付judge輸出。親跑卡片缺申請編號只有no機率和固定meaning；無具體缺欄依據或建議。Auto也未替卡片客服流程產生澄清，會先回unsupported。','requested_change':'最小修正文案為C回yes/no機率，外層依no與已定欄位契約決定澄清；若保留模型/服務會輸出建議及依據的宣稱，需提供對應已交付入口與原始證據。'},
            {'id':'19.13-I2','claim_id':'c11','status':'unresolved','necessity':'required','details':'「共同介面會讓人選成品」在三成品段落預告共同A/B/C入口，但現有共同CLI/HTTP與公開CPU操作只涵蓋B/C/Auto；A仍需走既有獨立入口。缺少三成品共同選擇器的可查證交付。','requested_change':'若目前交付是三個獨立入口，將句子改成可定位的獨立選用/入口說明；若確實已交付共同A/B/C選擇器，補具體入口並由原owner核對。'}],
        'checks':{
            'factual_accuracy':{'status':'revise','details':'A既有有限模型、B/C獨立隨機訓練及Auto組合均有直接證據；C建議/依據與三成品共同界面兩項仍不符合或不足。','claim_ids':['c1','c2','c3','c5','c6','c7','c8','c10','c11','c12']},
            'numeric_verification':{'status':'pass','details':'覆寫2→5→1手推與親跑等價完整輸入均為1。既有B核心每輪/C三類逐題count/correct重算和報告吻合；B算術五輪只有2/42，未誤寫為成功。GPU訓練步數/seed/權重digest已核對，沒有重訓。','claim_ids':['c1','c4','c7','c9']},
            'figure_consistency':{'status':'not_applicable','details':'freeze15的19.13原文與current lesson_slices均無SVG、圖片或圖示引用，figure map為空。本次未做網站版面驗收。','claim_ids':[]},
            'source_verification':{'status':'revise','details':'已獨立讀TypeSafe原HTML、程式、原始runner/metrics與逐題分數，來源SHA及scope已保存；共同A/B/C介面未找到直接可核實來源，C現有完整return字典反駁表格額外輸出。','claim_ids':['c5','c10','c11']},
            'limitations':{'status':'revise','details':'固定有限語法、A有限類別、隨機訓練不等於能力全通過、effort非品質保證等邊界有依據；但表格把外層可做決策寫成C已交付輸出，且共同入口預告未對上交付，需收窄。','claim_ids':['c1','c2','c3','c4','c6','c7','c8','c9','c10','c11','c12']}
        },
        'review_scope_note':'僅完整審閱freeze15指定19.13（snapshot沿用freeze10）。19.1–19.12與第9章只按必要直接來源定位其有限角色/契約，沒有替其他小節做全面技術驗收。未讀作者plan/處置或任何其他/舊review結論。沒有訓練、調參、formal-test重啟、Modal/HF寫入或教材修改。'}
path=Path('docs/course-revision-20261016/reviews/technical-19.13-backfill-original-01.json')
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(path)
