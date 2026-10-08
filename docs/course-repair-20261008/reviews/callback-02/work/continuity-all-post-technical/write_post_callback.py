import json, hashlib, re
from pathlib import Path
from datetime import datetime, timezone

C=Path('docs/course-repair-20261008/reviews/callback-02')
B=Path('docs/course-repair-20261008/reviews/freeze-01')
W=C/'work/continuity-all-post-technical'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda: datetime.now(timezone.utc).isoformat()
M=json.loads((C/'manifest.json').read_text())
T=json.loads((C/'reports/technical-all-callback.json').read_text())
TS=json.loads((C/'reports/technical-all-callback.seal.json').read_text())
G=json.loads((W/'technical-gate-read.json').read_text())
assert sha(C/'reports/technical-all-callback.json')==TS['sha256']==G['technical_report_sha256']
assert sha(C/'reports/technical-all-callback.seal.json')==G['technical_seal_sha256']
assert TS['integrity_verified']['manifest_sha256']==sha(C/'manifest.json')==T['manifest_sha256']
assert datetime.fromisoformat(TS['at']) < datetime.fromisoformat(G['at'])

preserved={
 B/'reports/continuity-extensions-main-notes.json':'dad0c2790feafb5c9014f5f3232af1b07b946f3fb9306a0b37337d014d77e879',
 B/'reports/continuity-extensions-initial.json':'9927c7bee0eb70de327bad4537e2c5fed2e1f6693029a34eb2a92aa0abd47fe8',
 B/'reports/continuity-extensions-initial.seal.json':'813a75fdf9d165dc792228ed1afce495c1e938d082465d27bf4448ffd6f6d270',
 B/'reports/cross-extensions.json':'b9f03cbf77be711365bf69dab58c2d41512fddd3dab07fd4207a692abb0aad68',
 B/'reports/cross-extensions.seal.json':'e4b5894147c9e69acd30b6975e3917c493c082388e8832c23b3c9c692f2b907d',
 C/'work/continuity-all-callback/main-understanding.json':'95aa4fed28af3cb94e8ebd595520dfc236bdc1504fd6ac7e66fc0910f9b21fbd',
 C/'reports/continuity-all-callback.json':'6da22e3d714d085038f83f61d59299ed90f7a85cad1bbcdeff487d6fad1e9145',
 C/'reports/continuity-all-callback.seal.json':'538506ebdd6f6c59a006290b4c2114f4814759c45ef001f6b59cffa8654b10b0',
}
for p,h in preserved.items(): assert sha(p)==h

notes={
'2.5':{
 'quotes':['表中的平均代價將全部下一字與結束目標的代價合在一起平均；每篇短文提供的計分位置數可能不同。'],
 'understanding':'需要是讓區分圓／方的紅藍線索真的進入輸入。1、3字兩題輸入一樣；5字才包含不同線索。接續2.3的混合配方、2.4的非線性，窗口擴大改的是可見材料，不是已學會的證明。整篇先切分再取窗口，9／1／2篇各有資料用途；訓練與驗證差異限制這次比較。',
 'technical_facts_reconnected':'技術角色核_simple_examples逐字加結束0、全目標攤平交叉熵；原訓練103=94字+9結束，驗證11=10字+1結束。故表中9篇／1篇是材料範圍，平均代價按有效目標位置加總再除，不是每篇先平均後等權。圖中横軸1／3／5窗口、833／1345／1857參數不是更新時間；5字訓練較低而驗證較高，不能選3為普遍最佳。',
 'operation_and_burden':'唯一選讀保留width16、seed42、200步及固定比較入口；其最後2篇不拿來選窗口。新增一句只把原表計量單位說清，沒有新演算法、前置長訓練或效果承諾。',
 'necessary_transitions':['2.3','2.4','5.10'],
 'tracking':['callback-2.5-target-average']},
'5.4':{
 'quotes':['單靠這兩個梯度，還不能判哪個步子合適。','若縮小共同學習率才讓某些格不跨太大，另一些格卻幾乎不動，就會想依各格自己的歷史尺度調整步幅。'],
 'understanding':'目的先是降低代價又避免跨太大；共同學習率可能讓各格有過大／過小的取捨，才有按各格歷史尺度調步幅的需要。5.3先分當輪梯度、歷史方向和真正位移，這裡m累積方向、v累積平方尺度，再經偏差修正形成更新量。第一步1與100得到近同大小更新只是手算現象，裸梯度不能判斷哪格更合適。',
 'technical_facts_reconnected':'技術角色核官方Adam逐格m／v與修正式，未提供Adam勝過其他更新器的成效比較。當減去更新量時，負值可使參數增加；後續方向看mhat而非只看當輪g。機制因此接得回各格尺度的目的，結果仍由訓練／驗證判斷，沒有由兩個指定梯度推出品質。',
 'operation_and_burden':'完整主文的負梯度練習仍可沿同式追算。唯一選讀是另列長配方，短例不必先長訓。用途與限定的增句省去猜設計理由，沒有加入必修內容。',
 'necessary_transitions':['5.3'],
 'tracking':['callback-5.4-adam-use']},
'9.6':{
 'quotes':['混合版在拒絕題出現拒絕句3/3，正常題完整答案ID匹配13/14','這個拒絕判讀只搜固定「無法提供」，不是真實安全裁判。'],
 'understanding':'需要同時避開越界協助與擋正常請求。9.4先提供可信授權和有限場景，本節分應拒絕與應完成／澄清，再用各組的責任判讀。短例兩組各2題不是正式17題分母；一句好呀沒有拒絕也不等於完成，內容與EOS需另外看。',
 'technical_facts_reconnected':'技術角色核原evaluate_lm：raw內容ID完全相同與EOS是分開指標，拒絕另搜固定短句。正式17題分3拒絕與14正常，14包含澄清責任；因此3/3與13/14代表不同事件，不能相加成拒絕率或把未拒絕當正確。普通加法另卷，EOS全17也不保證內容正確。',
 'operation_and_burden':'唯一選讀完整列責任分組、內容匹配與停止，和主文新增的指標名称一致。只命名原有分子，沒有新增安全裁判、標註規則或一般安全能力宣稱。',
 'necessary_transitions':['9.4','7.9'],
 'tracking':['callback-9.6-explicit-metrics']},
'13.3':{
 'quotes':['13.6的兩個加法偏好訓練版本','這是重複抽樣的訓練曝光量，不是9,448個不同答案。'],
 'understanding':'需要比較同題完整候選，而非只比第一字。逐項條件機率相乘，取log後相加成整篇sum；平均每位置代價是另一個量。13.2已將chosen、rejected各自輸入／目標對齊，回答與EOS有效、提示忽略，後面不再shift。主文也保留長答案sum更負與sum不等品質的限制。',
 'technical_facts_reconnected':'技術角色核sequence_log_probability用logsoftmax、gather、mask後sum，render_chat含EOS；正式兩支是13.6 beta0.1／1、同加法起點各250步，兩側有效回答目標曝光均9448。新回鏈能定位兩支身份，EOS／問題／角色／PAD約定與sum一致，不把曝光當獨立答案。13.6的候選排序、相對改善、自由生成三項結果仍分開。',
 'operation_and_burden':'唯一選讀補既有兩支的名字與回鏈，必要回讀13.6全文及其唯一選讀可核身份與條件；不需要先完成250步才懂本節sum。未換計分約定，沒有增加效果承諾。',
 'necessary_transitions':['13.2','13.6','13.4','13.5'],
 'tracking':['callback-13.3-two-branches']},
'17.14':{
 'quotes':['hard','10道留出的顏色、形狀與音高屬性問答，按完整回答內容ID匹配判答對。','有梯度只說明可以學'],
 'understanding':'目的在更新時先體驗量化還原誤差；浮點主權重可調、前向臨時還原、反向STE近似、step才更新，訓練後打包是另一動作。直接round梯度0，STE只沿外層x傳梯度。hard先斷舊求導、複製同值、開新記錄，讓兩條路的梯度不混。17.7給逐列scale和浮點還原，17.9給固定權重PTQ，足以分清本節適應誤差與部署。',
 'technical_facts_reconnected':'技術角色核fake_quantize與新hard graph，短程式沒有step。正式比較同350步，PTQ4內容匹配6/10、QAT packed4 4/10；test69有效回答／EOS位置、validation36位置是NLL分母，與10／5題不是同一計數。圖流程不是本短程式真的訓練；部署logit差0只證明該QAT和其packed對上，不證明答對。選讀維持FP32主權重、正式逐列scale及保存／執行記憶體／速度的邊界。',
 'operation_and_burden':'完整唯一選讀的表、題例、350更新／39348曝光、5／10留出題與所有格式限制均重讀。新增方法作用與任務判準減少猜程式／猜分子，沒有新長訓練前置；QAT本次較差仍明示。',
 'necessary_transitions':['17.7','17.9','7.9'],
 'tracking':['callback-17.14-hard-autograd','callback-17.14-task-and-nll']},
'19.12':{
 'quotes':['回答左／右／上／下指定格位的商品類別。','兩版都沒有通過全部能力驗收'],
 'understanding':'需要依任務、判準和留出資料回答模型會什麼。19.3先說3734是對話記錄、同素材衍生題留同側；19.6給公开格位／區域输入、特徵和共同核心。位置行因此是回答指定格的商品類別，不能伸展為開放商品關係理解。主表八項3662題加選讀四類工具72題是同一3734，與控制卷／入口分類別分母不同。',
 'technical_facts_reconnected':'技術角色核prepare_vision_ocr兩種排法左／右／上／下、query_slot類別目標，1427/1440與1426/1440未變。已核數字只支持指定格位責任。CTC欄最高分後先合併鄰重複再去blank是入口全串檢查，324/324和共同核心252/324、281/324不等；19.6必要CTC選讀明列train加總路徑與greedy解碼差別。30錄音不是30人、固定猜120/360不是模型、回答改變不等改對，全部能力原判準未過仍保留。',
 'operation_and_burden':'完整主表、五組成對控制、唯一工具選讀重讀，原資料／CTC／未見字體／授權／20章上游的限制保持。新句收窄一列用途，沒有新任務或重跑模型前置；必要前文回鏈只用來理解此表，不要求讀完全部實作。',
 'necessary_transitions':['19.3','19.6'],
 'tracking':['callback-19.12-position-task']},
'20.10':{
 'quotes':['台換成臺需要一次替換，參考5字','若漏掉北，需要一次插入，分母仍是原文的5字'],
 'understanding':'20.9依請求核可見事實與語意，本節原樣抄寫還需要逐字判準。編輯操作方向是讓預測變原文；今天去台北的台要換臺，漏北則在預測插入北。CER分母固定參考五字，不能改為預測長度；意思相近不會自動通過轉寫。',
 'technical_facts_reconnected':'技術角色核Unicode codepoint距離、text_error_report(reference,prediction)用len(reference)=5、空參考CER=None。距離總數對稱不會使敘述方向也可任意互換；新台→臺與漏字例同方向。NFKC不繁簡轉换，模糊有字與無字圖需各自標註，不由此短字串例宣稱OCR能力。',
 'operation_and_burden':'全文無選讀。只修例句方向，保留原數值與規約；沒有新增前置概念、資料或操作。',
 'necessary_transitions':['20.9'],
 'tracking':['callback-20.10-edit-direction']},
 'training':{
 'quotes':['本練習的聯合路線選`partial`，目的是讓接頭與首末文字區塊一起適應圖片、聲音兩種輸入','temperature 0','每次只執行指定項，不自動補齊依賴'],
 'understanding':'全文T1–T11與九選讀按原順序分三段重讀。T1先定输入／答案／家族；T2通路检查無更新且null不當零；T3按篇切分再取窗口，loss時點與checkpoint格式不混；T4文字和回答目標／EOS／有效位置／驗證用途分開。T5風格、指令與安全有各自責任；T6入口留出2題全對才接續而非保存檔即通過；T7偏好較佳不等自由生成正確。T8同寬／同200步不等參數與工作量，T9模型張量bytes不等整檔／執行記憶體／速度，T10縮小架構與教師訊號用共同學生對照，T11留下共同起點、分母與逐題紀錄。',
 'technical_facts_reconnected':'技術角色核CLI partial先freeze全部、再開兩接頭和首末language blocks；固定vqa只開最後區塊，正文和選讀分清。新增目的說一起適應兩輸入，示範／效果留出核對不稱優於projector。infer.py傳入temperature，generate在<=0直接argmax；新CLI零值約定接回1.15 greedy而不是除0，前後固定選字才可比較。T10三教師各自model.pt／dataset.json真是蒸餾所讀產物，runner不自動補依賴；三summary對選T10者明示固定SFT／style／MoE必做，不能拿小練習或下載推論包替代。',
 'operation_and_burden':'九選讀均完整讀，包括格式／續訓／模態檔／LoRA／授權／效率後端／量化／多模態蒸餾及歷史勘誤。操作仍從專案根目錄，先W1環境，CUDA另選設備。新增一個用途句、一個CLI約定、三條件式summary補原需要；未選T10者不必跑教師三組，summary改善原CE-OPT-1仍屬可選澄清，不能因技術追蹤標必要操作而把原可選嚴重度追改。名稱補充舊低優先可選項留存，無新必要負擔。',
 'necessary_transitions':['first-steps','1.15','5.10','7.9','13.4','13.5','18.8','18.9','2.3','17.7'],
 'tracking':['callback-training-parent-outputs','callback-training-partial-use','callback-training-greedy']},
 'readme':{
 'quotes':['各版本的顯卡、驅動與套件配套選擇見[環境說明]','Colab操作另見[W.1]','再用 `python scripts/fetch_training_assets.py --asset NAME` 下載、核對並解包所選包'],
 'understanding':'完整108行從閱讀／Notebook、CPU安裝、最小流程、v2試用／從零重做、局部下載到維護全部重讀。各用途有不同門檻：閱讀不用安裝；短例CPU可做；正式GPU／20章配套各分路。19章v2自行訓練的有限任務與3734最後記錄、舊局部實驗、20章上游能力分開；推論輸出不當完整續訓狀態。',
 'technical_facts_reconnected':'技術角色核environment提供extra／顯卡／驅動，W1提供Colab先準備工具／順序執行；本輪也重讀兩者，鏡像錯承諾已刪而沒有新承諾。fetch list列名稱大小後return，asset NAME才選包核對／解包；正文兩步與T1例一致，NAME取清單名稱即可操作。Colab實機未驗、MPS僅基本CI且全課未驗仍明寫，不能由有路由當平台通過。',
 'operation_and_burden':'全文無選讀。導航按職責拆開、下載补實際下一动作，屬原承諾所需而非新增訓練前置；圖ID雙向查回只說識別對照不把編號當語意或能力。未要求先讀額外長環境教程才知道清單後下一步。',
 'necessary_transitions':['environment','first-steps'],
 'tracking':['callback-readme-link-target','callback-readme-list-vs-fetch']},
 'readme-en':{
 'quotes':['List available snapshots with `python scripts/fetch_training_assets.py --list`','then download, verify and unpack the chosen archive with `python scripts/fetch_training_assets.py --asset NAME`, replacing `NAME` with a listed asset name.'],
 'understanding':'完整91行重讀，同中文分清閱讀／短程式／正式訓練、19章有限自訓v2／舊實驗／20章成熟模型。CPU流程先安裝、啟用、kernel再Jupyter；推論模型下載只驗載入與執行，能力看留出評估，不能精確續訓。平台與Colab未驗限制沒有被新下載句抹去。',
 'technical_facts_reconnected':'技術角色核同一fetch_assets兩分支與英文句一致。list只列可選包，再asset NAME取所選包，NAME從列表替换；與中文及T1兩行可接，沒有靠讀者猜list會不會下載。固定v2資料題數與推論／從零重做不同路線仍保留原用途，不由此說本人完成下載／模型驗收。',
 'operation_and_burden':'全文無選讀／圖。兩步句補原操作缺口，沒有新演算法或強制額外閱讀；我是原extensions銜接角色的知情英頁回查，不冒稱另一名英語盲讀者。',
 'necessary_transitions':['environment','first-steps'],
 'tracking':['callback-readme-en-list-vs-fetch']}
}

report={
 'schema_version':1,
 'kind':'informed post-technical continuity callback; not a new blind source initial',
 'reviewer':'/root/repair_continuity_extensions',
 'role':'原extensions銜接／累積負擔角色，指定C十頁技術封存後的知情後置回查',
 'original_group':'extensions',
 'at':now(),
 'manifest_sha256':sha(C/'manifest.json'),
 'technical_seal_gate':{
  'report':'reports/technical-all-callback.json','report_sha256':sha(C/'reports/technical-all-callback.json'),
  'seal':'reports/technical-all-callback.seal.json','seal_sha256':sha(C/'reports/technical-all-callback.seal.json'),
  'sealed_at':TS['at'],'reviewer':TS['reviewer'],
  'own_actual_read_completed_at':G['at'],
  'own_actual_read_receipt':{'path':'work/continuity-all-post-technical/technical-gate-read.json','sha256':sha(W/'technical-gate-read.json')},
  'scope':G['technical_report_actual_read'],
  'distinction':'已實讀技術報告全部十頁／限制和111個registry中繼資料條目，不聲稱實讀其111份原始證據檔或重演既有執行。先確認report與seal SHA及C manifest一致，才開始此次完整来源重讀。'
 },
 'instruction':{'path':'/workspace/work/tutorial-repair-20261008/callback-evidence-instructions.md','sha256':sha(Path('/workspace/work/tutorial-repair-20261008/callback-evidence-instructions.md'))},
 'criteria_used_unchanged':{k:M['criteria_unchanged'][k] for k in ['SKILL.md','references/review-protocol.md','references/calibration.md']},
 'reading_order':{
  'technical_first':'技術報告及seal實讀完成後保存新gate；不回填舊C main理解時間。',
  'current_pages_after_gate':'完整來源按原順序讀主文及全部15選讀：七個短頁各全文；training1–200、201–390、391–568；readme108行、readme-en91行各全文。不是只看diff或沿用舊主文理解代替重讀。',
  'predecessors_after_gate':'20個列明必要前文／受影響過渡當前身份核對，main全文實讀；first-steps只重讀W1；13.6全文及唯一選讀，19.6 main及CTC第一選讀，其餘選讀／圖不泛稱新讀。八份前文從舊delivery重讀，另以current-source含完整同SHA snapshot確認未變，其餘12份新delivery實讀。',
  'figures_after_gate':'四SVG原文完整讀，既有同SHA640／360 PNG各親自再view；不是全頁互動。',
  'limits':'知情回查已知道B／C原結論；不是新獨立初判，不讀作者、舊audit問題或root教材裁定，不因發布授權預期通過。'
 },
 'necessary_predecessors':json.loads((W/'prerequisite-identities.json').read_text()),
 'preserved_original_bytes':{str(p):h for p,h in preserved.items()},
 'pages':[],
 'figures':[],
 'tracked_item_count':sum(len(n['tracking']) for n in notes.values()),
 'new_necessary_findings':[],
 'unresolved_necessary_findings':[],
 'remaining_optional_findings':[{'id':'extensions-training-names','necessity':'保留原低優先選讀改善；非本輪新問題，也不升為必要。training實際角色／操作／指標可連；名稱回鏈可使直接跳頁更方便。'}],
 'execution_model_platform_limits':{
  'own_source_read':'本次十頁全文、全部15選讀、列明必要前文與四圖實讀／視讀，SHA實際核對；內容理解是本人的後置判斷。',
  'technical_facts':'已讀技術報告所核實作／原資料／計量，均逐頁歸屬技術角色與原報告pointer。未親自新跑模型、训练、推論、安裝、下載、原資料重計、GPU或kernel；不能把技術報告讀取等同本人重演執行。',
  'ui':'本次未重新build、操作瀏覽器或新看整頁捕捉；來源／圖SHA未變，引用舊本人C callback的有限6張静態視讀與root實際兩尺寸捲動／橫捲。中央回頂nav.top移除的局部效果有root證據，右下TOC仍可能遮邊字；不是全部浮動UI、十頁完整真人互動或Colab通過。',
  'publication':'本報告完成指定十頁來源關係的技術後置回查，不代替root的正式版本檢查／發布門檻。技術報告中舊formal contract失敗是當時程序記錄，不在此追改或推定新正式檢查結果。'
 },
 'overall':'後置完整實讀將技術已核事實接回十頁原需要、用途、分母、轉寫方向、介面和前文。未辨認到未解必要銜接項或新增必要負擔；仍保留名稱可選改善及執行／平台／原位整頁限制。舊C主文理解早於技術seal是原並行callback歷史；本報告另存實際技術後置時間，不回填歷史。'
}

for i,p in enumerate(M['pages']):
 pid=p['page_id'];n=notes[pid];src=Path(p['snapshot']);text=src.read_text()
 assert sha(src)==p['source_sha256'];assert text.strip() in Path(p['source']).read_text()
 assert T['pages'][i]['page_id']==pid and T['pages'][i]['source_sha256']==p['source_sha256']
 quotes=[]
 for quote in n['quotes']:
  assert quote in text,(pid,quote)
  quotes.append({'line':text[:text.index(quote)].count('\n')+1,'quote':quote})
 for f,h in p['figures_sha256'].items():assert sha(Path(f))==h
 report['pages'].append({
  'page_id':pid,'snapshot':p['snapshot'],'source_sha256':sha(src),'source_identity_actual_checked':True,
  'figures_sha256':{f:sha(Path(f)) for f in p['figures_sha256']},
  'main_and_all_details_reread_after_technical_gate':True,
  'details_count':len(re.findall(r'<details\b[^>]*>.*?</details>',text,re.S)),
  'current_source_quotes':quotes,
  'own_current_understanding':n['understanding'],
  'technical_fact_to_need_use_denominator':n['technical_facts_reconnected'],
  'operation_prerequisites_and_added_burden':n['operation_and_burden'],
  'necessary_transitions':n['necessary_transitions'],
  'technical_evidence_reference':{'report':'reports/technical-all-callback.json','sha256':sha(C/'reports/technical-all-callback.json'),'json_pointer':f'/pages/{i}','original_tracking_ids':n['tracking'],'raw_evidence_execution_attribution':'技術角色實讀／原B執行的reuse範圍見該頁actually_checked_support和unknowns_limits；本角色未新執行。'},
  'judgment':'在完整目前來源及列明前文中，技術事實與原用途／責任可相接，未發現新必要負擔。'
 })

for stem in ['rewrite-02-visible-window','rewrite-17-qat-flow','rewrite-01-character-ids','window_training']:
 f=Path('course/figures')/(stem+'.svg')
 report['figures'].append({'path':str(f),'sha256':sha(f),'full_svg_source_reread_after_gate':True,
  'actual_png_views_after_gate':[{ 'path':str(B/'renders'/f'{stem}-{width}.png'),'sha256':sha(B/'renders'/f'{stem}-{width}.png'),'width':width} for width in [360,640]],
  'interpretation':{'rewrite-02-visible-window':'灰字未輸入，藍框1／3不分兩題，5才見紅／藍；看見不等學好。','rewrite-17-qat-flow':'浮點主權重→前向模擬還原→STE反向→step→另打包；短程式無step。','rewrite-01-character-ids':'上排貓看狗，映3／2／1／4，下排狗看貓。映1／2／3／0；雙向查回不代表字義大小。','window_training':'横軸窗口1／3／5與參數833／1345／1857，兩線按9篇／1篇材料的有效目標平均，不是時間曲線。'}[stem],
  'scope_limit':'孤立圖及同身份既有PNG再視讀；未新驗原位網頁互動。'})

report['counts']={'pages_fully_reread':10,'all_assigned_details_reread':15,'necessary_predecessor_units':20,'necessary_predecessor_details':2,'svg_sources_and_two_size_views':4,'technical_tracking_items_reconnected':14,'new_or_unresolved_necessary_findings':0}
assert len(report['pages'])==10 and sum(p['details_count'] for p in report['pages'])==15 and report['tracked_item_count']==14
for p,h in preserved.items():assert sha(p)==h
out=C/'reports/continuity-all-post-technical-callback.json';seal=C/'reports/continuity-all-post-technical-callback.seal.json'
assert not out.exists() and not seal.exists()
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
receipt={
 'schema_version':1,'kind':report['kind'],'reviewer':report['reviewer'],'role':report['role'],'at':now(),
 'path':str(out.relative_to(C)),'sha256':sha(out),'manifest_sha256':sha(C/'manifest.json'),
 'technical_report_sha256':sha(C/'reports/technical-all-callback.json'),'technical_seal_sha256':sha(C/'reports/technical-all-callback.seal.json'),
 'technical_sealed_at':TS['at'],'technical_gate_actual_read_at':G['at'],
 'all_ten_complete_source_rereads_after_gate':True,'old_C_and_B_reports_preserved':True,
 'pages':10,'assigned_details':15,'new_necessary_findings':0,'unresolved_necessary_findings':0,
 'scope':'知情技術後置局部continuity-all，來源關係／負擔；不是blind、模型執行、全頁互動或發布門檻代驗'
}
seal.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
for p,h in preserved.items():assert sha(p)==h
assert sha(C/'reports/technical-all-callback.json')==G['technical_report_sha256']
assert sha(C/'reports/technical-all-callback.seal.json')==G['technical_seal_sha256']
print(json.dumps({'report':str(out),'report_sha256':sha(out),'seal':str(seal),'seal_sha256':sha(seal),'at':receipt['at'],'counts':report['counts']},ensure_ascii=False,indent=2))
