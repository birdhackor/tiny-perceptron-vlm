import json,hashlib,datetime,pathlib
root=pathlib.Path('.')
out=root/'docs/course-revision-20261029-omni/reviews/reader-products-v1'
out.mkdir(parents=True,exist_ok=True)
plan_path='docs/course-revision-20261029-omni/reviews/reader-products-plan-v1.json'
freeze_path='docs/course-revision-20261029-omni/reviews/wave-01/source-freeze.json'
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
plan=json.loads(pathlib.Path(plan_path).read_text())
freeze=json.loads(pathlib.Path(freeze_path).read_text())
frozen={p['page_id']:p for p in freeze['pages']}
sessions={
 'chapter-19':'7d18cefcdd194b3aa0c73e692dee0dd5',
 '19.13':'5870edfd274144d3a52250353acb12cc',
 '19.16':'dea5b41fd4e649dab7557f38780322d3',
 '19.17':'ac1e7bc9531848bda3da0c5e1cb65800',
 '19.19':'f24fd6eb1a9f4a719282bce5bcde9b05',
 '19.20':'57b8068adf494354ba52f994a4294d6c',
 '19.21':'6b93eebfcd35457b8221b6119727cd09',
 '19.22':'c03babd9357f4475afbfe31f2ed14577',
 '19.23':'3189393bc1b846c6863b55fb66fe49a5',
 '19.24':'4f6161ea4d8e46bc9d2a65d53bf5e563'}
# 以下逐頁摘要、分級、估時均為讀者已完成52個當時checkpoint後的親寫收束。
meta={
 'chapter-19':{
  'verdict':'pass','summary':'章入口已清楚分MoE素材到文字、Looped固定語法到數值、C條件候選到機率及Auto服務。V2與新版擴充、CPU契約與正式能力分開。導覽不須在此推導各網路。',
  'minutes':[3,5],'reason':'plan必要前文已讀後，閱讀兩段路線與三產品角色約3–5分鐘；不含46頁context補讀。',
  'code':'本頁無實作承諾；感知入口與接頭的最低角色已說，細節合理待各節。',
  'visual':'無圖。文字足以理解角色、章節路線及版本；未看完整網站。'},
 '19.13':{
  'verdict':'pass','summary':'三產品按客服工作需求分工。A設5再設1查A=1；C缺申請編號是資訊問題而非聊天。Auto非第四神經模型，檔位只對應Looped輪數，舊成績不替新版作證。',
  'minutes':[12,20],'reason':'已讀必要前文後，辨三種輸入輸出、手算覆寫例與成品選擇練習約12–20分鐘。',
  'code':'表格和固定語法例已足以教用途；完整B輸入語法精確回19.15（已補讀），沒有本頁必需新helper。',
  'visual':'無圖。三列最小輸入/輸出對應清楚，未見必要缺圖。'},
 '19.16':{
  'verdict':'需補說明','summary':'可以解三型共享候選評分、獨立列保重排、集中度不等可靠度、校準在新措辭失敗與天氣高信心錯例。必要批次/回填程式卻只外連，新helper與關鍵列對應未在教材攤開。',
  'minutes':[35,55],'reason':'讀兩段程式、逐候選對照及校準/失敗判讀約35–55分鐘；必要helper缺口另待補，估時不表示多花時間即可無外查理解。',
  'code':'CandidateScorer完整可追一列到一分；candidate_batch/grouped_scores首次被直接呼叫，正文只給角色與外部原始碼連結，仍缺必要編碼/切片/回填操作。',
  'visual':'無圖；獨立列概念已有文字，不要求另畫圖；具體一題的列/切片/分數對應可由短程式或表補。'},
 '19.17':{
  'verdict':'pass','summary':'C兩次argmax先門控，unsupported先於clarification；固定檔查表/auto依事件數規則，requested再min硬上限，裁減回budget_limited。policy選法與測試省84輪少對1題已回應成本目標，無Auto勝medium宣稱。',
  'minutes':[30,50],'reason':'追控制流程、手算1–6事件及cap、比較2998題工作/品質與路由標籤約30–50分鐘；不含已讀模型前文。',
  'code':'rule_depth完整、policy與EFFORT明列，硬上限與停止分支的必要運算可直接追；未把短規則例當實跑兩模型。',
  'visual':'無圖；三步條列、函式與具體policy足以追路徑。'},
 '19.19':{
  'verdict':'需補說明','summary':'兩層release/manifest的目的、下載/本地/續訓包差別及三CLI介面可以理解。標題要求具體核對版本，但B/C固定身份、manifest SHA與實際輸出核對位置導到外頁，教材內無一個完整版本查核例。',
  'minutes':[18,30],'reason':'在既有環境與前文已讀下，閱讀命令、手算2/no/3輪與版本包裹比較約18–30分鐘；不含安裝、網路等待與外跳查核，必要缺口仍待補。',
  'code':'CLI是重跑入口可接受；本頁給世界正解但未列這些命令的actual版本JSON/固定artifact身份，不能只以預設下載或正常完成核對版本。',
  'visual':'無圖。問題是少具體核對材料，並非需要裝飾性包裹圖。'},
 '19.20':{
  'verdict':'pass','summary':'事前門檻、固定後測試、主要/次要/子群限制及交付證據鏈可分開理解。用已讀天氣例能把錯誤定位C支援判斷；舊失敗集不冒新final。',
  'minutes':[15,25],'reason':'比較門檻表、四段證據與用既有失敗例作分層約15–25分鐘；不含外部原始結果重算。',
  'code':'摘要頁只用既有結果與原則，不要求新計算helper；練習已有19.16天氣錯例可用，無須外跳才能完成。',
  'visual':'表格已給分母與判準；無必要缺圖。'},
 '19.21':{
  'verdict':'需補說明','summary':'可追新A模組擴充、局部支持、物理時間與序列因果、additive/time-RoPE公平條件、兩層融合與唯一semantic紀錄。CPU亂數只驗通路/遮罩/梯度且正式門檻未測。新的影格CNN池化最低行為未教，完整參考程式閱讀仍有必要負擔。',
  'minutes':[70,110],'reason':'已讀12.17/14.19/4.5後，逐行讀約150行網路與驗證、追25位置和三mode、對圖/門檻/練習約70–110分鐘；CNN缺口須補，非保證能自行耗時跨過。',
  'code':'時間/遮罩/fusion/head/record必要程式全部內嵌，12.17前處理是精確已讀回指；Conv2d與AvgPool/AdaptiveAvgPool首次組合卻缺通道/空間縮減最低解釋或完整先備回指。',
  'visual':'已實渲染凍結omni-19.21-av-module。左影格CNN/右200ms音塊投影匯合時間+mask，經fusion/head到record，再分對話/Talker；原圖對應可追。完整網頁未看。'},
 '19.22':{
  'verdict':'pass','summary':'完整內嵌固定codec、GRU Talker、teacher forcing一次更新、自接argmax、EOS/上限、WAV與兩路記帳。輸出兩路僅100ms880Hz，明確失敗；三種正式驗收與分母可解。P4未定義屬可選用語改善。',
  'minutes':[85,130],'reason':'以已讀碼本/STE/因果decoder為背景，追約200行訓練/生成/WAV、手算碼和樣本數、三種評分與分組約85–130分鐘；不含正式訓練、播放器或聽測。',
  'code':'所有必要新helper在頁內，輸入/targets/teacher forcing/actual codes/800樣本皆給，未用參考碼回放冒自由生成。',
  'visual':'已實看凍結omni-19.22-sound-chain。Asemantic→字典→Talker→decoder→分析器，gold從底只入評分，沒有覆寫生成通路；網站與播放未驗。'},
 '19.23':{
  'verdict':'pass','summary':'byte入口、格式/語義限制、state-only M、獨立候選交叉讀取、對稱none、ID平手與等家族/型別溫度全部直接可追。能解重排與加候選scores/p差別、共享等價與Q1再問；正式語義/可靠決策未測。',
  'minutes':[75,120],'reason':'已讀16.15/9.17/9.18後，逐行追UTF8/校驗/注意力/對稱摘要與溫度函式，核對輸出及接受分母約75–120分鐘；不含正式訓練/性能量測。',
  'code':'utf8/validate/TinyDecision/selected/fit_temperature與fixture完整；必要接受核心精確回9.18已讀程式。來源CONTRACTS未開，理解不依賴它。',
  'visual':'本頁無圖；函式邊界、返回中間量及對應文字足夠追M到候選/none，沒有必要缺圖。'},
 '19.24':{
  'verdict':'pass','summary':'適配器不代算，手動入口優先但不繞B值域檢查；C presence不等影片內容。gold入口/真C入口同family三欄計數能分錯路與後端錯，3/4、2/4、1/4有明確人工示意標記；正式整合/回歸待做。',
  'minutes':[35,55],'reason':'追adapter/路由兩函式、四列AND計數、手動與auto小變化、整合100家族和舊84家族限制約35–55分鐘；不含真模型整鏈執行。',
  'code':'adapt_sum/select_route與人工rows計數完整，必要input/輸出都在頁；沒有把fixture0.96或返回入口描述當真正C/A/B推論。',
  'visual':'無圖；程式與布林表能定位路由/adapter/backend，無必要缺圖。'}
}
issues={
 '19.16':[{
  'id':'products-19.16-helper-inline','necessary':True,'severity':'增加理解負擔',
  'location':'19.16 unit 2–3（首個候選重排程式與其解說）；unit 6節末仍未補',
  'source_quote':'tokens, slices, max_candidates, _ = candidate_batch([row], tokenizer, model.config)\nprobabilities = grouped_scores(model(tokens), slices, max_candidates).softmax(-1)[0]\n兩者的原始碼見 [model.py](../../tiny_perceptron/lesson_products/model.py)',
  'detail':'正文已教一候選一列、共參數與按題回填的概念，我可推出獨立列→分數/機率隨ID搬動。但candidate_batch與grouped_scores是本次實作直接調用的新必要helper；實際ID如何變意義文字/編號列、slices/max_candidates約定與分數如何回填都只外連，沒有已完整教過的精確教材回指。CandidateScorer原碼不會補回上游資料安排。',
  'learner_impact':'學生能復述結構性質，仍只能信import或外跳原始碼，無法只在教材追出/重寫這一題的完整必要計算與名稱對應；不符合本輪必要程式在教材的要求。',
  'already_taught':'每候選同state/question/自己的meaning、一列一分、共享網路、重排性質及結果邊界已足。',
  'minimal_revision':'內嵌兩helper必要精簡版或一題完整列/切片/回填計算，給關鍵輸入與輸出對應；不需重貼整個資料框架。',
  'discovery':'獨立首讀unit 2先保存未知，unit 3追加概念釐清但程式缺口保留；未讀作者或他人答案。',
  'checkpoint_indices':[2,3,6]
 }],
 '19.19':[{
  'id':'products-19.19-version-evidence-inline','necessary':True,'severity':'增加理解負擔',
  'location':'19.19 unit 0版本承諾、unit 2命令、unit 3外連操作頁、unit 4節末',
  'source_quote':'一份可使用的成品，要把**權重、網路設定、tokenizer及決策規則**對到同一個固定版本。\n列出固定HF版本、候選ID、三種問題、本機JSON服務及重訓命令。',
  'detail':'兩層release/manifest的抽象角色已說清，CLI與世界正解2/no也可理解。但B/C的具體固定HF commit、artifact manifest指紋與一筆actual載入輸出的核對欄位都沒有在本頁列出，只說成品操作頁會列。讀者尚無可在教材直接比對的身份與版本輸出例。',
  'learner_impact':'要完成標題的「確認自己用對版本」仍必須跳到非本頁操作文件，或相信預設download；命令正常結束/答案剛好2不能確認載的是哪套權重和policy。這不是要求貼全部安裝文件。',
  'already_taught':'為何版本需整體鎖定、推論包和續訓包、下載與離線、正確答案及實際輪數契約皆足。',
  'minimal_revision':'在教材給B/C固定release身份及一次實際下載/載入JSON必要片段，指出哪個欄位與哪個manifest SHA比較、核對成功是什麼；Colab/CLI僅重跑。',
  'discovery':'首讀unit 3保存具體查核材料未知，unit 4未釐清；未開外部README補。',
  'checkpoint_indices':[3,4]
 }],
 '19.21':[{
  'id':'products-19.21-cnn-minimum-operation','necessary':True,'severity':'增加理解負擔',
  'location':'19.21 unit 2 TinyAV.vision；unit 4節末仍未解說',
  'source_quote':'nn.Conv2d(1, 4, 3, padding=1), nn.GELU(), nn.AvgPool2d(2),\n            nn.Conv2d(4, 8, 3, padding=1), nn.GELU(),\n            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 12)',
  'detail':'可用先備只有12.17對CNN「共享小窗計算局部特徵」的最低定義。這份完整CPU模型新用了Conv2d通道/窗/補邊參數與AvgPool2d、AdaptiveAvgPool2d兩種空間縮減；正文沒有夠用的操作/中間形狀說明，也未精確回指已完整教過的地方。最終Linear(8,12)、輸出shape與梯度True不能代替這些新動作的介紹。',
  'learner_impact':'指定背景的讀者能追影音時間、attention與最後head，卻仍需課外PyTorch知識或查API才能說明影格像素如何成為8格摘要再到12維；此新影格入口是完整計算必要部分。',
  'already_taught':'整體CNN角色、輸入灰階圖/輸出向量與共享小窗概念；時間/遮罩/兩層融合/record已充分。',
  'minimal_revision':'短述兩Conv的共享小窗/通道、2×2平均縮減和全圖每通道平均的行為與中間形狀，連起8格摘要；不要求完整卷積理論或手寫算子。若前文已完整教，給精確可用回指。',
  'discovery':'首讀unit 2明記未教新操作，unit 3–4結果/驗收文字未補；未用模型課外知識代作者。',
  'checkpoint_indices':[2,4]
 }],
 '19.22':[{
  'id':'products-optional-p4-label','necessary':False,'severity':'選讀改善',
  'location':'19.22 unit 6；同一用語亦見19.23 unit 7',
  'source_quote':'正式模組與下載介面在P4才實作。\n正式模型工程預定在P4寫入`tiny_perceptron/omni_products/decision.py`',
  'detail':'P4沒有在已讀教材或通用說明定義；我只能由「才實作」知道正式工程未完成，不能知道它指教材哪一段。無需為此讀作者PLAN解碼。',
  'learner_impact':'不影響CPU機制、輸出及未完成狀態的理解，但留下對學生無法定位的工程代號。',
  'minimal_revision':'改成可讀的後續實作階段敘述，或提供學生可用的明確定位；不增加本頁必要概念。',
  'checkpoint_indices':[6]
 }]
}
pages=[]
nonpass=[]
for page_id in plan['main_pages']:
 session=sessions[page_id]
 directory=pathlib.Path('docs/course-revision-20261016/reader-sessions')/session
 opening=directory/'opening.json'
 note_paths=sorted(directory.glob('note-*.json'))
 note_refs=[{'path':str(p),'sha256':sha(p)} for p in note_paths]
 authored_notes=[json.loads(p.read_text()) for p in note_paths]
 checks=[]
 for p,n in zip(note_paths,authored_notes):
  for i,c in enumerate(n['concept_checks']):
   check=dict(c)
   check['checkpoint']={'path':str(p),'sha256':sha(p),'unit_index':n['unit_index'],'concept_index':i}
   checks.append(check)
   if c['verdict']!='已足夠':
    nonpass.append({'page_id':page_id,'note':{'path':str(p),'sha256':sha(p)},'unit_index':n['unit_index'],'concept_index':i,'original_verdict':c['verdict'],'concept':c['concept'],'final_status':'待作者修正後由本讀者本人callback','reason':'概念角色後段部分釐清已追加在後續note，但本輪該必要缺口仍未解；原note不覆寫。'})
 m=meta[page_id]
 page_issues=[]
 for issue in issues.get(page_id,[]):
  item=dict(issue)
  item['session_evidence']=[note_refs[i] for i in item.pop('checkpoint_indices')]
  page_issues.append(item)
 pages.append({'page_id':page_id,'source_sha256':frozen[page_id]['source_sha256'],
  'figures_sha256':frozen[page_id]['figures_sha256'],'verdict':m['verdict'],
  'reader_summary':m['summary'],'missing_visuals':m['visual'],'checks':checks,'issues':page_issues,
  'necessary_code_check':m['code'],'visual_check':m['visual'],
  'ai_estimate':{'minutes_min':m['minutes'][0],'minutes_max':m['minutes'][1],'reason':m['reason']},
  'session':session,'session_evidence':{'opening':{'path':str(opening),'sha256':sha(opening)},'notes':note_refs}})
report={
 'reviewer_task':'/root/omni_reader_products',
 'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_freeze':{'path':freeze_path,'sha256':sha(freeze_path)},
 'reading_plan':{'path':plan_path,'sha256':sha(plan_path)},
 'reading_route':{
  'context_pages':plan['context_pages'],'main_pages':plan['main_pages'],
  'context_method':'依plan先以reader.py supplement實際取得全部46頁context，再按main順序start/next。supplement以完整頁方式顯示context（含原位置折疊文字）；這不是context逐段盲讀驗收。沒有開作者PLAN/OUTLINE/CONTRACTS/SOURCES/project-context、他人review或未開放主文補洞。',
  'exposure_effects':'9.12已預告19.23的對稱none；9.13/9.16已預告19.17 Auto；11.7/12.17/12.18/17.17已預告新影音/聲音擴充；16.15已完整示範共享M。19.14/19.15/19.18等context先暴露19.13用途例、舊B/C成績與Auto成本。主文notes逐次記當時可用前文，不把交叉回指當本頁已教。',
  'main_method':'52個正文單位，每次親寫五欄與concept_checks後才reader.py next。主文折疊選讀未開。兩張main當前圖按freeze snapshot實際渲染及view_image；PNG保留/tmp。',
  'first_reading_limit':'這是指定context補讀路線下的主文首次順序閱讀，並非從全書入口無預告盲讀。未用後段覆寫當時未知。'},
 'estimation_basis':'AI對高中數學、依指定路線且已讀必要context學生的主觀逐頁閱讀/手算範圍；非模型執行秒數、非人類受試資料。context補讀、安裝網路、正式訓練及未解缺口的外查時間均不含。',
 'unverified':['完整網站桌面/手機排版與實際頁面閱讀路徑','教材程式實際執行、公開下載/離線載入與HTTP服務','正式訓練、研究數字/來源技術真偽與未實測新版能力','demo-sound.wav實際播放與音質/延遲','歷史context loop-shared-depth.svg在本渲染環境中文字成缺字方塊，未據此判該圖視覺通過；兩張main新圖可讀'],
 'pages':pages,
 'original_nonpass_checkpoints':nonpass,
 'closure_policy':'本original封存首讀。必要問題未解不pass；修稿後須本讀者依新凍結版本逐項callback另存checkpoint_closure/current，不由作者改判。',
 'scope_signature':'只簽第一輪理解與所列實看圖；不代技術或銜接兩輪簽名。未修改教材、未提交Git、未啟動正式訓練。'}
path=out/'reader-original.json'
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(path)
print('pages',len(pages),'notes',sum(len(p['session_evidence']['notes']) for p in pages),'necessary',sum(i['necessary'] for p in pages for i in p['issues']),'optional',sum(not i['necessary'] for p in pages for i in p['issues']),'nonpass checkpoint concepts',len(nonpass))
print('report sha256',sha(path))
