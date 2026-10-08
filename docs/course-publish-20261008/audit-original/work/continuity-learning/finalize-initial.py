import pathlib,json,hashlib,re,datetime
B=pathlib.Path('/workspace/work/tutorial-audit-20261008');W=B/'work/continuity-learning';M=json.loads((B/'manifest.json').read_text());P={p['page_id']:p for p in M['inventory']['pages']};G=M['groups']['learning']['pages']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
main=B/'reports/continuity-learning-main-notes.json';ms=json.loads((W/'main-notes-seal.json').read_text());assert sha(main)==ms['sha256']
obj=json.loads(main.read_text());obj['stage']='initial independent full-source continuity review after frozen main notes and supplemental text reading'
obj['main_notes_preservation']={'path':str(main),'sha256':ms['sha256'],'helper_seal':str(W/'main-notes-seal.json'),'unchanged_after_extras':True}
notes={
'5.1':'補充另用9篇/600更新/兩層GPU結果，明說不能套width8/40診斷；固定題更新與新文loss分開，無須移回正文。',
'5.6':'補充暖身上限total/4及1–3步邊界；正文40步5→10練習均在合法範圍，當下理解不依賴30被截成10，保留選讀。',
'5.7':'80步第40續訓逐權重差0是特定模型與排程的額外驗證，主文已教資料/亂數/排程須存；不替所有GPU保證。',
'5.8':'補充side未提供所以right非必然錯與故事代價降但生成零碎；正文已有需線索與loss/生成分判，資料分母/完整報告屬證據延伸。',
'5.9':'補充342462曝光非不同文章、6.1885訓練/16.0354整組計時含哪些工作；主文已教秒數邊界，無必要依賴藏折疊。',
'5.10':'上游train檔自行分側不是官方測試，詩按整首分；主文已寫用途非檔名和自己的留出非官方成績。',
'5.11':'實跑合空白僅比較，原文仍保留，512/365無精確重複；與主文正規化/訓練原文區分相合。',
'5.12':'實跑未做近重複家族判定，不能宣稱完全隔離；正文未將該實跑當全部家族洩漏已排除，保留證據限定並記原始報告未核。',
'5.13':'實跑16/32×16/64、150步，20765/20623目標不同、訓練均不同材料；正文已明示計畫非實測與監督/算力不同尺度。',
'5.16':'回放兩支500步16筆卻18453/36384有效目標；正文筆數與token比例完整，補充提供其實際發生位置。',
'6.1':'詩生成32byte截半字與重複是不同失敗，提示在完整碼點邊界；6.7正文已建完整解碼規則，例子不需搬入成本段。',
'6.3':'192篇分153/19/20、BPE512只訓練側，空格換行完整還原；原例全256byte有還原契約，不靠長實驗補機制。',
'6.4':'19篇3894/2112普通token不含BOS/EOS，264/512字表與41824/57696模型；延伸明晰實測分母不改正文取捨。',
'6.5':'20篇4213/2503含EOS目標、4193原文bytes，曝光同664759但非相同參數/監督/運算；主文已教共同分母與EOS約定。此圖曾提前看，路線偏差保留，不改main。',
'6.6':'BPE普通<user>引用不含控制ID；單登special不足是正文同契約的額外核對，不需新必要前提。',
'6.8':'同256原文byte片段對兩工具、384位置能容全，不因BPE短多取文；正文固定byte/token不同預算已教。',
'7.9':'cire原ID末2真EOS但circle內容錯，驗5/test10全結束；主文停止與內容分開已成立。',
'7.10':'UltraChat首輪各120byte加EOS只示片段，80步各10題0匹配不宣稱完整對話；正文有分母非完整答案已足夠。',
'7.11':'直接SFT900與文字250再SFT900不匹總預算，非前章故事模型；正文已有多文字預算限制，無需補正文路線。',
'7.12':'另12屬性組合實験支線不同預算，四格示例只切分；不把另一套成績當此例已訓練。',
'7.13':'基準sft/model.pt直接900步5/10不是較好pretrain-sft，逐題ID/EOS存；正文已定保存實際起點與分項，版本連結是復現補充。',
'7.14':'同45題/300更新/33733目標，只4shape錯標、兩家族，7/10vs6/10 loss反向；正文明限單種子少題，原始值待技術驗。',
'7.15':'原A5/10→0/10，新B0/7仍0/7，交換加數家族一致；正文已沒有把新能力學好當遺忘必要條件。',
'7.16':'回放A6/10但新B0/7，舊長答監督近翻倍，輸入成本也不同；正文已承認條件配方與非同token純因果。',
'7.17':'文字同45題而非廣泛知識，1/10→7/10與直接5/10多250步，正文成本/重用理由非以此小實驗證產業必勝。',
'7.18':'R1-Zero底座直RL與少量cold-start再後訓練兩路，正文已說三教法不必依序；合理延伸而非當下PPO操作先備。',
'7.19':'HF新增與vLLM總容量版本來源，滑窗另契約、forcedEOS不同；主文已限定保持完整歷史並算手設16，外API不為當下必要。',
'7.20':'平行分支有Transformer層且共用詞表投影，主圖只教任務；主文用網路而非固定一Linear且未說各有獨立整矩陣，不需搬內部。',
'7.21':'t+1/t+2與論文加權不同，.5本例非普遍默認已正文；完整論文配方不必先學才能理解邊界。',
'7.22':'V3圖3順序/4.2D1來源，省投影/LN/Transformer內部保資料流；正文已說一段額外神經網路並實際一模組，圖不用補更多内部方能分前文與答案。',
'8.1':'風格成績只固定積木句，不量機智洞察，JSON當地給格式例；正文已按分組比喻rubric而非誇大實測範圍。',
'8.2':'三已教style固定權重貪婪切換，新名字不保懂；主文已同樣限定提示未必補能力。',
'8.3':'同加法基模七題原也0，450步短16591/長318991監督，文風改不因果破壞原加法；正文新題仍數錯與預算差明說。',
'8.4':'三style各7/7風格0/7數值、日期同模型，固定句/鍵int/EOS明晰；非任意創意，正文同限制。',
'8.5':'七JSON都可解析鍵int而值全錯，直接印证正文兩門；不把解析代替任務內容。',
'8.6':'24日期按19/2/3家族、缺題date?帶id，六最後3澄清對/3保日期錯；正文核心缺才問/够就做已齊。',
'8.7':'短7題14有效與長308，固定句拉低loss8.41357/.26805但數全錯；正文已講平均答案組成影響，具體分母是證據補充。',
'8.8':'11Linear插rank4，9504可訓練/141568原凍結；同318991監督對照，風格6/7vs7/7且算術0/7；費用非6.7%記憶體、失敗UTF8均限定。',
'8.9':'真adapter切回差0、merge同前文差6.68e-6在1e-4容差，正文已有浮點順序和重複加入禁則；不用另教部署即可理解。',
'8.13':'數字指紋按參數固定順序供基模核對，α/rank=1，未訓不同rank；正文已有形狀/公式不等完整實験限制。',
'8.14':'論文指令/正確任務/明約束支持分項，主文告示/要求/答案自行可驗，無需外文先備。',
'8.15':'SFT示範與FLAN指令對目標的來源及未見任務邊界；正文同材料對照與EOS必要部分均已有。',
'8.16':'IFEval可去markdown外框且不查naturalEOS，非此裸JSON/字段/list全契約；主文已明確本任務檢查規則非通用裁判。',
'9.1':'外部偏好/安全欄各保留、裁短上下文標簽需重查；主文兩越界相對更安全不當正例已足夠，外授權不當教學先備。',
'9.2':'保持缺信息改問猜6，補充具體失敗；主文未把規則程式當模型自知。',
'9.3':'行为版不对8与混合不对4单题，正文已說查纠錯内容、不能只搜不對。',
'9.4':'正式輸入只有id/permission無owner/公开，6/6只固定目标对应；正文已完整揭示不能證真實所有權，补充非隐藏必要限定。',
'9.6':'17题3拒/14正常含澄清，基模零拒且零功能，固定無法提供非真安全分類器；正文兩組責任与不同加法分母已有。',
'9.8':'136八規則家族切102/17/17、行為/混合不同目標預算，原模板16/17与改問法0/6；主文已限制新颖度且green→red非任務成功。',
'9.9':'320Hz>300阈值任务错low，T.5信心.749→.899不改答案；本地给task判准够观察校准、不需要学音訊编码内部。',
'9.10':'驗證CE選T.5，音訊test11/14固定但ECE/Brier上升；Brier候選平方差本地定義、五箱空保存与紙上tool略空不同规则明确；圖整平均距離不代ECE。'
}
log=[json.loads(s) for s in (W/'source-delivery.jsonl').read_text().splitlines()];delivered={r['page_id'] for r in log if r['phase']=='extras'};assert set(G)<=delivered
supp=[]
for pid in G:
 raw=pathlib.Path(P[pid]['snapshot']).read_text();ds=re.findall(r'<details\b[^>]*>.*?</details>',raw,flags=re.S)
 if not ds:j='實際核對無折疊補充；正文判斷不變。'
 elif pid in notes:j=notes[pid]
 else:j='實際讀到的唯一補充為本課工具/安裝/長訓練與重做入口；明說不需先完成長配方，沒有藏本頁必要機制或理由。'
 body=re.sub(r'<summary[^>]*>.*?</summary>','',ds[0],flags=re.S) if ds else ''
 body=re.sub(r'</?details[^>]*>','',body).strip()
 first=body.split('\n\n')[0] if body else '[本頁沒有折疊區]'
 supp.append({'page_id':pid,'source_sha256':P[pid]['source_sha256'],'details_count_read':len(ds),'quoted_basis':first,'post_main_supplement_judgment':j,'required_explanation_hidden_in_details':False,'main_notes_unchanged':True,'external_links_read':False})
for r in obj['pages']:r['supplemental_reading']=next(s for s in supp if s['page_id']==r['page_id'])
obj['supplemental_scope']={'pages_checked':79,'details_read':sum(s['details_count_read'] for s in supp),'route':'所有主文逐頁判斷寫入且helper封存後讀details文字；6.5提前實看補充圖的偏差另列','external_training_recipes_or_raw_reports_read':False,'notes':supp}
obj['actual_prerequisites']=[dict(p,used_as_dependency_for=[r['page_id'] for r in obj['pages'] if p['page_id'] in r['actual_dependencies']],reading_role=('necessary relation support' if any(p['page_id'] in r['actual_dependencies'] for r in obj['pages']) else 'supporting context actually read, not assumed essential')) for p in obj['actual_prerequisites']]
obj['visual_scope']['figure_views'].append({'page_id':'9.10','figure':'course/figures/audio_calibration_test.svg','source_sha256':P['9.10']['figures_sha256']['course/figures/audio_calibration_test.svg'],'viewed_renders':[str(B/'renders'/f'audio_calibration_test-{s}.png') for s in [640,360]],'route':'extras after main seal','judgment':'實看後核對兩卡同0–100%尺度，87.96/92.73藍條與固定78.57%紫虛線、驗8段選T.5/test14段11對標明；數值標注是ECE/Brier不等條與線整體差，與相鄰文字一致。'})
obj['unknown'][0]['scope']='正文與折疊引用的實測原始結果JSON、設定/資料manifest及外部原論文未獨立讀查；只核其教材敘述、分母解釋、比較限定，不能宣稱實測數字與原論文事實均驗真。'
strong_changes={'5.3':'原練習μ=0時第三梯度立刻反向、μ=.9仍走原方向；理由由历史保留而非负位置质量。','5.17':'原練習凍權重但x需grad仍x.grad等固定w；已教链式敏感度支持，非把冻結当全段不可微。','7.4':'原練習Q→QQ首有效index4→5但inputID4/target73不變，A→B改目标ID不改index。','7.7':'原练习仅去valid或仅去positions分别添PAD可读或换位置向量，所以预测不等；两个依赖不能互替。','7.19':'原练习小小/EOS仅2新增但漏书店；G充足与选EOS不代内容完成，容量不是PAD。','7.22':'原练习雨又停时主目标又、额外读标准又预测停；不能先读目标停；圖橙箭頭不返主核心。','8.8':'原练习rank/alpha2→4倍率维持1，可学56→112；增加窄通道宽度而不是原权重自动更新。','8.14':'原練習只抄第一行後两行中文目标反多一行，当前请求而非材料真实性决定交付范围。','8.15':'原練習请求JSON中文→英文，旧中文列表格式仍可同但示範违范围，必须请求和答案一起改。','8.16':'原練習中文→英文，旧回答format/EOS可過而内容范围错；同题joint不能平均。','9.4':'原练习owner/permission一起改自己True才在有效小世界，仅permission True他人超范围，不能推广公开性。','9.10':'原练习新增.25且对使低箱1正确/.25信心差.75，ECE.35变大并非新增错答；分箱与真值共同决定。'}
ids=list(strong_changes);obj['strong_pass_samples']=[{'page_id':pid,'quoted_basis':next(r['quoted_basis'] for r in obj['pages'] if r['page_id']==pid),'mechanism_property':next(r['checks']['mechanism_property'] for r in obj['pages'] if r['page_id']==pid),'necessity_use':next(r['checks']['necessity_use'] for r in obj['pages'] if r['page_id']==pid),'source_supported_variation':strong_changes[pid],'status':'strong pass within stated scope; empirical/runtime facts remain unverified'} for pid in ids]
obj['coverage'].update({'extras_scope_read':79,'folded_sections_read':len([s for s in supp if s['details_count_read']]),'strict_main_text_route':True,'strict_main_visual_route_except':['6.5'],'main_dependency_judgments_not_retrofitted_from_extras':True})
obj['sealed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();out=B/'reports/continuity-learning-initial.json';out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
seal={'reviewer':obj['reviewer'],'role':'continuity','group':'learning','at':obj['sealed_at'],'report':str(out),'sha256':sha(out),'main_notes_sha256':sha(main),'criteria_sha256':obj['criteria_sha256'],'coverage':{'main':79,'extras':79},'counts':obj['counts'],'meaning':'獨立來源初判byte seal；封存前未讀同行或舊判斷；非盲增量首讀或真人學生測試；6.5提前補充圖明記'}
sp=B/'reports/continuity-learning-initial-seal.json';sp.write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n');(W/'supplemental-judgments.json').write_text(json.dumps(supp,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(seal,ensure_ascii=False,indent=2));print('main seal unchanged:',sha(main)==ms['sha256'])
