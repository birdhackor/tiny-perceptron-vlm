import json,hashlib,datetime,pathlib,collections
ROOT=pathlib.Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/course-revision-20261005/continuity'
ART=BASE/'artifacts/13_16'
assignment=json.loads((BASE/'progress.json').read_text())['assignments']['13_16']
inv=json.loads((BASE/'inventory.json').read_text())
meta={p['page_id']:p for p in inv['pages']}
trace=[json.loads(s) for s in (BASE/'traces/13_16.jsonl').read_text().splitlines()]
views=[json.loads(s) for s in (ART/'pageviews.jsonl').read_text().splitlines()]
first=[r for r in trace if not r.get('event')]
by_page=collections.defaultdict(list)
for r in first: by_page[r['page_id']].append(r)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def note(r):
 return {k:r[k] for k in ['recorded_at','scope','unit_index','unit_subheading','reference_subunit','unit_sha256','prior_material','new_problem','switch_notice','understanding','prediction','questions','figure_check','status','incremental_disclosure_note'] if k in r}
def page_record(pid):
 records=by_page[pid]
 status='revise' if any(r.get('status')=='revise' for r in records) else 'pass'
 return {'page_id':pid,'title':meta[pid]['title'],'source':meta[pid]['source'],'selector':meta[pid]['selector'],'source_snapshot':meta[pid]['snapshot'],'source_sha256':meta[pid]['source_sha256'],'figures_sha256':meta[pid]['figures_sha256'],'status':status,'pass_or_revise_reason':[r.get('understanding') for r in records], 'reading_units':[note(r) for r in records]}
order=[]
for n,r in enumerate(trace,1):
 d={'sequence':n,'page_id':r.get('page_id'),'recorded_at':r['recorded_at'],'event':r.get('event','first_reading'),'scope':r.get('scope'),'source_sha256':r.get('source_sha256'),'figures_sha256':r.get('figures_sha256')}
 for k in ['unit_index','unit_subheading','reference_subunit','unit_sha256','resolves_page_id','clarified_at']:
  if k in r:d[k]=r[k]
 order.append(d)
issues=[]
for r in first:
 for q in r.get('questions',[]):
  if not q.get('id'): continue
  item=dict(q,page_id=r['page_id'],first_recorded_at=r['recorded_at'],source_sha256=r['source_sha256'],figures_sha256=r['figures_sha256'],status='revise' if q['severity']=='增加理解負擔' else 'optional')
  if 'unit_sha256' in r:item['unit_sha256']=r['unit_sha256']
  item['later_clarifications']=[{k:v for k,v in c.items() if k!='original_text'} for c in trace if c.get('event')=='later_clarification' and c.get('issue_id')==q['id']]
  issues.append(item)
by_id={i['id']:i for i in issues}
by_id['C13_16_02']['source_exact_quote']='沿用[prefill/decode](16.md#16.2)的四格例子：cache保存每層注意力裡的Key與Value，沒有把完整回答存成答案，也沒有免去新位置的所有工作。'
by_id['C13_16_02']['prior_location']='16.2正文：提示IDs [1,2,3,4]，直接假定接著選到ID5，K位置軸由4變5；16.3正文改為三格提示加第四格。'
by_id['C13_16_02']['mainline_impact']='本節首段、圖與切片已足以重新分辨3+1，沒有卡住cache機制；需避免讓讀者把不同四格稱呼當成同一例子。'
by_id['C13_16_03']['source_exact_quotes']=['代價只計助手回答與EOS，兩側分母為36／69個有效目標；匹配則按5／10題的完整原始答案ID核對，最多生成32個新ID。','還要看改動：起點是[T.4](../training.md#T.4)的單頭SFT模型。']
by_id['C13_16_03']['prior_location']='15.13的TinyStories代價/補全文字既有實測；16.2正式速度提示Once upon a time，且明說不是屬性問答；16.4選讀突然換成5/10題完整答案匹配。'
by_id['C13_16_03']['mainline_impact']='影響選讀品質表的材料辨識；384與96 bytes的主線結構示範清楚。後文16.5的shape?→circle及16.7家族切分補明任務，不抹去首次16.4僅猜到屬性問答的紀錄。'
by_id['C13_16_04']['source_exact_quote']='圖解curriculum_learning_flow.svg中的「X 與可讀前文的遮罩」「Y 與指定計分的位置」「依代價算出梯度」。'
by_id['C13_16_04']['mainline_impact']='桌機能辨識完整流程；手機圖中小標籤需費力辨讀。正文已解釋mask與梯度，不能因此宣稱圖中每個流程標籤對初入讀者都可讀。'
by_id['C13_16_04']['screenshot_evidence']=[s for v in views if v['page_id']=='curriculum' for s in v['screenshots']]
for i in issues:
 i['scope_limit']='學生讀者的前後銜接與當頁可讀性判斷；未作新的技術查證或能力實驗。'
report={
 'schema_version':1,
 'stage':'independent_continuity',
 'assignment':'13_16',
 'reviewer_task':'/root/continuity_13_16',
 'fork_turns':'none',
 'reviewer_identity':'本輪全新第三輪學生讀者；未沿用作者、第一輪學生或正確性審閱者身分。',
 'reader_background':'入門Python、高中基本數學；沒有以專家模型知識填補教學空白。',
 'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'revise',
 'verdict':'58個primary頁完成首次閱讀。55頁pass、3頁revise；三項均增加理解負擔，沒有主線理解阻礙。另三個正文用字提供選讀改善。',
 'method':{'instructions':'docs/review-tools/continuity-reviewer-instructions.md','protocol':'.agents/skills/clear-tutorial/references/review-protocol.md','instructions_sha256':sha(ROOT/'docs/review-tools/continuity-reviewer-instructions.md'),'protocol_sha256':sha(ROOT/'.agents/skills/clear-tutorial/references/review-protocol.md')},
 'source_inventory_path':'docs/course-revision-20261005/continuity/inventory.json',
 'source_inventory_sha256':sha(BASE/'inventory.json'),
 'figure_inventory_path':'docs/course-revision-20261005/continuity/figure-snapshots.json',
 'figure_inventory_sha256':sha(BASE/'figure-snapshots.json'),
 'trace_path':assignment['trace_path'],
 'trace_sha256_at_report':sha(BASE/'traces/13_16.jsonl'),
 'primary_page_ids':assignment['primary_page_ids'],
 'primary_page_count':len(assignment['primary_page_ids']),
 'primary_page_results':[page_record(p) for p in assignment['primary_page_ids']],
 'context_before':[page_record(p) for p in assignment['context_before']],
 'context_after':[page_record(p) for p in assignment['context_after']],
 'supplementary_context':[dict(page_record('4.5'),reason='chapter-14導讀明確連結一層文字模型，依引用補讀Attention/FFN/殘差的背景；只讀4.5，没有聲稱已讀完整第3/4章。')],
 'actual_read_order':order,
 'handoffs':[
  {'from':['7.17','7.18'],'to':['chapter-13','13.1'],'understanding':'共用底座可分支接示範、偏好或作答後回饋；不是必走SFT→DPO→PPO串列。13章先談兩篇回答的偏好，再談規則評分有限卡PPO；素材與方法切換在導讀與13.1有說明。','status':'pass'},
  {'from':['12.15','12.16'],'to':['chapter-13'],'understanding':'前文聲音入口與共用歷史是自行訓練的有限設計、手寫希望回答；13章沒有聲稱接著訓練這個語音模型完成泛用聊天。圖中的希望回答與既有實測分得清楚。','status':'pass'},
  {'from':['13.1','13.2','13.3'],'to':['13.4','13.5','13.6'],'understanding':'格式偏好兩篇先變特殊ID，再對助手回答與EOS求整段log機率；policy相對固定reference的改變才進DPO代價。小表與梯度示範不是整個模型訓練成功的證據；實際七題結果另外說明。','status':'pass'},
  {'from':['13.9'],'to':['13.10','13.11','13.12','13.13','13.14','13.15'],'understanding':'13.10明說改用有限预寫回答卡与規則回饋。reward、critic、old policy與fixed reference各有不同工作；ratio重用舊作答，clip只停止特定方向鼓勵，不保證所有機率留在區間。13.15按保存→重算→梯度→step順序接起。','status':'pass'},
  {'from':['13.16'],'to':['13.17'],'understanding':'13.16首次見DPO分支時只知它不是先前七題語言模型，尚未知卡如何用DPO；13.17立即用卡log機率接入13.5代價，保留首次疑問的later_clarification紀錄。兩路不同採樣流程、同360步不同成本；既有選卡通過不等成品已回灌。','status':'pass','first_question_preserved':True},
  {'from':['13.17'],'to':['chapter-14','4.5','14.1'],'understanding':'先從偏好更新轉到同底座內部設計；導讀重新說明先前一層文字模型的位置/注意力問題。我依正文引用僅補讀4.5，才帶著Query/Key比對、Value取回、FFN每位置轉換與殘差進14章。','status':'pass'},
  {'from':['14.1','14.6'],'to':['14.7','14.8','14.9','14.10'],'understanding':'旋轉分工、規範與門控、共享權重逐項獨立示範；進長文支線明說一字代token的縮小例。舊整數位置0..7與新0..15、座標刻度縮小而非砍掉內容可分清；快慢轉速是後續新示意。YaRN旋轉混合與分數scale各自影響不同步驟，未把座標可算當長文已懂。','status':'pass'},
  {'from':['14.10'],'to':['chapter-15','15.1','15.2','15.3'],'understanding':'MoE導讀重新提出每token都算完整FFN的成本，15.1固定Dense兩張表，再建獨立expert和router。expert是可學網路，不是假稱真人學科專家；soft全算還未節省。','status':'pass'},
  {'from':['15.4','15.5','15.6'],'to':['15.7','15.8','15.9'],'understanding':'topk的索引/連續比例分開，兩個expert同寬加權後放回原token。15.5暫留top1 p/p恆1的疑問，15.7用微動不改輸出的手算接上梯度零；15.8統計soft與真dispatch差別，15.9用可微p接負載代價，沒有把離散索引變可微。','status':'pass'},
  {'from':['15.10','15.11','15.12','15.13'],'to':['chapter-16','16.1'],'understanding':'總權重、局部active代理量、實際派工/容量丟路徑以及CPU/既有L4時間各自有範圍。TinyStories結果有材料與家族保留說明；16章導讀先固定成本問題與語義核對，16.1開列品質/time/weight/cache等指標，沒有把MoE少啟用直接當速度勝利。','status':'pass'},
  {'from':['16.2'],'to':['16.3'],'understanding':'prefill保存舊位置K/V，decode新Query仍讀所有cache；兩路比較須都讀完第四個ID、位置offset保持一致。16.3文字說沿用四格，但16.2四格是prefix後接第五，本節三加一，產生局部材料稱呼負擔。','status':'revise','issue_ids':['C13_16_02']},
  {'from':['16.3'],'to':['16.4','16.5'],'understanding':'四Query仍各自計算但KV少組，384→96只量cache。主線結構示範清楚；選讀表突然以5/10問答評品質，首次只有T.4起點，16.5 shape?→circle才幫我確認屬性任務。packing則明說新文件A/B與之後純答案續寫。','status':'revise','issue_ids':['C13_16_03']},
  {'from':['16.5','16.6'],'to':['16.7'],'understanding':'隔離packing避免跨文件讀取與錯誤下一個目標，梯度累積改批份額但需全份等價；混精度是另個局部計算改動。16.7既有實測清楚重新載入單頭SFT而不是延續四頭efficiency，原權重與optimizer仍FP32，有限/成功更新不等答對。','status':'pass'},
  {'from':['16.8'],'to':['16.9','16.10','16.11','16.12','16.13'],'understanding':'SDPA介面不等選到Flash核心；Flash分塊保全部候選並維持online softmax分母/分子，checkpoint改中間存留、compile另算首呼叫與回本。sliding/local則真的改可見位置；最後三格圖把可見/計算排程分開，不把Flash三個字當長上下文閱讀能力。','status':'pass'},
  {'from':['chapter-13','chapter-14','chapter-15','chapter-16'],'to':['curriculum'],'understanding':'課程入口總表把這幾章任務和方法放回主線/支線；自訓有限任務與成熟模型路線分開，後續19/20成果只當本頁預覽而非本輪已讀證據。正文/Notebook同源与实读审阅原则清楚；手機學習流程圖細字需改善。','status':'revise','issue_ids':['C13_16_04']},
  {'from':['16.7','16.13','curriculum'],'to':['chapter-17','17.1','17.2'],'understanding':'16.7權重仍FP32留下真正縮權重bytes的問題；17導讀與17.1先按格數×每格bytes說明dtype，不把astype整數當量化完整配方。17.2重新定義scale為浮點相鄰格距，手算round→restore與範圍/精度權衡可跟上，沒有偷用14章attention scale的用途。','status':'pass'}
 ],
 'issues':issues,
 'resolved_during_first_reading':[ {k:v for k,v in r.items() if k!='original_text'} for r in trace if r.get('event')=='later_clarification'],
 'visual_followups':[ {k:v for k,v in r.items() if k!='original_text'} for r in trace if r.get('event')=='visual_followup'],
 'browser_method':{'browser':'/usr/bin/chromium through Playwright','args':['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'],'goto_wait_until':'domcontentloaded','goto_timeout_ms':15000,'screenshot_full_page':False,'screenshot_timeout_ms':15000,'viewports':[{'width':1280,'height':900},{'width':390,'height':844}],'actual_site':'http://127.0.0.1:8765/<page_id>.html','view_image_used':True,'coverage_statement':'必要圖於正文實際桌機及手機大小實看；高圖以多段viewport截圖補齊，沒有以原SVG字串、DOM尺寸或原SVG放大代替觀看。pageview紀錄包含無圖頁的真實開頁；這些開頁不宣稱已看遍該頁所有像素。details打開後只讀本頁公開內容，未跟進所連技術報告或外部資料。'},
 'pageviews':views,
 'pageview_count':len(views),
 'screenshot_count':sum(len(v['screenshots']) for v in views),
 'unverified_scope':[
  '未讀完整第1–12、17之後各章；前文僅7.17、7.18、12.15、12.16及依正文引用補讀4.5，後文僅chapter-17、17.1、17.2。',
  '未讀作者筆記、其他審閱、來源實作、原始技術報告或外部論文；本頁公開選讀摘要與連結標籤有讀，未以被連結內容補理解。',
  '本輪沒有執行CPU教材程式、GPU、訓練或能力測試；網站既有CPU輸出與本頁報告數字只作當頁閱讀。',
  '未重新查證技術事實、資料家族切分或既有模型結果；本輪證據是來源閱讀與圖解可讀性的個人紀錄。',
  '未檢查所有瀏覽器、所有螢幕尺寸、每頁所有像素、互動功能或每個CPU輸出框；只在指定桌機/手機viewports看必要圖和其附近內容。',
  'curriculum各主要##逐單位讀並記錄；unit2的main與其nested閱讀路線在同一次source bundle揭露後分别追加紀錄，沒有宣稱nested小段独立盲讀。unit5參考索引的各###同時揭露作一個來源索引單位，再逐項保存自己的紀錄與小段SHA，沒有先讀外鏈。',
  '讀者問題後來解開另有追加紀錄；首次13.16與16.4的猜測沒有改寫成事後全懂。',
  '本報告仍為原始版本的首次審閱；尚未回讀未發生的修正。'
 ]
}
# Validate record identities and actual artifacts. This does not establish reading by itself.
assert len(report['primary_page_results'])==58
assert set(by_page)==set(assignment['primary_page_ids']+assignment['context_before']+assignment['context_after']+['4.5'])
assert sum(p['status']=='revise' for p in report['primary_page_results'])==3
for pid in by_page:
 assert sha(ROOT/meta[pid]['snapshot'])==meta[pid]['source_sha256'],pid
for v in views:
 assert v['status']==200,v
 for s in v['screenshots']:assert sha(pathlib.Path(s['path']))==s['sha256'],s['path']
for i,r in enumerate(trace):
 if i: assert trace[i-1]['recorded_at']<=r['recorded_at']
 assert r.get('source_sha256')==meta[r['page_id']]['source_sha256']
out=BASE/'reports/13_16.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report_path':str(out),'report_sha256':sha(out),'primary':len(report['primary_page_results']),'pass':55,'revise':3,'first_reading_units':len(first),'trace_records':len(trace),'pageviews':len(views),'screenshots':report['screenshot_count'],'issues':[{'id':i['id'],'page_id':i['page_id'],'severity':i['severity']} for i in issues]},ensure_ascii=False))
