import pathlib,json,hashlib,re,datetime
B=pathlib.Path('/workspace/work/tutorial-audit-20261008'); M=json.loads((B/'manifest.json').read_text());P={p['page_id']:p for p in M['inventory']['pages']}; G=M['groups']['learning']['pages'];W=B/'work/continuity-learning'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
rows=[json.loads(s) for s in (W/'actual-judgments.jsonl').read_text().splitlines()];R={r['page_id']:r for r in rows}
for pid in ['5.14','7.4','7.9','7.13']:R[pid]['quoted_basis']=R[pid]['quoted_basis'].rstrip('。')
R['5.6']['actual_dependencies']=['1.12'];R['5.9']['actual_dependencies']=['5.4'];R['5.9']['later_clarification']={'page_id':'5.17','reason':'eval與no_grad差別本頁已預告、不影響當前計時操作；5.17完成三開關機制'}
R['5.6']['later_clarification']={'page_id':'5.14','reason':'排程不自動選最佳率的限制；5.14另教同起點範圍試驗'}
prereq_basis={
'1.2':'通常先按完整來源分組，再在各組內切片',
'1.7':'先用指數函式 `exp` 把每個分數換成正數，再除以正數的總和',
'1.8':'正確答案得到的機率越少，代價越大；代價不是猜錯題數。',
'1.9':'它描述微小改變，不能保證跳很遠也變好。',
'1.10':'每段的敏感度相乘，就是**鏈式法則**',
'1.11':'它與參數形狀相同，每格敏感度都有自己的參數可對應。',
'1.12':'更新規則是 `新參數=舊參數−學習率×梯度`。',
'1.13':'若每輪只想用當前資料更新，應先清舊梯度，再算新代價、求導與更新。',
'1.14':'先算下一字分配、選一字、接回句尾，再把新字當下一次前文。',
'2.1':'真正訓練時，答案代價的梯度沿運算關係傳回用到的表格列，再由更新步驟改表',
'2.3':'每列一套配方，每套三個乘數',
'2.4':'一般的兩套純線性加權與偏移也能合成一套。',
'3.1':'每張卡乘自己的比例，再逐格相加',
'3.5':'取變異數平方根得到**標準差**，能用原數字的尺度看典型起伏。',
'3.6':'位置 i 只准讀位置 j≤i，即自己與更早的位置。',
'4.1':'位置特徵給每位置額外數字，兩者用途不同。',
'4.3':'每位置先減自己的平均，再除自己的標準差',
'4.5':'主路沒有被 LN 永久替換；LN 在分支中。',
'4.6':'每個候選各有一套權重，把八格加權成一個分數。',
'4.7':'它不會再幫你 shift。'
}
log=[json.loads(s) for s in (W/'source-delivery.jsonl').read_text().splitlines()];actual_prior=[]
for l in log:
 if l['phase']=='main' and l['page_id'] not in G and l['page_id'] not in actual_prior:actual_prior.append(l['page_id'])
for r in rows:
 pid=r['page_id'];r['source_path']=P[pid]['snapshot'];r['source_location']={'source':P[pid]['source'],'selector':P[pid]['selector']};r['quoted_basis']=[{'quote':r['quoted_basis'],'location':P[pid]['selector'],'route':'main'}]
 r['dependency_support']=[{'page_id':d,'relation_basis':(prereq_basis[d] if d in prereq_basis else (R[d]['quoted_basis'] if isinstance(R[d]['quoted_basis'],str) else R[d]['quoted_basis'][0]['quote'])),'source_sha256':P[d]['source_sha256'],'status':'實際已讀正文；全文銜接審可回讀前文'} for d in r['actual_dependencies']]
 raw=pathlib.Path(P[pid]['snapshot']).read_text();main=re.sub(r'<details\b[^>]*>.*?</details>','',raw,flags=re.S)
 r['unverified']=[]
 if '```python' in main:r['unverified'].append('程式未由本reviewer執行；正文已列操作與輸出解釋，未驗實際runtime輸出')
 if pid not in ['7.4','7.22']:r['unverified'].append('未看本頁實際網站截圖；圖檔實看另列，不能代替整頁版面驗收')
 r['checks']={'identity_role':r['current_learning_check'],'mechanism_property':r.pop('mechanism_property'),'necessity_use':r.pop('necessity_use'),'operation_and_next_use':r['example_scope_and_next_use'],'example_support_and_limit':r['example_scope_and_next_use'],'basis_status':'正文明說或附已教前提的最短正常推論；未將外部結果回算為教過'}
assert len(rows)==79 and set(R)==set(G)
issues=[
{'issue_id':'CL-L-01','page_id':'7.5','quote':'`logits` 是中間結果，PyTorch一般不保留中間節點的 `.grad`，所以 `retain_grad()` 明確要求留下它，相關葉參數區別見 [1.11](01.md#1.11)。','already_taught':'正文在程式前已完整給這段，程式後第一句原樣再給；中間梯度保存規則已足夠。','minimal_missing_relation':'無必要機制缺口；兩段重複造成不必要回讀。','reader_impact':'移除第二次重複後可直接讀問題logits=0與Q字表梯度>0的核心差別，縮短間距。','severity':'optional','minimal_repair':'刪程式後重複的retain_grad介紹句，保留兩種梯度對象與norm說明。','rewrite_scale':'paragraph','discovered_at':'7.5主文全文銜接審、初判封存前','provenance':'source-only independent；未讀同行或舊結論'},
{'issue_id':'CL-L-02','page_id':'7.5','quote':'相關葉參數區別見 [1.11](01.md#1.11)','already_taught':'1.11教各參數偏導與.grad同形狀；7.5本地已教中間logits一般不留.grad且retain_grad要求保留。','minimal_missing_relation':'所連1.11沒有教葉節點與中間節點的區別，連結的承諾與已讀內容不符。','reader_impact':'讀者選擇追這個輔助回看時找不到所指區別；本地必要retain_grad操作仍清楚，故不升必要負擔。','severity':'optional','minimal_repair':'刪除「相關葉參數區別見1.11」或改成實際指向1.11的每格參數梯度複習。','rewrite_scale':'paragraph','discovered_at':'7.5主文，1.11正文已實際讀過後核對','provenance':'source-only independent；未讀同行或舊結論'}]
figs=[]
for pid in G:
 for f,h in P[pid]['figures_sha256'].items():
  if f.endswith('audio_calibration_test.svg'):continue
  figs.append({'page_id':pid,'figure':f,'source_sha256':h,'viewed_renders':[str(B/'renders'/f'{pathlib.Path(f).stem}-{s}.png') for s in [640,360]],'judgment':'640/360標記、字、線及当前依赖對应可讀，具體關係見逐頁example_scope；非整頁驗收','route':'supplemental image viewed early' if f.endswith('tokenizer_common_scale.svg') else 'main'})
visual={'figure_views':figs,'page_capture_views':[{ 'path':str(B/'page-captures'/f),'sha256':sha(B/'page-captures'/f)} for f in ['7.4-1280.png','7.4-390.png','7.4-1280-context-0.png','7.4-390-context-0.png','7.22-1280.png','7.22-390.png','7.22-1280-figure-0.png','7.22-390-figure-0.png']], 'page_judgments':{'7.4':'桌面/手機主文說明→三欄圖→程式→對應解釋→閉合選讀，圖在首次需要原位相鄰；手機上下文截圖三欄與索引4可讀。','7.22':'桌面/手機主文先給主/額外兩路資料，再原位圖，teacher-forcing與防洩漏說明緊接下方；手機figure截圖可讀橙標準項只往後讀。'},'limitations':['全頁長截圖在工具顯示有縮放；可讀性依原尺寸context/figure補圖判，不用縮略長圖代判。','新預覽省略runtime outputs與reading annotations，本reviewer未驗這兩層或互動點開/網站實際執行。','其餘77頁網站位置、折疊/捲動/數學排版未驗。'], 'route_deviation':{'page_id':'6.5','event':'封存main notes之前依manifest圖清單實看折疊補充中的tokenizer_common_scale-640/360；沒有讀details文字。','new_material':'圖含共同4193原文bytes、逐byte4213與BPE2503計分目標數；loss/BPB值亦已在主文。','effect':'6.5不能聲稱嚴格只讀主文視覺路線；main文字判斷及7byte機制圖不依赖新增數量，其餘頁未用此圖補必要先備。'}}
obj={'reviewer':'/root/continuity_learning','role':'continuity','group':'learning','stage':'main-only text judgments saved before supplement text delivery; see one supplemental visual route deviation','criteria_sha256':sha(B/'freeze/criteria/SKILL.md'),'criteria_references':{n:sha(B/'freeze/criteria/references'/n) for n in ['review-protocol.md','calibration.md']},'actual_prerequisites':[{'page_id':p,'source_sha256':P[p]['source_sha256'],'quoted_basis':prereq_basis[p],'scope':'main only','reason':'所列依賴所需的更新、梯度、表示/答案/讀取或成本關係'} for p in actual_prior], 'pages':[R[p] for p in G],'issues':issues,'coverage':{'assigned':79,'main_text_read':79,'actual_page_judgments':79,'ids':G,'not_blind_increment_reader':True,'ai_assisted_not_human_student':True},'visual_scope':visual,'execution_scope':{'executed_training':False,'executed_course_code':False,'read_prior_human_judgments':False,'raw_empirical_results_verified':False,'work':'僅讀凍結正文與必要先備、看圖及截圖、保存審閱JSON；未改教材未重訓'},'unknown':[{'id':'CL-U-01','scope':'正文引用既有實測的原始結果JSON、配置和分母實際資料未查；本輪只核其教材說明與限定，不能驗真實結果。','kind':'unverified evidence scope; not diagnosed continuity defect'},{'id':'CL-U-02','scope':'其餘頁網站版面與runtime/annotation層，以及預覽互動未驗。','kind':'unverified presentation/execution scope; not diagnosed continuity defect'}],'strong_pass_samples':['5.3','5.10','5.17','7.4','7.7','7.19','7.21','7.22','8.8','8.14','8.15','8.16','9.4','9.9','9.10'],'counts':{'necessary_issues':0,'optional_issues':2,'unknown_scope_items':2}}
(B/'reports').mkdir(exist_ok=True);(B/'reports/continuity-learning-main-notes.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n'); print('Saved actual main notes:',len(obj['pages']),'pages; necessary0 optional2 unknown-scope2; early supplemental visual explicitly disclosed')
