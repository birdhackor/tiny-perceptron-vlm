from pathlib import Path
import json,hashlib,datetime,urllib.request
root=Path('docs/course-revision-20261006-phase6/continuity');base=root/'artifacts/a'
order=['18.12','18.13','18.14','chapter-19','19.1','19.2','19.3','19.4','19.5']
primary=['chapter-19','19.1','19.2','19.3','19.4']
inv=json.loads((root/'inventory.json').read_text());entries={r['page_id']:r for r in inv['pages'] if r['page_id'] in order}
trace=[json.loads(line) for line in (root/'traces/a.jsonl').read_text().splitlines()]
traces={r['page_id']:r for r in trace}
summary={
 '18.12':'共同候選與同前文的輸出分布讓異架構能學習，不要求內部寬度對上。隨機MoE/Dense程式只驗證形狀與梯度；故事蒸餾的選讀是另一次資料預算有限的實測。',
 '18.13':'圖或聲音前文展開長度不同，要各自找預測相同答案的前一列。16/4格與KL示範是人造表，真實聊天列另列；正常/空白圖的既有觀察只限該合成題組。',
 '18.14':'架構縮小、教師訊號與量化各有不同作用，四版對照才能分辨品質變化。程式只算隨機結構的tensor bytes，屬性题blue變ble是另一次訓練後對照。',
 'chapter-19':'由局部蒸餾/量化方法轉入具名、已訓練公開V2成品，三個自行訓練感知入口由同一文字核心回答；先看行為，再拆模型、資料與歷程。',
 '19.1':'歷史格式和當前App問題共同決定兩點回答。商品、字卡、錄音各有入口，仍由同核心回答；每種材料與可驗能力有明確範圍，一題成功不能代替整份考卷。',
 '19.2':'MoE每層四FFN選二加權，Dense一FFN；同寬同層同入口仍不代表同運算量，也因訓練歷程不同不能把品質差全歸架構。550字元詞表與隨機參數計數都明確區隔前章264-ID例子和已訓練成品。',
 '19.3':'先按來源或情境家族分資料，再在同份做衍生題，避免包原圖和換位版本跨考卷。記錄數、素材數與說話者數清楚分開；已知字元新組合和有限錄音意圖不被擴寫為陌生字或人。',
 '19.4':'同成品從自己的隨機起點沿best權重鏈接續；新段init重建求解/抽樣狀態，同段resume還原latest狀態。各階段學習/更新範圍明列，native完成4000選1000由已執行離線紀錄讀取接回19.1發布步數。',
 '19.5':'SFT以條件改變就改答案的完整示範教格式、澄清、工具狀態和有限範圍。模型生成、作者訓練答案、手寫標籤檢查三者清楚分開，assistant文字與EOS才算作答目標；下一頁明說僅替換材料。'
}
canonical=[]
for page in order:
 e=entries[page]
 canonical.append({'page_id':page,'source':e['source'],'selector':e['selector'],'source_sha256':e['source_sha256'],'figures_sha256':e['figures_sha256'],'role':'primary' if page in primary else 'context','verdict':'pass','own_summary':summary[page],'trace_sequence':traces[page]['sequence'],'raw_utf8_snapshot':str(base/'source-pages'/f'{page}.md')})
visual=[]
observations={
 '18.6':'貓/狗身份、教師與學生ID交叉對齊、[1,0]重排與MAE0.6→0可讀；圖置於身份解釋和程式之間。',
 '18.13':'A0/A1/A2各自配15/3、16/4、17/5，與正文前一格位移一致；圖置於位置解釋和程式之間。',
 '19.1':'文字歷史、商品、指定字卡、實際錄音四框匯到同一MoE核心，工具真結果交回核心；限定三類商品與連續1–4格也可讀。',
 '19.3':'來源A的原圖/配對/換位同框，B用於選檢查點、C定版後核對，圖與三份資料用途表一致。',
 '19.4':'從隨機起點到八個阶段的縱向箭頭，以及每箭頭载best/新段重建狀態均可讀；圖在鏈引入與阶段用途表之間。'
}
for page in observations:
 for r in json.loads((base/'browser'/f'{page}-capture.json').read_text()):
  for path in r['viewed_files']:
   assert Path(path).is_file()
  visual.append({**r,'actually_viewed_by_this_reviewer':True,'verdict':'pass','readability':'桌機與手機原始viewport可辨識主要標籤與限定，無需點放大完成核心理解。','position_and_caption':observations[page],'capture_timing_note':'桌機分兩個實際viewport滾動查看長圖；第二張上部已捲出是正常閱讀滾動。判斷以完整actual viewport為準，沒有用element screenshot的header遮擋判成layout缺陷。'})
transitions=[]
bridge={
 ('18.12','18.13'):'從相同候選的輸出介面轉到相同答案的預測列；新模態前文長度與一格位移在例子前交代。',
 ('18.13','18.14'):'蒸餾對齊後另談量化成本，以四版分離方法效果；不把模態列數或KL當儲存/速度成果。',
 ('18.14','chapter-19'):'前章是可選壓縮支線，導言以具名V2已訓練公開成品重新建立任務；與隨機成本模型、屬性蒸餾學生有辨識界線。',
 ('chapter-19','19.1'):'導言承諾先看真正回答，由歷史兩點格式與App當前問題的CPU例子接住，再展開三入口與驗收表。',
 ('19.1','19.2'):'圖中同一MoE核心下一頁立即教FFN/expert/router，說明為何採MoE、保留Dense以及比較限制。',
 ('19.2','19.3'):'結構計數留能力給生成考題；下一頁先教家族隔離，連回19.1同素材換位題和有限字/聲音範圍。',
 ('19.3','19.4'):'先固定資料與驗證用途，再固定權重傳承與選點；best與latest的差異直接交代，不靠同名檔判定。',
 ('19.4','19.5'):'19.4最後明說先看SFT完整示範，19.5由相同App行為目標轉到條件變化的作者示範與labels/EOS安排。'
}
for before,after in zip(order,order[1:]):
 t=traces[after]
 transitions.append({'from_page':before,'to_page':after,'verdict':'pass','own_transition_summary':bridge[(before,after)],'five_point_first_read_record':t['five_point_understanding'],'recorded_trace_sequence':t['sequence'],'necessary_revision':False})
version=json.loads((base/'version-check.json').read_text());supp=[]
for page in ['18.6','18.8']:
 p=base/'source-pages'/f'{page}.md';url=f'http://127.0.0.1:8793/{page}.html'
 with urllib.request.urlopen(url,timeout=20) as r:html=r.read();status=r.status
 htmlpath=base/'browser'/f'{page}.html';htmlpath.write_bytes(html)
 s={'page_id':page,'linked_from':'18.12首段','read_after':'18.12','read_before':'18.13','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_utf8_snapshot':str(p),'actual_range':'本節全部正文、程式、練習及存在的details；沒有讀18.7或其他未循必要連結的頁面。','preview_url':url,'preview_status':status,'preview_html_sha256':hashlib.sha256(html).hexdigest(),'figures_sha256':traces[page].get('figures_sha256',{}),'resolution':traces[page]['question_resolution']}
 for source,sha in s['figures_sha256'].items():
  urlfig='http://127.0.0.1:8793/figures/'+Path(source).name
  with urllib.request.urlopen(urlfig,timeout=20) as r:b=r.read()
  assert hashlib.sha256(b).hexdigest()==sha
 supp.append(s)
physical=[r['page_id'] for r in trace if r['event'] in ['page_first_read','linked_prerequisite_read']]
assert [p for p in physical if p in order]==order
assert version['all_pass']
report={'schema_version':1,'reviewer_task':'a','reviewer_identity':{'task_name':'/root/p6_continuity_a','round':'fresh third continuity reading','reader_background':'數學不錯的高中生／基本數學大學生，知道入門Python；不以模型專家背景填補本文未教步驟。','fresh_reader':True,'not_author_first_reader_or_technical_reviewer':True,'read_other_review_reports':False,'delegated_agents':False},'verdict':'pass','finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary':primary,'actual_order':order,'actual_physical_order_including_linked_backreads':physical,'order_note':'actual_order記指定九頁的實際順序；18.12後在進18.13前循明確必要連結補讀18.6/18.8，實體順序另列。每頁五項理解均已當時追加trace，未全看後倒填。','canonical_pages':canonical,'supplementary_reads':supp,'source_versions':{'inventory':'docs/course-revision-20261006-phase6/continuity/inventory.json','inventory_sha256':version['inventory_sha256'],'assignment_group':'a','canonical_source_and_figures_version_check':str(base/'version-check.json'),'all_inventory_source_and_figure_hashes_match':True,'current_canonical_contains_saved_section_verbatim':True,'actual_preview_served_figure_hashes_match':True,'raw_source_encoding':'UTF-8'},'transitions':transitions,'issues':[{'id':'a-optional-18.14-PTQ','page_id':'18.14','severity':'optional','necessary_revision':False,'quote':'再對同一份權重做PTQ','own_question':'PTQ在本頁未展開；先訓練保存驗證、再量化的流程與量化連結已能支持理解。','suggestion':'可在此寫「訓練後量化（PTQ）」方便沒有記住縮寫的讀者。','impact':'不影響蒸餾/量化分工或章界，不阻擋pass。','first_recorded_trace_sequence':5}],'trace_file':'docs/course-revision-20261006-phase6/continuity/traces/a.jsonl','visual_checks':visual,'notebook_checks':[{'page_id':r['page_id'],**r['notebook_read']} for r in trace if r.get('notebook_read')],'repetition_and_terminology':{'verdict':'pass','own_observation':'19.1與19.5重用短App例子，前者建立成品接口，後者轉到怎樣教條件化回答，重複有明確用途。18章的成本/品質限制對應不同當節機制；19.4操作細節在details，不打斷主線。核心、入口、接頭、資料家族、驗證選點在本組有一致對象。'},'unverified':['未重新執行19.1/19.5公開權重CPU生成命令；其Notebook只有相應seed/SVG或手寫標籤示範，不能代替推論驗證。','未重新運行素材去重/家族隔離程式，未逐檔重驗真實父checkpoint與訓練執行紀錄；只實讀本文及已執行Notebook中的離線紀錄讀取輸出。','沒有重訓、沒有重新執行18章蒸餾或量化品質實驗；它們的敘述僅作銜接所需的證據種類辨識。','未讀本組以外的後續正文如19.6/19.12；只記錄本頁明確交接句，不宣稱那些頁面也通過。'],'necessary_revisions':[],'scope_complete':True}
path=root/'reports/a.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(path),'verdict':report['verdict'],'canonical_pages':len(canonical),'trace_entries':len(trace),'physical_order':physical,'visual_checks':len(visual),'actually_viewed_screenshots':sum(len(r['viewed_files']) for r in visual),'notebooks_read':len(report['notebook_checks']),'version_check_all_pass':version['all_pass']},ensure_ascii=False,indent=2))
