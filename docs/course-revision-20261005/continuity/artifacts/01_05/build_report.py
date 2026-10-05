import json,hashlib
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/course-revision-20261005/continuity'
ART=BASE/'artifacts/01_05'
assignment=json.loads((BASE/'progress.json').read_text())['assignments']['01_05']
inventory={x['page_id']:x for x in json.loads((BASE/'inventory.json').read_text())['pages']}
trace=[json.loads(x) for x in (BASE/'traces/01_05.jsonl').read_text().splitlines()]
pageviews=[json.loads(x) for x in (ART/'pageviews.jsonl').read_text().splitlines()]
for v in pageviews:
    v['screenshot']=str(Path(v['screenshot_absolute']).relative_to(ROOT))
    v['screenshot_sha_verified']=hashlib.sha256(Path(v['screenshot_absolute']).read_bytes()).hexdigest()==v['screenshot_sha256']
figchecks=json.loads((ART/'figure-version-checks.json').read_text())
html_manifest=json.loads((ART/'rendered-page-manifest.json').read_text())
issues=[]
for line,r in enumerate(trace,1):
    for d in r['doubts']:
        issues.append({
            'issue_id':f'01_05-{len(issues)+1:02d}',
            'page_id':r['page_id'],'section':r['section'],
            'first_recorded_at_utc':r['recorded_at_utc'],'trace_line':line,
            'source_sha256':r['source_sha256'],'figures_sha256':r['figures_sha256'],
            'original_quote':d['quote'],'first_understanding':d['understanding'],
            'missing_bridge':d['missing_bridge'],'severity':d['severity'],
            'later_clarification':d.get('resolved_later'),
            'revision_recheck':'尚未修訂或交回複查；原問題保留。'
        })
for i in issues:
    if i['page_id']=='1.12':
        i['later_clarification']={'page_id':'1.14','trace_line':next(n for n,r in enumerate(trace,1) if r['page_id']=='1.14'),'location':'既有接字表實驗的12篇顏色／形狀欄位短文、9篇訓練段落','extent':'補上素材類型；完整9篇與T.3配方未在本範圍讀，原初疑未刪。'}
        i['classification']='當時未解問題，後文部分釐清；不要求此段另加完整配方。'
    if i['page_id']=='5.8':
        i['later_clarification']={'page_id':'6.1','trace_line':next(n for n,r in enumerate(trace,1) if r['page_id']=='6.1'),'location':'UTF-8序列與碼點／byte／token長度比較','extent':'知道一byte不等於一字；5.8首次讀時仍未知道此關係。'}
    if i['page_id']=='5.9':
        i['later_clarification']={'page_id':'6.1','location':'文字儲存與拆分單位比較','extent':'只補byte與文字長度關係；FP32與dtype仍未解釋。'}
required=[i for i in issues if i['severity']=='增加理解負擔']
required_by_page={i['page_id'] for i in required}
recommendations={
 '2.5':'把圖中圓／方與「答案」標籤放在一起，或直接寫「答案：圓／方」。目前答案標籤在最右「是」上方，圓／方卻在最左，讓初看者容易把是當答案。',
 '5.1':'把「梯度為正」改成「梯度大小大於0」，直接沿用上段.grad.norm()定義，區分導數符號與大小。',
 '5.9':'在首次比較儲存成本前簡短定義byte、FP32與dtype，讓格數乘四的單位有依據；6.1的文字拆分解釋不能代替此處數值格式說明。',
 'readme':'README入口改用1.1新版兩排編號圖，或為舊圖提供正常正文大小可讀的手機版；目前手機字表與兩行說明讀來吃力。'
}
for i in required:i['suggested_revision']=recommendations[i['page_id']]
pages=[]
for p in assignment['primary_page_ids']+assignment['context_after']:
    rows=[(n,r) for n,r in enumerate(trace,1) if r['page_id']==p]
    meta=inventory[p]
    pv=[v for v in pageviews if v['page_id']==p]
    pi=[i for i in issues if i['page_id']==p]
    status='revise' if p in required_by_page else 'pass'
    pages.append({
        'page_id':p,'title':meta['title'],
        'coverage_role':'primary' if p in assignment['primary_page_ids'] else 'context_after',
        'source_snapshot':meta['snapshot'],'source_sha256':meta['source_sha256'],
        'source_snapshot_sha_verified':all(r['source_sha_verified'] for _,r in rows),
        'figures_sha256':meta['figures_sha256'],
        'actual_trace_lines':[n for n,_ in rows],
        'boundary_understanding':[{'section':r['section'],'understood_task':r['understood_task'],'prior_basis':r['prior_basis'],'next_question':r['next_question'],'changes':r['changes'],'boundary':r.get('boundary','此筆是已讀頁面較晚補查輸出，不是新節交接；詳見首次紀錄。')} for _,r in rows],
        'issue_ids':[i['issue_id'] for i in pi],
        'decision':status,
        'reason':recommendations[p] if status=='revise' else '；'.join(dict.fromkeys(r['boundary'] for _,r in rows if r.get('coverage_role')!='supplemental_revisit')),
        'figure_pageviews':pv,
        'figure_visual_check':'實際桌機與手機截圖都已view_image；判斷詳見逐節紀錄。' if meta['figures_sha256'] else '無內嵌必要圖；已逐節回答是否需素材或圖。非圖頁完整響應版面未另做瀏覽器驗證。',
        'unverified':[x for _,r in rows for x in r.get('unverified',[])]
    })
first_order=list(dict.fromkeys(r['page_id'] for r in trace))
report={
 'schema_version':1,'stage':'independent_continuity','assignment_id':'01_05',
 'reviewer_task':'/root/continuity_01_05','fork_turns':'none',
 'reader_background':'入門Python與高中數學；陌生模型概念只用已讀正文解釋，不用專家知識補缺步驟。',
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'decision':'revise','overall_reason':'62主頁與指定3接續頁均已初讀，主要學習路線能接上；2.5圖答案位置、5.1梯度符號／大小措辭、5.9數值儲存術語、README手機圖說需修訂後實際複查。',
 'primary_page_ids':assignment['primary_page_ids'],'context_before':assignment['context_before'],'context_after':assignment['context_after'],
 'coverage':{'primary_assigned':62,'primary_read':62,'context_after_read':3,'trace_units':len(trace),'actual_screenshots_viewed':sum(v['viewed'] for v in pageviews),'embedded_figure_sources_verified':len(figchecks)},
 'actual_read_order':first_order,
 'actual_section_read_order':[{'trace_line':n,'recorded_at_utc':r['recorded_at_utc'],'page_id':r['page_id'],'section':r['section'],'coverage_role':r.get('coverage_role','primary' if r['page_id'] in assignment['primary_page_ids'] else 'context_after')} for n,r in enumerate(trace,1)],
 'additional_previous_reading':[],
 'methodology':{
   'instructions':['docs/review-tools/continuity-reviewer-instructions.md','.agents/skills/clear-tutorial/references/review-protocol.md'],
   'incremental_recording':'指南與操作頁按各##小節逐節實讀、記錄後開下一節；各教材小節按指定序列，章導讀先於章內小節。',
   'independence':'沒有讀作者筆記、其他讀者／技術報告、實作程式或外部解釋；共同檔案系統不是技術隔離，本報告不聲稱無法存取那些檔案。',
   'exception':'首次index記錄因trace目錄不存在寫入失敗，當次工具接著已顯示course開場；修復後才留下實際UTC紀錄，index原紀錄載明此事，沒有偽造較早時間。此小範圍順序例外保留供協調者判斷，未提前看後續章節全文。',
   'supplemental_revisits':'全範圍初讀後補看1.1真CPU輸出；first-steps未插獨立CPU輸出塊，只讀正文预期結果。補查都有較晚新紀錄，沒有改寫初讀。',
   'browser':'/usr/bin/chromium與Playwright；1280x900及390x844；no-sandbox/disable-dev-shm-usage/disable-background-networking；goto domcontentloaded與非全頁screenshot timeout15000。圖只按正文正常大小查看，長桌機圖上下分段實看，沒有放大SVG代替。'
 },
 'boundary_reviews':[
  {'from':'入口／指南／暖身','to':'chapter-01與1.1','understanding':'猜下一字的動機先交代，ID來回還原先於接字能力；W.2/W.3/W.4/W.5/W.6提供本範圍實際用到的Python、軸、機率、配方與求導背景。','decision':'pass'},
  {'from':'1.14／1.15','to':'chapter-02與2.1','understanding':'顏色與形狀同末字等號導致一字表不能分欄；新章先改中間特徵再接多字線索。Embedding名稱相同而用途變，2.1開場已說清。','decision':'pass'},
  {'from':'2.5','to':'chapter-03與3.1','understanding':'窗口先確保顏色線索可見，注意力再解決內容選讀；拼接與按比例混合用途不同。正文能接，2.5圖答案位置需修。','decision':'revise'},
  {'from':'3.7','to':'chapter-04與4.1','understanding':'多頭取回與拼回表示已完成；第4章把位置、殘差、LN與FFN接成模型，未假定隨機結果就是學會語言。','decision':'pass'},
  {'from':'4.7／4.8','to':'chapter-05與5.1','understanding':'資料→模型→loss→梯度通路先完成而沒有更新；第5章新固定題真正四步更新，圖與aux0不混答案loss。5.1一句梯度為正需改成大小為正。','decision':'revise'},
  {'from':'5.8／5.9／5.16／5.17','to':'chapter-06／6.1／6.2（context）','understanding':'評估與成本要求固定文字單位，6章改問每個模型位置代表什麼；6.1才補byte與字/token的關係，未補5.9的FP32/dtype。BPE合一次常見片段跟得上Counter與shift前文。','decision':'revise','limitation':'6章僅指定導讀、6.1、6.2，不計01_05主覆蓋；沒有讀6.3以後。'},
  {'from':'README／environment入口','to':'閱讀與執行選路','understanding':'前自訓小模型、19章尚待新成品與20成熟模型應用範圍分明，下載／執行通路與任務品質分開；README舊圖手機說明需改善。','decision':'revise'}
 ],
 'required_revision_issue_ids':[i['issue_id'] for i in required],
 'issues':issues,'page_reviews':pages,
 'actual_pageviews':pageviews,
 'source_figure_version_checks':figchecks,
 'rendered_page_snapshots':html_manifest,
 'trace_path':'docs/course-revision-20261005/continuity/traces/01_05.jsonl',
 'unverified_scope':[
  '沒有執行教材程式、改例練習、GPU訓練或重新驗算實驗報告；網站CPU輸出是閱讀材料，不算我執行的測試。',
  'Colab登入與整份執行、macOS／Windows／MPS／CUDA安裝、Jupyter kernel操作、下載資料與權重、部署與外部帳戶未實際驗證。',
  '實際瀏覽器圖檢查限本範圍19份內嵌圖及51張截圖；其他非圖頁的全版面、全部連結、深色模式、搜尋、Colab開啟與所有輸出塊佈局未驗證。',
  '2.5補充連結window_training.svg未另開；正文表已足以判讀當節，該選讀曲線不列視覺通過。',
  '正文連到T.3/T.4或實驗原始報告只作來源路標，未打開範圍外操作頁、作者筆記／研究紀錄或實作。',
  '本輪只是受限背景AI讀者的銜接判斷，沒有以程式可跑、網站建置或技術查證取代理解；未有真人目標讀者參與。',
  '未修訂後重讀，四個revise頁與其受影響邊界尚未通過；原初疑即使後文釐清仍保留。'
 ],
 'writes_scope':'只寫01_05指定artifact目錄、trace及report；沒有修改教材、來源快照或別人紀錄，沒有派其他agent。'
}
(BASE/'reports').mkdir(exist_ok=True)
(ROOT/assignment['report_path']).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
assert all(p in first_order for p in assignment['primary_page_ids'])
assert all(v['viewed'] and v['screenshot_sha_verified'] for v in pageviews)
assert all(c['matches'] and c['snapshot_sha256']==c['expected_sha256'] for c in figchecks)
assert all(r['source_sha_verified'] for r in trace)
print(json.dumps({'report':assignment['report_path'],'decision':report['decision'],'primary':62,'context':3,'trace_units':len(trace),'screenshots':len(pageviews),'required_issue_ids':report['required_revision_issue_ids'],'revision_pages':sorted(required_by_page)},ensure_ascii=False))
