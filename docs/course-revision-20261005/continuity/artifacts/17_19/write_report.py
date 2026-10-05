import datetime, hashlib, json
from pathlib import Path

root=Path('/workspace/tiny-perceptron-vlm')
base=root/'docs/course-revision-20261005/continuity'
assignment=json.loads((base/'progress.json').read_text())['assignments']['17_19']
inventory={p['page_id']:p for p in json.loads((base/'inventory.json').read_text())['pages']}
trace=[json.loads(line) for line in (base/'traces/17_19.jsonl').read_text().splitlines()]
views=[json.loads(line) for line in (base/'artifacts/17_19/pageviews.jsonl').read_text().splitlines()]
issues=[]
for item in trace:
    for gap in item.get('gaps',[]):
        if gap['id'].endswith('-first'): continue
        issues.append(dict(gap,page_id=item['page_id'],unit=item['unit'],first_recorded_at=item['recorded_at'],source_sha256=item['source_sha256'],figures_sha256=item['figures_sha256'],status='unresolved',decision='revise'))
for issue in issues:
    if issue['id']=='C1715-01':
        issue['later_clarification']='T.4釐清固定sft/model.pt與CPU attributes.pt不是同一父檔；T.9釐清固定比較另做120更新，但仍未在17.15提供所預期的三版品質結果。'
    if issue['id']=='CT10-01':
        issue['initial_question_recorded_at']=next(t['recorded_at'] for t in trace if t['page_id']=='training' and t['unit']=='T.10')
        issue['supplemental_read_scope']=['12.12 entire lesson including figure and review links text; linked reports/code not opened']
        issue['scope_limit']='局部準備入口的依賴指称/命令缺橋梁，並非宣稱第12章未教joint機制。'

figure_observations={
 '16.12':'兩viewport的位置、層與3→5→7箭頭可讀，包含自己三格的規則對得上。',
 '16.13':'手機整張三配對表可讀；桌機自然捲動上/下兩畫面，藍灰格及第三組只讀自己的圖可辨。',
 '17.2':'0.7/1.2除刻度、取整、還原與差0.2在兩viewport正文可讀。',
 '17.8':'第一碼低半、第二碼高半、112與拆回順序在兩viewport正文可讀。',
 '17.14':'手機全流程可讀，桌機上/下畫面核對最後打包與本短式未更新的說明。',
 '18.5':'三條100%比例與固定候選顏色在兩viewport可讀，次要份量增加而第一名不變。',
 '18.6':'教師貓ID0配學生貓ID1、order[1,0]及0.6→0在兩viewport可讀。',
 '18.13':'三答案框各配教師15/16/17與學生3/4/5，前一格說明在兩viewport可讀。',
 '19.1':'文字/圖片/聲音入口接同MoE與工具真結果回交，兩viewport內容可讀。',
 '19.6':'入口/出口與指定下框/紙上答案在兩viewport可讀；它是作者字卡示意非已驗收OCR。',
 '19.7':'問題/第一模型/真程式/回填/第二模型五框在兩viewport可讀，期待與真驗收界線清楚。',
 '12.12':'紅圓與440Hz示意各供circle/high，兩箭頭及完整答案在兩viewport可讀。',
 '20.1':'打字/Whisper逐字稿/照片配套處理器接Qwen無LoRA圖在兩viewport可讀，與19自訓入口不同。'
}
for view in views:
    view['actually_viewed_with']='tools.view_image'
    view['inspection']=figure_observations[view['page_id']]
    assert hashlib.sha256((root/view['screenshot']).read_bytes()).hexdigest()==view['sha256']

page_results=[]
for pid in assignment['primary_page_ids']:
    p=inventory[pid]
    reads=[t for t in trace if t['page_id']==pid and t['role']=='primary']
    assert reads
    assert hashlib.sha256((root/p['snapshot']).read_bytes()).hexdigest()==p['source_sha256']
    own_issues=[issue['id'] for issue in issues if issue['page_id']==pid]
    page_results.append({
        'page_id':pid,'source':p['source'],'source_snapshot':p['snapshot'],
        'source_sha256':p['source_sha256'],'figures_sha256':p['figures_sha256'],
        'actual_units_read':[r['unit'] for r in reads],
        'first_read_at':reads[0]['recorded_at'],'last_initial_read_at':reads[-1]['recorded_at'],
        'decision':'revise' if own_issues else 'pass','issue_ids':own_issues,
        'pass_or_revise_reason':'；'.join(r['first_understanding'] for r in reads),
        'visual_scope':'實際網站桌機與手機必要圖皆已看。' if p['figures_sha256'] else '來源逐節判斷無必要缺圖；未另渲染這頁的無圖排版。'
    })

report={
 'schema_version':1,'stage':'independent_continuity','assignment':'17_19',
 'reviewer_task':'/root/continuity_17_19','fork_turns':'none',
 'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'revise','decision':'revise','initial_read_complete':True,
 'reader_identity':{
     'role':'全新第三輪前後銜接讀者',
     'background':'高中基本數學／入門Python，只使用本次實際依序讀過的教材；不以既有模型專業知識補未教步驟。',
     'not_author_or_prior_reviewer':True,
     'sequential_reveal':'每個單位讀完先以真UTC追加trace，再取得下一單位；共同檔案系統可存取全文，這是自控順讀而非技術隔離。',
     'prohibited_material_not_read':['作者來源筆記','其他讀者報告','技術實報','程式實作','外部補充解釋']
 },
 'method_sources':['docs/review-tools/continuity-reviewer-instructions.md','.agents/skills/clear-tutorial/references/review-protocol.md'],
 'source_inventory':'docs/course-revision-20261005/continuity/inventory.json',
 'figure_inventory':'docs/course-revision-20261005/continuity/figure-snapshots.json',
 'trace_path':assignment['trace_path'],'artifacts_directory':assignment['artifacts_directory'],
 'primary_page_count':48,'primary_page_ids':assignment['primary_page_ids'],
 'context_before':assignment['context_before'],'context_after':assignment['context_after'],
 'supplemental_context':{'page_ids':['12.12'],'reason':'按18.13明確引用確認T.10多模態固定joint教師身分；沒有擴讀完整第12章。'},
 'actual_read_order':[{'page_id':t['page_id'],'unit':t['unit'],'role':t['role'],'recorded_at':t['recorded_at'],'source_sha256':t['source_sha256'],'figures_sha256':t['figures_sha256']} for t in trace if t['role']!='clarification_after_supplemental_context'],
 'unit_count':80,'trace_record_count':81,
 'primary_pages':page_results,
 'boundary_records':trace,
 'issues':issues,
 'chapter_boundary_judgments':[
   {'boundary':'16.11–16.13 → chapter-17/17.1','decision':'pass','reader_understanding':'前章優化安排/可見範圍與成本核對，17導讀明轉每數所用位元，17.1全零隔離空間再預告刻度。'},
   {'boundary':'17.14–17.15 → chapter-18/18.1','decision':'pass','reader_understanding':'從同結構表示誤差與離散選擇，轉為較小學生向教師文字/分布學；導讀與18.1直接說新方法、來源與可信度不同。','local_issue':'C1715-01不阻章18新方法入口，但17.9結果承諾需要修。'},
   {'boundary':'18.13–18.14 → chapter-19/19.1','decision':'pass','reader_understanding':'多模態對齐與壓縮實测不代表通用途；19导读直接说明新从零有限助手尚待训练，旧328128是机制/历史证据，19.1素材表是目标非新成績。'},
   {'boundary':'19.12 → chapter-20/20.1/20.2','decision':'pass','reader_understanding':'19待填能力矩阵与旧题库分开，20接Qwen/Whisper已有权重，ASR转字而非19自训声入口；LoRA未采用，CPU短例非完整能力，20.2安装只取配置再官方权重。'},
   {'boundary':'正文17–19 → training/asset-storage/validation/publishing','decision':'revise','reader_understanding':'操作指南多数分清CPU练习/固定配方/共同成品/成熟路线；T3四种指称、T10固定joint依赖、validation19摘要须补桥梁。','issue_ids':['CT3-01','CT10-01','CV2-01']}
 ],
 'optional_improvements':[
   {'page_id':'17.10','quote':'文件','severity':'選讀改善','suggestion':'與17.1等頁統一為檔案。'},
   {'page_id':'19.2','quote':'共同核心的注意力、字元嵌入與輸出表仍共享','severity':'選讀改善','suggestion':'加上expert之間，避免與輸入/輸出權重綁定混讀；本頁補充已釐清兩表各自儲存。'},
   {'page_id':'19.5 / 19.7','quote':'不是隻靠 / 不是隻差','severity':'選讀改善','suggestion':'隻改只。'},
   {'page_id':'19.11','quote':'四站FP32推論檔','severity':'選讀改善','suggestion':'可稱四份階段／分支檔，與DPO非必經站更一致；下文已明說分支，不阻理解。'}
 ],
 'figure_assessment':{
     'necessary_primary_figures':9,'all_necessary_primary_figures_actually_viewed_on_desktop_and_mobile':True,
     'additional_context_figures_viewed':['16.12','16.13','12.12','20.1'],
     'served_figure_hash_verification':'docs/course-revision-20261005/continuity/artifacts/17_19/served-figure-hashes.json',
     'served_and_snapshot_figures_all_match':True,'missing_necessary_figures':[],
     'screenshot_count':len(views),'pageviews':views,
     'basis':'判讀源於真正文截圖實際view_image，不由SVG字串、DOM尺寸或原圖放大代替。'
 },
 'repetition_and_terminology':'新舊界線與實測範圍在各節可獨立使用時適度重現，長歷史細節多置補充；未見阻斷主要理解的重複負擔。可選的文件/檔案、隻/只與共享指稱已列。',
 'unverified_scope':[
     '未執行本文程式、下載權重／資料、訓練、GPU、實測／單元測試或學生端完整安裝；網站現有CPU輸出僅作正文所展示內容。',
     '未重查知識與數字的技術正確性，未開實報或來源程式；第三輪只判銜接與理解界線。',
     '未讀完整1–16章，只讀指定16.11/12/13與按明確引用補讀12.12；不將未讀完整前章當成全書缺教。',
     '無圖頁沒有另外渲染桌機/手機排版，沒有實際操作頁的互動、搜尋、Colab或外部下載連結；必要圖正文兩viewport已驗。',
     '未驗新第19章工程成品能力，它在此版本明示尚待訓練與驗收；舊成品實測仍只是其素材/模板範圍。',
     '沒有估閱讀時間、發布、更改教材或讀寫別人的審閱紀錄。'
 ],
 'final_reader_restatement':'量化缩表示，学生架構缩数目，教師訊號改學習目標，各自品質/儲存/速度证据不同。第19新共同核心需逐入口/歷史/工具與旧能力保留驗收，旧合成实測不能补新能力表。第20成熟模型从上游权重出发、语音转文字进同聊天历史，是另一条应用路线。',
 'rereview_required':['17.9 → 17.15 affected result promise','training T.3 four-model entry','training T.6 / 18.13 / T.10 fixed joint preparation as applicable','chapter-19 / 19.12 → validation scope summary → chapter-20']
}
(base/'reports/17_19.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report_path':str((base/'reports/17_19.json').relative_to(root)),'decision':report['decision'],'primary_pages':len(page_results),'actual_units_read':report['unit_count'],'trace_records':len(trace),'issues':[i['id'] for i in issues],'screenshots':len(views)},ensure_ascii=False))
