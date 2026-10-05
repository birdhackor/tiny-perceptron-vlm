import datetime, hashlib, json, pathlib

root=pathlib.Path('/workspace/tiny-perceptron-vlm')
base=root/'docs/course-revision-20261005/continuity'
assignment=json.loads((base/'progress.json').read_text())['assignments']['10_12']
inventory={p['page_id']:p for p in json.loads((base/'inventory.json').read_text())['pages']}
trace=[json.loads(line) for line in (base/'traces/10_12.jsonl').read_text().splitlines()]
first_order=[]
for entry in trace:
    if entry['page_id'] not in first_order:
        first_order.append(entry['page_id'])

captures=[json.loads(line) for line in (base/'artifacts/10_12/pageviews.jsonl').read_text().splitlines()]
latest={capture['screenshot']:capture for capture in captures}
seen={name:entry['timestamp_utc'] for entry in trace for name in entry.get('pageviews',[])}
pageviews=[]
for path, capture in latest.items():
    path=pathlib.Path(path)
    assert path.name in seen
    assert hashlib.sha256(path.read_bytes()).hexdigest()==capture['screenshot_sha256']
    capture.update(viewed=True,view_method='functions.exec tools.view_image of the actual viewport screenshot',view_confirmed_in_trace_at_utc=seen[path.name],screenshot_path=str(path.relative_to(root)))
    pageviews.append(capture)
(base/'artifacts/10_12/pageviews-final.json').write_text(json.dumps(pageviews,ensure_ascii=False,indent=2)+'\n')

pages=[]
for page_id in first_order:
    page=inventory[page_id]
    entries=[entry for entry in trace if entry['page_id']==page_id]
    assert hashlib.sha256((root/page['snapshot']).read_bytes()).hexdigest()==page['source_sha256']
    pages.append({'page_id':page_id,'title':page['title'],'kind':page['kind'],'scope':entries[0]['scope'],'source':page['source'],'snapshot':page['snapshot'],'source_sha256':page['source_sha256'],'figures_sha256':page['figures_sha256'],'first_read_at_utc':entries[0]['timestamp_utc'],'units':entries,'pageviews':[v for v in pageviews if v['page_id']==page_id],'decision':'pass','reason':'；'.join(entry['reason'] for entry in entries if entry['scope']!='clarification')})

issues=[{
    'issue_id':'10_12-SFT-first-read-scope',
    'page_id':'11.6',
    'unit':'full_page',
    'quote':'描述對齊與問答 SFT 都可以用文字預測代價，差在材料、任務與開放參數。',
    'first_understanding':'我從句子把 SFT 理解成問答訓練的名稱，但在已讀範圍尚不知道縮寫全名。',
    'missing_bridge':'已讀前文沒有展開 SFT；沿本節回顧連結補讀 7.3 只補明有效目標與 bytes，仍未展開名稱。',
    'can_answer_main_question_from_read_content':True,
    'effect':'不阻礙本節預算比較：有效位置、本次材料與更新範圍已直接交代。這是本次部分前文章節範圍造成的名詞疑問，不能據此斷言全書沒教。',
    'severity':'選讀改善',
    'suggestion':'若此頁要支援由第 10 章跳入的讀者，可把這一次名稱寫成「問答監督式微調（SFT）」。協調者應先核對已分配的第 7 章是否已有定義，再決定是否需要改動。',
    'clarified_later_at':'7.3 補明了有效位置，不補縮寫；首次疑問保留於 trace。',
    'decision':'pass',
    'needs_required_revision':False,
    'reread_status':'not_requested'
}]
issues.append({
    'issue_id':'10_12-context-9.9-script-consistency',
    'page_id':'9.9',
    'scope':'context_before',
    'unit':'full_page',
    'quote':'文件颜色是紅，三個候選按藍、紅、绿编號0、1、2，分數2、1、0卻偏藍。',
    'first_understanding':'首次 trace 仍能理解同一錯誤最高分只因倍率變得更尖，不能取得新事實；此句的藍／紅／綠與索引對應沒有卡住。',
    'missing_bridge':'沒有概念橋梁缺漏，但作者說明同句混用颜色、绿、编與繁體；接續 9.10 及 10–12 章多用繁體，字形不一致。',
    'severity':'選讀改善',
    'suggestion':'統一作者說明為顏色、綠、編號、它們、轉等繁體寫法；實際資料字串或輸出仍保留原文。此頁是本組指定前文，可由 06_09 協調者合併處理。',
    'decision':'pass',
    'needs_required_revision':False,
    'recording_note':'這是完整初讀後整理報告時的字形觀察，沒有回寫首次概念理解。',
    'reread_status':'not_requested'
})

report={
    'schema_version':1,
    'stage':'independent_continuity',
    'assignment':'10_12',
    'reviewer_task':'/root/continuity_10_12',
    'fork_turns':'none',
    'reviewer_identity':'新的第三輪前後銜接讀者；模擬只知道入門 Python 與高中數學，不用模型專家背景補文中步驟。',
    'method_paths':['docs/review-tools/continuity-reviewer-instructions.md','.agents/skills/clear-tutorial/references/review-protocol.md'],
    'report_created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'decision':'pass',
    'status':'initial_read_complete_pass_with_optional_scope_note',
    'summary':'49 個 primary 頁面、兩頁指定前文、三頁指定接續與一頁正文連結背景已實讀。未發現阻礙主要概念的交接斷點或必要圖缺漏。各頁通過理由及首次理解在 pages.units 與不可倒填的逐節 trace；SFT 縮寫疑問屬本讀者未讀全前章的選讀改善，不是必改問題。',
    'primary_page_ids':assignment['primary_page_ids'],
    'context_before':assignment['context_before'],
    'context_after':assignment['context_after'],
    'additional_read_scope':[{'page_id':'7.3','trigger':'11.6 的回顧連結','when':'11.6 初讀紀錄之後、11.7 之前','purpose':'確認有效學習位置與 byte 計數，不追讀其他前章。'}],
    'actual_read_order':first_order,
    'actual_unit_read_order':[{'page_id':entry['page_id'],'unit':entry['unit'],'scope':entry['scope'],'timestamp_utc':entry['timestamp_utc']} for entry in trace],
    'primary_pages':[page for page in pages if page['page_id'] in assignment['primary_page_ids']],
    'pages':pages,
    'chapter_boundaries':[
        {'from':'9.10','through':'chapter-10','to':'10.1','decision':'pass','understanding':'由信心評估換到圖片輸入；導讀明說黑底紅方塊、介面與能力各需證據，10.1 真展示像素與 RGB。','reason':'新模態與新素材先宣告；9.10 補充六類圖片不是 10.1 未訓練示範的成績。'},
        {'from':'10.11','through':'chapter-11','to':'11.1','decision':'pass','understanding':'前章檢索只選候選卡，接頭尺寸合仍待完整答案。第 11 章改查圖線索是否造成正確內容。','reason':'導读接住 10.6 留的對齊問題；11.1 明分輸出敏感和生成正確，未冒認检索會生成。'},
        {'from':'11.18','through':'chapter-12','to':'12.1','decision':'pass','understanding':'由圖片範圍／讀字改到聲音振幅；合成單音先用來建立數字流，真人意圖是後段另有材料的計畫。','reason':'導讀分合成、聽寫、真人短句；12.1 明說不是人說的句子，波形真正可見。'},
        {'from':'12.16','through':'chapter-13','to':'13.1','side_reference_read_between':'training-assets（四個單位依序實讀）','decision':'pass','understanding':'有限需求的正確短答與跨輪歷史設計收束後，改問同題哪份回答更合適；先按問題與理由存偏好。','reason':'導讀宣告新訓練問題與兩條可選路；13.1 的 2+2 格式例和歷史加法正誤分清，未說直接音訊計畫已經訓練或必須接兩種偏好流程。'}
    ],
    'issues':issues,
    'required_revision_count':0,
    'unresolved_first_read_terms':[
        {'page_id':'12.8','quote':'本入口先做線性投影，再經正規化、另一線性層與非線性函數GELU','understanding':'只從本節知道 GELU 是不改時間位置的非線性函數名稱；沒有假裝會手推公式。','scope_limit':'未讀此前整個文字模型章，故不評其早先公式是否充分。逐框 16→8 的任務與軸變化可憑已讀段落完成。'},
        {'page_id':'training-assets','quote':'要在clone時先只取程式與LFS pointer','understanding':'先只取得程式與指標，後續 fetch 再取得大資料实体；pointer 未在這頁展開。','scope_limit':'未讀主 README／環境前置，也未實際安裝或下載。此處只評操作去向與資料／訓練界線。'}
    ],
    'task_and_evidence_distinctions':{
        'hand_calculations_and_data_records':'9.9–9.10 主算例、10.4/10.9–10.11、11.4 記錄、11.6 預算、11.7 人工成績、11.11 分組、11.13/11.16 字串、12.11/12.14 手寫條件、13.1 人工偏好均依正文標示，不視為訓練成果。',
        'interface_or_gradient_demonstrations':'10.1–10.8、11.1–11.3/11.5 名單、12.1–12.9 固定前處理與梯度、12.12 雙路 backward 的 shape/True 只按正文解讀，不判成能力。',
        'historical_measurements':'各節明說「既有／原實驗」的表與分數只讀為正文轉述的有限結果；未開實報或重跑。尤其 9.10 分類信心不是生成答案機率，11.8 兩端各 3/6 不是成對成功，12.10/12.12 移除輸入按原答案是依賴控制。',
        'planned_capstone':'11.17–11.18 與 12.15–12.16 的商品、字卡、真人意圖和歷史均是設計／希望答案；圖的作者手繪與色塊不是來源原圖或錄音量測。成熟第 20 章入口只讀為另外延伸。'
    },
    'visual_review':{
        'viewports':[{'width':1280,'height':900},{'width':390,'height':844}],
        'pageview_count':len(pageviews),
        'figure_count':26,
        'method':'以真網站正文 img 滾動至視窗內；高圖另保留桌機 top/bottom 視窗，full_page=False。所有列出的 70 張截圖均經 view_image 實看，未靠 alt、SVG 標籤、DOM 幾何或放大原圖判可讀。',
        'pageviews':pageviews,
        'small_text_observation':'9.10 手機補充單音圖的數字較小，但可辨長條／虛線與數字；正文先列出同數值，主校準座標圖可讀。其餘必要圖的主要標籤與資料對應可在正文尺寸讀取。',
        'capture_log_note':'原 capture JSONL 保留初次工具輸出。11.14 有兩筆同名舊截圖在補拍時被新截圖取代；本報告與 pageviews-final.json 只列目前實際保存且重看過的最新檔案及雜湊。'
    },
    'fingerprint_verification':{'source_page_snapshots_checked':55,'unique_figure_snapshots_and_live_figure_urls_checked':26,'mismatches':[],'screenshot_hashes_checked':len(pageviews)},
    'trace_path':assignment['trace_path'],
    'artifacts_directory':assignment['artifacts_directory'],
    'unverified_scope':[
        '未閱讀除 9.9、9.10、7.3 外的前章正文；不能宣稱已學會完整 tokenizer、注意力、優化器、GELU 或 SFT 先前推導。',
        '未讀作者筆記、其他 reader/reviewer 報告、技術實報、模型實作程式或外部研究補理解；正文中的連結存在不算已讀來源。',
        '沒有執行教材程式、GPU 訓練、模型生成、資料下載、安裝、時長估算、部署或帳戶操作。网站标明 CPU 输出时仅读取已发布显示，未独立重跑。',
        '沒有播放／聽辨真人錄音；12.11、12.14–12.16 明為手寫示例或計畫，不能宣稱做過真人語音能力驗收。',
        '沒有實際 Fashion-MNIST 原圖／MInDS-14 音訊、資料授權獨立查證或跨說話者測試；只核正文如何定位材料與成果。',
        '無圖頁未全部另拍桌機／手機，逐頁已回答是否需要圖；必要原圖、示意圖與波形均在兩尺寸正文真看。',
        '未審第 13 章 13.3 以後、成熟第 20 章或其操作頁；這些只知道當前正文的導覽說法。',
        '未要求或完成修正後回讀；本報告只是此快照的第三輪初讀判定。'
    ]
}
assert set(assignment['primary_page_ids'])<=set(first_order)
assert len(report['primary_pages'])==49
(base/'reports/10_12.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print({'report_path':assignment['report_path'],'primary_pages':49,'actual_read_pages':len(first_order),'trace_units':len(trace),'pageviews':len(pageviews),'decision':report['decision']})
