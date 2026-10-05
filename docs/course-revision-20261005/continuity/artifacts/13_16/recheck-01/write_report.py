import json,pathlib,hashlib,datetime,collections
R=pathlib.Path('/workspace/tiny-perceptron-vlm'); B=R/'docs/course-revision-20261005/continuity'; A=B/'artifacts/13_16/recheck-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((B/'reports/13_16.json').read_text()); invpath=B/'revised-02/inventory.json'; inv=json.loads(invpath.read_text()); meta={p['page_id']:p for p in inv['pages']}
trace=[json.loads(l) for l in (B/'traces/13_16-recheck-01.jsonl').read_text().splitlines()]; views=[json.loads(l) for l in (A/'pageviews.jsonl').read_text().splitlines()]
initial_results={p['page_id']:p for p in old['primary_page_results']}; actual=collections.defaultdict(list)
for t in trace:
 if t['event']=='actual_reread': actual[t['page_id']].append(t)
changed=['13.12','13.14','16.3','16.4','16.5','16.8','16.12','curriculum']
def clean(t):return {k:v for k,v in t.items() if k!='original_text'}
results=[]
for pid in old['primary_page_ids']:
 p=meta[pid]; q=initial_results[pid]; unchanged=p['source_sha256']==q['source_sha256'] and p['figures_sha256']==q['figures_sha256']
 if pid not in actual:
  assert unchanged and q['status']=='pass',pid
  basis='unchanged_inherited';decision=q['status'];reason='沿用本人真首次閱讀：來源與所有圖SHA相同，本次未再次全文讀此頁。原理解：'+' / '.join(q['pass_or_revise_reason'])
 else:
  basis='actual_reread_changed_primary' if pid in changed else 'actual_reread_unchanged_boundary_context'; decision='pass' if all(t['status']=='pass' for t in actual[pid]) else 'revise';reason=' / '.join(t['understanding'] for t in actual[pid])
 results.append({'page_id':pid,'title':p['title'],'source':p['source'],'selector':p['selector'],'source_snapshot':p['snapshot'],'source_sha256':p['source_sha256'],'figures_sha256':p['figures_sha256'],'decision':decision,'status':decision,'basis':basis,'reason':reason,'same_text_and_figures_as_first_reading':unchanged,'initial_source_sha256':q['source_sha256'],'initial_figures_sha256':q['figures_sha256'],'initial_status':q['status'],'initial_report_path':'docs/course-revision-20261005/continuity/reports/13_16.json','current_reading_records':[clean(t) for t in actual.get(pid,[])]})
issue_results=[]
for issue in old['issues']:
 checks=[dict(c,page_id=t['page_id'],rechecked_at=t['recorded_at'],source_sha256=t['source_sha256'],figures_sha256=t['figures_sha256']) for t in trace for c in t.get('issue_rechecks',[]) if c['issue_id']==issue['id']]
 assert checks and all(c['decision']=='pass' for c in checks)
 issue_results.append({'issue_id':issue['id'],'original_issue':issue,'original_report_preserved':True,'current_rechecks':checks,'current_decision':'pass','required_change_remaining':False,'optional_change_remaining':False})
preservation=json.loads((A/'original-preservation.json').read_text())
for f in preservation['files']:
 p=R/f['path'];assert sha(p)==f['sha256'] and p.stat().st_size==f['bytes'],f['path']
for pid in old['primary_page_ids']+list(actual):
 assert sha(R/meta[pid]['snapshot'])==meta[pid]['source_sha256'],pid
for t in trace:
 assert t['source_sha256']==meta[t['page_id']]['source_sha256']
 assert t['figures_sha256']==meta[t['page_id']]['figures_sha256']
for prev,current in zip(trace,trace[1:]):assert prev['recorded_at']<=current['recorded_at']
for v in views:
 assert v['status']==200
 for s in v['screenshots']:assert sha(pathlib.Path(s['path']))==s['sha256']
assert ''.join(t['original_text'] for t in actual['curriculum'])==(R/meta['curriculum']['snapshot']).read_text()
actual_changed=[p['page_id'] for p in results if not p['same_text_and_figures_as_first_reading']];assert actual_changed==changed
http_images=json.loads((A/'http-image-hashes.json').read_text())
report={
 'schema_version':1,
 'stage':'independent_continuity_recheck',
 'phase':'Phase 4',
 'assignment':'13_16',
 'recheck_id':'recheck-01',
 'reviewer_task':'/root/continuity_13_16',
 'fork_turns':'none',
 'identity_statement':'本人是以fork_turns none建立的原第三輪13_16學生讀者，此回讀由同一canonical agent執行，非新盲讀。未讀作者修復筆記、其他審閱報告、技術判斷或來源實作。',
 'reader_background':'入門Python、高中基本數學；以本人真初讀已學素材理解，不以專家知識填補正文空白。',
 'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'pass',
 'current_verdict':'pass',
 'verdict_reason':'八個改動primary頁及必要前後文實際回讀通過；其餘primary以本人原真初讀和相同文字/圖SHA繼承。原三項增加理解負擔與三項用字建議都有實際當地修正可讀證據，沒有必改或可選剩餘問題。',
 'must_fix':[],
 'optional':[],
 'source_inventory_path':str(invpath.relative_to(R)),
 'source_inventory_sha256':sha(invpath),
 'method_paths':['docs/review-tools/continuity-reviewer-instructions.md','.agents/skills/clear-tutorial/references/review-protocol.md'],
 'method_sha256':{p:sha(R/p) for p in ['docs/review-tools/continuity-reviewer-instructions.md','.agents/skills/clear-tutorial/references/review-protocol.md']},
 'original_report_path':'docs/course-revision-20261005/continuity/reports/13_16.json',
 'original_report_sha256':sha(B/'reports/13_16.json'),
 'original_trace_path':'docs/course-revision-20261005/continuity/traces/13_16.jsonl',
 'original_trace_sha256':sha(B/'traces/13_16.jsonl'),
 'original_bytes_preservation':{'manifest_path':str((A/'original-preservation.json').relative_to(R)),'manifest_sha256':sha(A/'original-preservation.json'),'file_count':len(preservation['files']),'all_original_report_trace_artifact_bytes_unchanged':True,'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'原report、原trace與原artifacts所有88檔；新recheck-01目錄從manifest範圍排除。'},
 'trace_path':'docs/course-revision-20261005/continuity/traces/13_16-recheck-01.jsonl',
 'trace_sha256':sha(B/'traces/13_16-recheck-01.jsonl'),
 'artifacts_directory':str(A.relative_to(R)),
 'changed_primary_page_ids':changed,
 'actual_reread_page_ids':list(actual),
 'actual_reread_primary_page_ids':[p['page_id'] for p in results if p['basis']!='unchanged_inherited'],
 'unchanged_inherited_primary_page_ids':[p['page_id'] for p in results if p['basis']=='unchanged_inherited'],
 'actual_read_order':[dict(sequence=i+1,**clean(t)) for i,t in enumerate(trace)],
 'primary_page_ids':old['primary_page_ids'],
 'primary_page_count':len(results),
 'primary_page_results':results,
 'reread_primary_count':17,
 'unchanged_inherited_primary_count':41,
 'actual_reread_unique_pages':18,
 'actual_reread_units':sum(len(v) for v in actual.values()),
 'context_actual_source_sha256':{pid:{'source_sha256':meta[pid]['source_sha256'],'figures_sha256':meta[pid]['figures_sha256'],'records':[clean(t) for t in ts]} for pid,ts in actual.items() if pid not in changed},
 'original_assignment_context':{'statement':'初次四頁context_before、三頁context_after與補讀4.5仍保存；這次只把chapter-17作必要接續實讀，其餘未再次閱讀。首次有效context的當前文字/圖SHA是否相同另外列明。','pages':[{'page_id':p['page_id'],'source_sha256':meta[p['page_id']]['source_sha256'],'figures_sha256':meta[p['page_id']]['figures_sha256'],'basis':'actual_reread' if p['page_id'] in actual else 'initial_reader_record_with_hash_comparison','same_text_and_figures_as_first_reading':meta[p['page_id']]['source_sha256']==p['source_sha256'] and meta[p['page_id']]['figures_sha256']==p['figures_sha256']} for p in old['context_before']+old['context_after']+old['supplementary_context']]},
 'issue_rechecks':issue_results,
 'additional_format_rechecks':[dict(page_id=t['page_id'],recorded_at=t['recorded_at'],source_sha256=t['source_sha256'],**t['format_recheck']) for t in trace if t.get('format_recheck')],
 'handoffs':[
  {'pages':['13.11','13.12','13.13','13.14','13.15'],'decision':'pass','reason':'固定優勢→本輪old機率比→裁切方向→reward/critic與固定reference→一次收集更新，回讀仍按不同角色/時間接上。字修與Python框沒有讓梯度示範成已更新/已答對證據。'},
  {'pages':['16.2','16.3','16.4','16.5','16.6'],'decision':'pass','reason':'16.3現在明說沿用工作分工而縮為3+1，不误称16.2同4+1材料。16.4品質表前就近說屬性題材、7.13正文位置與非計時英文開頭，16.5 shape?僅具体QA驗收例。packing改任務、梯度累积保份量各有通知。'},
  {'pages':['16.7','16.8','16.9'],'decision':'pass','reason':'浮點格式與容差→手寫/SDPA同縮放mask→Flash分塊仍全候選，條字修正不改語義；只核心補驗與整模型實測分開，不由後端名稱推能力/訓練成果。'},
  {'pages':['16.11','16.12','16.13'],'decision':'pass','reason':'先量執行/編譯成本，再看真正改可見距離的滑窗，最後三種可見配對與計算安排分開。Python fence实際渲染和3→5→7圖仍可分本層直接/跨層可能傳遞。'},
  {'pages':['16.13','curriculum','chapter-17'],'decision':'pass','reason':'總覽按問題與支線導航。新版竖向流程圖在390手機正文能辨原樣本→角色/編號→X/Y、讀mask與計分mask、zero/backward/step；和16.6刻意累积条件分开。量化導讀再從bytes與細節損失開始，未借未讀19/20結果補已學機制。'}
 ],
 'browser_method':{'actual_site':'http://127.0.0.1:8765/<page_id>.html','browser':'/usr/bin/chromium via Playwright','args':['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'],'goto_wait_until':'domcontentloaded','goto_timeout_ms':15000,'screenshot_full_page':False,'screenshot_timeout_ms':15000,'viewports':[{'width':1280,'height':900},{'width':390,'height':844}],'all_successfully_logged_screenshots_viewed_with_view_image':True,'visual_source_fingerprints':'实际HTTP图bytes SHA另核与revised-02对应，未用此hash核对代替view_image看图。'},
 'pageviews_path':str((A/'pageviews.jsonl').relative_to(R)),
 'pageviews_sha256':sha(A/'pageviews.jsonl'),
 'pageviews':views,
 'successful_logged_pageview_count':len(views),
 'viewed_screenshot_count':sum(len(v['screenshots']) for v in views),
 'current_http_image_sha256_checks':http_images,
 'current_http_image_hashes_path':str((A/'http-image-hashes.json').relative_to(R)),
 'current_http_image_hashes_file_sha256':sha(A/'http-image-hashes.json'),
 'capture_tool_issue':{'path':str((A/'capture-errors.jsonl').relative_to(R)),'sha256':sha(A/'capture-errors.jsonl'),'statement':'13.14追加代码框截圖helper初次假設pre含nested code，metadata查詢超時；實際desktop導航/截圖已發生但當次未成功寫pageviews，故不列為完整成功pageview证据。失败截圖原样另存且未使用作可读性判断。修helper後對實際Python框重新擷取並view_image；這不是教材失敗。'},
 'unverified_scope':[
  '本次不是又全文讀58頁：17個primary頁真回讀，其中八改動頁與九未變邊界頁；41頁僅沿用本人初讀與同文字/圖SHA。未聲稱那些頁有本次新瀏覽器或新視覺驗證。',
  '實讀必要非primary接續僅chapter-17；17.1、17.2、四個原context_before與4.5補讀本次沒有再讀。沒有宣稱完整1–12或17以後已讀。',
  'curriculum全份15單位逐讀逐記：六個主要unit，unit2主段/nested路線分開，unit5標題與八個###各自讀/記；沒有按外連參考內容補理解。原首次同bundle揭露限制仍保留，本次回讀不冒稱新盲讀。',
  '未讀作者repair notes、其他讀者/正確性報告、實作或外部原論文/技術報告；當頁公開選讀正文已讀，但未跟其證據連結。',
  '無新CPU程式執行、GPU、訓練、量測、能力實驗或閱讀時間估計。實站CPU框只作閱讀/格式證據，不作本人新執行或獨立技術查證。',
  '只在1280×900与390×844看必要图/兩处Python框，未驗所有viewport、互動、所有像素或每個CPU輸出框。原图SHA相同并不自动建立新視覺实看；有本次實看的各自列新截图與URL。',
  '13.14截图helper一次metadata定位失敗有單獨紀錄，未倒填未知导航时戳或把失败截圖说成view_image看過。'
 ]
}
assert report['reread_primary_count']==len(report['actual_reread_primary_page_ids'])
assert report['unchanged_inherited_primary_count']==len(report['unchanged_inherited_primary_page_ids'])
assert len(results)==58 and all(p['decision']=='pass' for p in results)
out=B/'reports/13_16-recheck-01.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report_path':str(out),'report_sha256':sha(out),'trace_path':report['trace_path'],'trace_sha256':report['trace_sha256'],'current_verdict':report['current_verdict'],'primary':58,'actual_reread_primary':17,'inherited_primary':41,'actual_unique_pages':18,'actual_units':report['actual_reread_units'],'trace_records':len(trace),'successful_logged_pageviews':len(views),'viewed_screenshots':report['viewed_screenshot_count'],'actual_http_figures':len(http_images),'original_88_files_preserved':True},ensure_ascii=False))
