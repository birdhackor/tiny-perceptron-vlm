import json,pathlib,hashlib,datetime,urllib.request
R=pathlib.Path('/workspace/tiny-perceptron-vlm');B=R/'docs/course-revision-20261005/continuity';A=B/'artifacts/13_16/recheck-02'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((B/'reports/13_16.json').read_text());v2=json.loads((B/'reports/13_16-recheck-01.json').read_text());prior={p['page_id']:p for p in v2['primary_page_results']}
inv=B/'revised-03/inventory.json';meta={p['page_id']:p for p in json.loads(inv.read_text())['pages']}
trace=[json.loads(l) for l in (B/'traces/13_16-recheck-02.jsonl').read_text().splitlines()];by={t['page_id']:t for t in trace};views=[json.loads(l) for l in (A/'pageviews.jsonl').read_text().splitlines()]
def clean(t):return {k:v for k,v in t.items() if k!='original_text'}
assert [t['page_id'] for t in trace]==['16.3','16.4','16.5']
assert all(t['status']=='pass' and not t['questions'] for t in trace)
assert sha(inv)=='842270f74df245015978a85687f8d842d3d7f74f741e51f2fbac0a758ac9e827'
assert meta['16.4']['source_sha256']=='9ed82afa85398a2d362b99c86a95cc2456b7dfe9db708a9b4835aefab0f47796'
results=[]
for pid in v2['primary_page_ids']:
 p=meta[pid];q=prior[pid];same=p['source_sha256']==q['source_sha256'] and p['figures_sha256']==q['figures_sha256'];assert q['decision']=='pass'
 if pid!='16.4':assert same,pid
 assert sha(R/p['snapshot'])==p['source_sha256'],pid
 if pid in by:
  basis='actual_reread_changed_primary' if pid=='16.4' else 'same_exact_version_with_actual_context_reread';reason=by[pid]['understanding']
 else:basis='same_exact_version_inherited';reason='本人已在initial/recheck-01真閱讀並PASS；本頁文字與所有圖SHA相同，這次沒有又全文閱讀。原通過理解：'+q['reason']
 results.append({'page_id':pid,'title':p['title'],'source':p['source'],'selector':p['selector'],'source_snapshot':p['snapshot'],'source_sha256':p['source_sha256'],'figures_sha256':p['figures_sha256'],'decision':'pass','status':'pass','basis':basis,'reason':reason,'same_exact_text_and_figures_as_recheck_01':same,'previous_decision':q['decision'],'previous_source_sha256':q['source_sha256'],'previous_figures_sha256':q['figures_sha256'],'previous_report_path':'docs/course-revision-20261005/continuity/reports/13_16-recheck-01.json','actual_reread_record':clean(by[pid]) if pid in by else None})
assert sum(not p['same_exact_text_and_figures_as_recheck_01'] for p in results)==1
pres=json.loads((A/'prior-preservation.json').read_text())
for f in pres['files']:
 assert sha(R/f['path'])==f['sha256'] and (R/f['path']).stat().st_size==f['bytes'],f['path']
for v in views:
 assert v['status']==200
 for s in v['screenshots']:assert sha(pathlib.Path(s['path']))==s['sha256']
for a,b in zip(trace,trace[1:]):assert a['recorded_at']<=b['recorded_at']
prior_proofs=[];http_fig=[]
for pid in ['16.3','16.4','16.5']:
 assert meta[pid]['figures_sha256']==prior[pid]['figures_sha256']
 prior_proofs.append({'page_id':pid,'reason_no_new_figure_view':'只有16.4品质表前任务说明文字改变，cache/KV共享/packing图及图说用途未变，不需新图解释短文字high/low。','current_figures_sha256':meta[pid]['figures_sha256'],'actual_prior_viewed_screenshots':[s for v in v2['pageviews'] if v['page_id']==pid for s in v['screenshots']],'prior_actual_pageviews':[v for v in v2['pageviews'] if v['page_id']==pid],'basis':'本人recheck-01实际view_image看過，原所有图截图bytes保留；不稱recheck-02又看圖。'})
 for path,h in meta[pid]['figures_sha256'].items():
  url='http://127.0.0.1:8765/figures/'+path.rsplit('/',1)[-1]
  with urllib.request.urlopen(url,timeout=10) as resp:data=resp.read();status=resp.status
  digest=hashlib.sha256(data).hexdigest();assert digest==h
  http_fig.append({'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'page_id':pid,'source_figure':path,'url':url,'status':status,'sha256':digest,'same_as_current_inventory':True,'scope':'只核图bytes身份，不是新视觉阅读。'})
(A/'current-figure-fingerprints.json').write_text(json.dumps(http_fig,ensure_ascii=False,indent=2)+'\n')
issue=next(i for i in old['issues'] if i['id']=='C13_16_03');check=by['16.4']['issue_rechecks'][0]
report={
 'schema_version':1,'stage':'independent_continuity_recheck','phase':'Phase 4','assignment':'13_16','recheck_id':'recheck-02',
 'reviewer_task':'/root/continuity_13_16','fork_turns':'none',
 'identity_statement':'同一位实际原第三轮13_16学生读者；本次局部回讀非新盲讀。入门Python、高中基本数学，不以专家知识补理解。',
 'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','current_verdict':'pass','must_fix':[],'optional':[],
 'verdict_reason':'當前16.3→16.4→16.5已依序真正回讀。16.4品質表前就近說保存屬性QA基準、短文字中的顏色/形狀/high-low音高標記及非英文計時提示，接16.5形状问答仍清楚，没有新的模态猜测或必改项。其余57页 exact文字/图版本与本人已PASS的recheck-01相同；其中16.3/16.5此次作边界真读，其余55頁仅继承。',
 'source_inventory_path':str(inv.relative_to(R)),'source_inventory_sha256':sha(inv),
 'changed_primary_page_ids':['16.4'],'actual_reread_page_ids':['16.3','16.4','16.5'],'actual_read_order':[dict(sequence=i+1,**clean(t)) for i,t in enumerate(trace)],
 'primary_page_ids':v2['primary_page_ids'],'primary_page_count':58,'primary_page_results':results,
 'same_exact_version_primary_count':57,'actual_reread_primary_count':3,'same_exact_version_with_context_reread_primary_count':2,'same_exact_version_without_new_read_primary_count':55,
 'necessary_context_current_versions':{pid:{'source_sha256':meta[pid]['source_sha256'],'figures_sha256':meta[pid]['figures_sha256'],'source_snapshot':meta[pid]['snapshot'],'actual_reread_record':clean(by[pid])} for pid in ['16.3','16.5']},
 'original_records':[{'path':p,'sha256':sha(R/p)} for p in ['docs/course-revision-20261005/continuity/reports/13_16.json','docs/course-revision-20261005/continuity/traces/13_16.jsonl','docs/course-revision-20261005/continuity/reports/13_16-recheck-01.json','docs/course-revision-20261005/continuity/traces/13_16-recheck-01.jsonl']],
 'prior_bytes_preservation':{'manifest_path':str((A/'prior-preservation.json').relative_to(R)),'manifest_sha256':sha(A/'prior-preservation.json'),'file_count':len(pres['files']),'all_prior_bytes_unchanged':True,'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},
 'issue_recheck':{'issue_id':'C13_16_03','initial_issue_preserved':issue,'previous_recheck_preserved':next(i for i in v2['issue_rechecks'] if i['issue_id']=='C13_16_03'),'current_recheck':dict(check,recorded_at=by['16.4']['recorded_at'],source_sha256=meta['16.4']['source_sha256']),'statement':'initial缺任务题材的首次猜测与16.5后文澄清、recheck-01该版理解/判决均原样留存。本次知道high/low是短文字标记，不把这个新句倒填成以前就见过。'},
 'handoffs':[{'from':'16.3','to':'16.4','decision':'pass','reason':'cache一致性先比較同一模型两条路径，随后四Query/少KV组是结构新问题；品质表另外用保存文字属性基準，沒有將英文速度提示当QA。'},{'from':'16.4','to':'16.5','decision':'pass','reason':'品质任务已经当场定义，packing仍明示改成纯助手续写；回测shape?理想circle/错误low作为原QA掉品质的具体例，文字属性与真实音频入口未混。'}],
 'figure_visual_inheritance':prior_proofs,'current_figure_http_fingerprints':http_fig,
 'trace_path':'docs/course-revision-20261005/continuity/traces/13_16-recheck-02.jsonl','trace_sha256':sha(B/'traces/13_16-recheck-02.jsonl'),'artifacts_directory':str(A.relative_to(R)),
 'pageviews_path':str((A/'pageviews.jsonl').relative_to(R)),'pageviews_sha256':sha(A/'pageviews.jsonl'),'pageviews':views,'actual_pageview_count':len(views),'actual_current_text_screenshots_viewed_count':sum(len(v['screenshots']) for v in views),
 'browser_method':{'browser':'/usr/bin/chromium via Playwright','args':['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'],'goto_wait_until':'domcontentloaded','goto_timeout_ms':15000,'screenshot_full_page':False,'screenshot_timeout_ms':15000,'viewports':[{'width':1280,'height':900},{'width':390,'height':844}],'view_image_for_current_text_screenshots':True},
 'unverified_scope':['本次只实际回讀16.3、16.4、16.5三個完整页；沒有又全文讀58頁。其余57 exact文字/图版本未变，其中两頁作context真读，55页完全沿用本人已PASS紀錄。','本次沒有新看cache/KV共享/packing圖，明确沿用本人recheck-01图实际view_image與同SHA；兩張新截图是16.4任务说明文字，不伪称新图实看。','没有补读7.13、T.4、其他前章或外来源；新版当地短文字属性材料已够本轮衔接判断。未讀作者repair notes、他人/技术报告、实作或训练数字旁证。','沒有CPU/GPU新执行、训练、量测、能力或技术事实查证；网站已有结果只当正文阅读。','未查所有屏幕/浏览器/互动或每页所有像素；指定两viewport看了任务说明与周边文字，没有以HTTP图hash替新视读。']
}
assert all(p['decision']=='pass' for p in results)
out=B/'reports/13_16-recheck-02.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report_path':str(out),'report_sha256':sha(out),'trace_path':report['trace_path'],'trace_sha256':report['trace_sha256'],'pageviews_path':report['pageviews_path'],'pageviews_sha256':report['pageviews_sha256'],'prior_preservation_manifest':report['prior_bytes_preservation']['manifest_path'],'prior_preservation_sha256':report['prior_bytes_preservation']['manifest_sha256'],'current_verdict':'pass','actual_reread':['16.3','16.4','16.5'],'new_text_screenshots':2,'actual_pageviews':len(views),'prior_preserved_files':len(pres['files'])},ensure_ascii=False))
