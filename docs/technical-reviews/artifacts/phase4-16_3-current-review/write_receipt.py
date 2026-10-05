import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
P=Path(__file__).resolve().parent
OLD=ROOT/'docs/technical-reviews/artifacts/phase4-16_3-independent'
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((P/'input/extraction.json').read_bytes())
history=json.loads((P/'history/prior-manifest.json').read_bytes())
raw_primary=json.loads((OLD/'primary-inspections.json').read_bytes())['sources']
claim_checks={
'causal-cache':'本人重讀整段因果/cache/new work主張；重讀HF Cache Attention matrices/Cache storage、current model/attention/FFN契約；固定權重/前文/位置的因果條件與原check支持範圍不變。',
'original-fence-api':'本人讀現在unchanged fence，核3-ID prefix+1-new-ID slices、兩路output位置axis、vocab264；重讀eval/no_grad/allclose原官方片段；含default rtol1e-5；舊實執行證據hash和所有dependency bytes相同。',
'position-and-prefix':'本人重讀當前offset/rebuild練習；current model offset及position embedding/current attention concat/mask不變；親讀原wrong offset/last-ID/stale-prefix/rebuilt-prefix真數值，僅復用已測範圍。',
'padding-limit':'本人重讀current PAD例及ByteTokenizer/PAD0/current attention valid與cache拒絕分支；原B2不同有效長度variant契約與證據不變，未擴充變長cache。',
'query-kv-heads':'本人再讀GQAv3 Figure2/§2.2；current attention projection和repeat契約、原兩model configs heads4/kv4-or1一致；不套用論文成績。',
'raw-cache-measurements':'本人再讀具名raw measurement pointers及original _fixed_decode/_cache_probe；script重新核25/12/arrays/maxima/EOS/decode，Git code hash匹配；每模型own routes，共同reference history的score probe支持範圍不变。',
'floating-scope':'本人再讀PyTorchv2.14.1 numerical_accuracy原片段及allclose公式；current容差/非bit/有限prompt-dtype-length範圍原句不變。',
'near-tie':'本人重讀current原句、原near-tie CPU actual stdout及allclose契約；小差argmax可不同算例及支持範圍不變，沒有新的GPU勝負結論。'}
receipt={
'id':'16_3-current-owner-reinspection',
'reviewer_task':'/root/phase4_factual_coordinator/factual_16_3',
'reinspection_kind':'same original independent technical owner; actual current-version follow-up, not a new fresh reviewer',
'current_source':'course/chapters/16.md#16.3','current_source_sha256':meta['source_sha256'],
'current_section_snapshot':'input/section.md','current_full_chapter_frozen_input':{'path':'input/chapter16-current-frozen-input.md','sha256':H(P/'input/chapter16-current-frozen-input.md'),'meaning':'Actual full chapter bytes frozen for this follow-up only; not chapter-introduction review.'},
'intro':{'applicable':False,'sha256':None,'reason':'Assigned16.3 is not first section; chapter introduction not substantively inspected.'},
'previous_report':{'snapshot':'history/16.3-prior.json','sha256':history['prior_report_sha256'],'preserved_opaque':True},
'previous_complete_evidence':{'archive':'history/prior-proof.tar.gz','sha256':history['opaque_archive_sha256'],'manifest':'history/prior-manifest.json','original_directory':history['original_evidence_directory'],'original38_files_hash_checked':True},
'actual_source_reads':['Entire current16.3 section; only change is transition now explicitly3prefix+fourth ID','Entire needed current16.2; SHA573f962836128ef5b67281a635460b05125abacd254b29c7a89c0082ddd6f5a0, identical bytes to prior needed previous section','Current model.py1–131 AST first, attention.py1–74 AST first, modern.py35–53, data.py11–28','Original immutable architecture AST-located _fixed_decode514–521 and _cache_probe525–555 re-read; previous other method/read provenance retained unchanged'],
'change_supported':'Current16.2 has4-token prefill then ID5; current16.3 explicitlyshrinks to3-token prefix and ID4. Currentunchangedfence ids[:, :3]/ids[:, 3:] and图saved positions0/1/2+new3 agree. No newalgorithm,axis,unit,denominatoror empirical result introduced.',
'per_claim_current_inspection':[{'claim_id':k,'status':'verified','inspection':v} for k,v in claim_checks.items()],
'read_original_authorities_again':[{'source_id':r['id'],'url':r['url'],'version':r['version'],'accessed_on_original':r['accessed_on'],'authority_reason':r['authority_reason'],'locator':r['locator'],'snapshot_path':str((OLD/r['snapshot']).relative_to(ROOT)),'snapshot_sha256':H(OLD/r['snapshot']),'original_sha256':r['original_sha256'],'current_support_scope':r['supports'],'reinspection':'Personally re-read relevant original snapshot; same immutable version and unchanged bytes. No new network retrieval claimed.'} for r in raw_primary],
'raw_JSON_policy':'Top-level key/types before named pointers; only per-model cache prompt_tokens/generated_tokens/generated_ids_full/generated_ids_cached/generated_text/identical_greedy_ids/per_step_logit_max_error and model/config, plus original provenance used by Git/hash gate. No notes/review/correction/results explanation read.',
'current_execution':[{'code':'verify_reuse.py','command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-16_3-current-review/verify_reuse.py','exit_code':0,'stdout':'execution/verify-reuse.stdout.txt','stderr':'execution/verify-reuse.stderr.txt','purpose':'Byte identity/code provenance plus current existing raw measurement arithmetic; no modelforward.'},{'code':'render_current_page.py','command':'timeout 45s .venv/bin/python docs/technical-reviews/artifacts/phase4-16_3-current-review/render_current_page.py','exit_code':0,'stdout':'execution/render-current-page.stdout.txt','stderr':'execution/render-current-page.stderr.txt','purpose':'Actual local current page/figure render at1280×800 and390×844, details expanded for measurement screenshots.'},{'code':'inspect_mobile_code.py','command':'timeout 30s .venv/bin/python docs/technical-reviews/artifacts/phase4-16_3-current-review/inspect_mobile_code.py','exit_code':0,'stdout':'execution/mobile-code-inspection.stdout.txt','stderr':'execution/mobile-code-inspection.stderr.txt','purpose':'Mobile code/output DOM/text; initial attempt to scrollPREdid not scroll real code container, recorded honestly.'},{'code':'inspect_mobile_scroll.py','command':'timeout 25s .venv/bin/python docs/technical-reviews/artifacts/phase4-16_3-current-review/inspect_mobile_scroll.py','exit_code':0,'stdout':'execution/mobile-scroll.stdout.txt','stderr':'execution/mobile-scroll.stderr.txt','purpose':'Follow-up computed layout confirmsCODElocal overflow:auto client375/scroll541, actualscrollLeft166; screenshotrightedge seen.'}],
'environment':{'python':'3.13.5','torch':'2.14.1+cpu','Chromium':'151.0.7922.173','new_inference_or_training':'none; original own unchangedCPUevidence reused after source/support gates'},
'current_visual':{'page_url':'http://127.0.0.1:8765/16.3.html','page_snapshot':'input/current-page.html','page_sha256':H(P/'input/current-page.html'),'figure_path':'course/figures/rewrite-16-cache-append.svg','figure_sha256':H(P/'input/current-cache-append.svg'),'servedSVGcanonicalbytes_match':True,'actual_viewed_screenshots':['desktop-top','desktop-figure','desktop-code','desktop-measurement','mobile-top','mobile-figure','mobile-code','mobile-measurement','mobile-code-right','mobile-output','mobile-code-actual-right'],'viewport_dimensions':[[1280,800],[390,844]],'inspection':'親看兩寬度：新3prefix銜接句與圖的0/1/2old+3new一致；Queryall4labels/arrows清楚；正文code/output/測量無遮住必要圖標籤。Mobilecode為局部水平scroll，真正CODE容器scroll後右側cache/allclose可見；無整頁水平overflow。','fullpage_images_captured_but_not_used_as_sole_visual_proof':True},
'contamination':'No author revision summary, third reader/other technical judgement read. Old canonical report preserved opaque. Current教材 claims read as object of verification; primary source snapshots and raw measurement/methodreadings are original evidence.',
'unresolved_dependencies':[],'verdict':'pass'}
for row in receipt['current_execution']:
 row['code_sha256']=H(P/row['code']);row['stdout_sha256']=H(P/row['stdout']);row['stderr_sha256']=H(P/row['stderr'])
(P/'reinspection-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'reinspection_id':receipt['id'],'receipt_path':str((P/'reinspection-receipt.json').relative_to(ROOT)),'receipt_sha256':H(P/'reinspection-receipt.json'),'source_sha256':receipt['current_source_sha256'],'verdict':receipt['verdict']},ensure_ascii=False))
