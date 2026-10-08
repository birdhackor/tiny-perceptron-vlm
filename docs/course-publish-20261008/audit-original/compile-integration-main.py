import json
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
b=Path(__file__).resolve().parent
m=json.loads((b/'manifest.json').read_text())
records=[json.loads(x) for x in (b/'traces/reader-integration.jsonl').read_text().splitlines()]
ids=m['groups']['integration']['pages']; inventory={p['page_id']:p for p in m['inventory']['pages']}
checkpoints=[r for r in records if r['kind']=='checkpoint' and r['phase']=='main']
assert len(checkpoints)==148 and len([r for r in records if r['kind']=='page_end' and r['phase']=='main'])==30
pages=[];issues=[]
for pid in ids:
 rs=[r for r in checkpoints if r['page_id']==pid]; last=rs[-1]['note']
 assert 'four_questions' in last and 'page_result' in last
 pages.append({'page_id':pid,'source':inventory[pid]['source'],'selector':inventory[pid]['selector'],'source_sha256':inventory[pid]['source_sha256'],'figures_sha256':inventory[pid]['figures_sha256'],'result':last['page_result'],'first_read_checkpoint_count':len(rs),'page_end_four_questions':last['four_questions'],'first_read_checkpoints':rs})
 for r in rs:
  for issue in r['note'].get('issues',[]):
   issues.append({**issue,'page_id':pid,'unit_index':r['unit_index'],'source_sha256':r['source_sha256'],'discovered_by':'/root/read_integration','discovery_phase':'main-only first reading','prompted_by_prior_issue':False})
main_figures=['p6-19-start-core','p6-19-start-splits','p6-19-start-lineage','p6-19-modal-fashion','p6-19-modal-ocr','p6-19-modal-voice','p6-19-modal-tool','p6-19-delivery-package-roles','rewrite-20-input-routes','natural-v4-training-cat','natural-v4-family-crops','natural_base_adapter','natural-v4-answer-mask','natural_photo_evidence','natural_reading_order','natural-v4-asr-two-routes']
context_figures=['rewrite-15-combine','rewrite-02-embedding-purpose','new-11.18-two-regions','rewrite-12-frame-features','rewrite-07-18-feedback-materials','p7-7.20-future-targets','rewrite-16-cache-append','rewrite-16-query-kv-sharing','rewrite-16-causal-mask','new-11.17-single-object','new-11.17-left-right','rewrite-08-lora-paths','rewrite-01-update']
report={'schema_version':1,'reviewer':'/root/read_integration','kind':'independent sequential main-only reader audit','created_at':datetime.now(timezone.utc).isoformat(),'baseline_commit':m['baseline_commit'],'criteria_sha256':{k:v for k,v in m['criteria_sha256'].items() if k in ['SKILL.md','references/review-protocol.md','references/calibration.md']},'reader_background':'高中／基本數學大學生；只使用實際已讀的必要前文，不假定讀完全書。','summary':{'necessary_issues':1,'blockers':0,'burdens':1,'optional_name_observations':3,'rewrite_scale':'19.12局部補控制結果摘要；目前首讀證據不支持整章重排或整章／全書重寫。','scope_warning':'這是AI輔助的閱讀理解與證據呈現診斷，未作正式能力驗收、實作核對、訓練复現或真實網頁視覺驗收。'},'pages':pages,'issues':issues,'coverage':{'assigned_pages':30,'completed_main_pages':30,'main_checkpoints':148,'page_results':dict(Counter(p['result']['status'] for p in pages)),'actual_prerequisite_route':[r for r in records if r['kind']=='prerequisite_read'],'read_criteria':['freeze/criteria/SKILL.md','freeze/criteria/references/review-protocol.md','freeze/criteria/references/calibration.md'],'not_read':['project-context','作者歷史','舊審閱','同行報告','實作與外部答案','選讀折疊區（本檔封存前）'],'report_aggregation':'僅機械彙整本人先前逐段手寫checkpoint與page_result，不由程式生成理解答案。','unknowns_preserved':'各checkpoint保留當時未知；後文澄清只追加，沒有回寫早期答案。'},'visual_scope':{'viewed_group_figures':main_figures,'viewed_context_figures':context_figures,'viewed_render_widths':[640,360],'method':'逐張view_image親看相應renders PNG；必要對象、位置、箭頭及圖文對照寫在checkpoint。','group_figures_count':16,'context_figures_count':13,'real_webpage_desktop':'unverified','real_webpage_mobile':'unverified','ui_function_and_layout':'unverified','actual_page_math_fold_and_navigation':'unverified','standalone_svg_does_not_imply_page_acceptance':True}}
out=b/'reports/reader-integration-main.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(out),'pages':len(pages),'checkpoints':len(checkpoints),'issues':len(issues)},ensure_ascii=False))
