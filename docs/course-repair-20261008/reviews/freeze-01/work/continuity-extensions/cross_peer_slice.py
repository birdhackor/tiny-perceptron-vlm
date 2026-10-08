import json,sys,hashlib
from pathlib import Path
B=Path('docs/course-repair-20261008/reviews/freeze-01')
R=json.loads((B/'reports/reader-extensions.json').read_text()); RM=json.loads((B/'reports/reader-extensions-main.json').read_text());T=json.loads((B/'reports/technical-extensions-initial.json').read_text())
for pid in sys.argv[1:]:
 r=next(p for p in R['pages'] if p['page_id']==pid);rm=next(p for p in RM['pages'] if p['page_id']==pid);t=next(p for p in T['pages'] if p['page_id']==pid)
 same=r['main_four_question_review']==rm['four_question_review']
 print('\nPAGE',pid,'reader original main-four exact equality',same)
 def qs(q):
  return {k:q[k] for k in ['source_promise_quotes','depth','q1','q2','q3','q4','shortest_normal_inferences','unknowns','issues','reading_limitations'] if k in q}
 def compact(q):
  if isinstance(q,dict):return {k:compact(v) for k,v in q.items() if k!='original_question'}
  if isinstance(q,list):return [compact(x) for x in q]
  return q
 print('READER MAIN',json.dumps(compact(qs(r['main_four_question_review'])),ensure_ascii=False,indent=2))
 print('READER COMPLETE',json.dumps(compact(qs(r['complete_page_four_question_review'])),ensure_ascii=False,indent=2))
 print('READER EXTRA',json.dumps({k:r['extras'][k] for k in ['actual_read','q2_mechanism','q2_need','q4_scope','unknowns','new_issues','later_clarifications','issues_reconciled'] if k in r['extras']},ensure_ascii=False,indent=2))
 print('TECH',json.dumps({k:t[k] for k in ['Q2_機制','Q2_用途與選用理由','已教關係','仍缺且影響當節承諾的關係','necessary_findings','optional_findings','unknowns_limits','判定']},ensure_ascii=False,indent=2))
 print('TECH CHECKS',json.dumps([{k:c[k] for k in ['check_id','核查結果','執行程度','未驗證範圍']} for c in t['technical_checks']],ensure_ascii=False,indent=2))
