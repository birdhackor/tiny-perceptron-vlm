from pathlib import Path
import json, re, hashlib, datetime
from collections import Counter
b=Path('/workspace/work/tutorial-audit-20261008');mainpath=b/'reports/reader-extensions-main.json';mainbytes=mainpath.read_bytes();mainhash=hashlib.sha256(mainbytes).hexdigest();assert mainhash=='4a8dd71309a58266d11d38bf2e41c899e933545d609258cffd3004069ec44809'
r=json.loads(mainbytes);m=json.loads((b/'manifest.json').read_text()); inv={p['page_id']:p for p in m['inventory']['pages']};ids=m['groups']['extensions']['pages']; supplements={pid:[] for pid in ids};new=[];notes=[]
for path in sorted((b/'notes/extensions').glob('extras-*.json')):
 d=json.loads(path.read_text());notes.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'record':d})
 if 'pages' in d:
  for q in d['pages']:supplements[q['page_id']].append(q)
 elif d.get('page_id')=='training':
  for q in d['sections']:
   q=dict(q);q['page_id']='training'; supplements['training'].append(q)
for page in r['pages']:
 pid=page['page_id'];src=Path(inv[pid]['snapshot']);ds=re.findall(r'<details\b[^>]*>.*?</details>',src.read_text(),re.S)
 sp=supplements[pid];assert len(sp)==len(ds),(pid,len(sp),len(ds))
 if pid=='training':sp.sort(key=lambda t:t['detail_index'])
 for i,q in enumerate(sp):
  q['detail_index']=i;q['detail_sha256']=hashlib.sha256(ds[i].encode()).hexdigest();q['source']=inv[pid]['source'];q['source_sha256']=inv[pid]['source_sha256']
  for issue in q.get('issues',[]):
   issue=dict(issue);issue.update({'page_id':pid,'detail_index':i,'detail_sha256':q['detail_sha256'],'source':q['source'],'source_sha256':q['source_sha256'],'discovery_phase':'supplemental after main seal'});new.append(issue)
 page['supplemental_read']={'details_total':len(ds),'details_read':len(sp),'read_scope':'所有本页折叠正文；不打开所连外部材料' if ds else '本页无折叠区，正文已全读；外部链未打开','records':sp,'initial_main_status_preserved':page['status']}
 if pid in ('training','curriculum'):page['final_required_rewrite_scale']='修補'
 if pid=='training':
  page['supplemental_impact']='固定sft/style与正文练习产物不同、固定入口不自动补依赖确认T10前置修补；新增3条补名optional，main原判不改。'
 elif pid=='readme':page['supplemental_impact']='主读之后asset-storage明确model.safetensors，只追加safe格式的后续理解；readme本地补名optional初判保留。'
 else:page['supplemental_impact']='选读提供历史分母/条件/引用范围，未改main判断。' if sp else '无折叠区；正文初判保持。'
 page['issues']+= [q for q in new if q['page_id']==pid]
r['phase']='final independent first-read report with separately recorded supplemental impact';r['created_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();r['main_seal']={'path':str(mainpath),'sha256':mainhash,'original_bytes_preserved':True};r['issues']+=new;r['supplemental_notes']=notes
r['summary']={'pages':36,'main_units':199,'optional_details_read':sum(len(x) for x in supplements.values()),'pages_with_optional_details':sum(bool(x) for x in supplements.values()),'issues_total':len(r['issues']),'issues_by_severity':dict(Counter(q['severity'] for q in r['issues'])),'issues_by_rewrite_scale':dict(Counter(q['rewrite_scale'] for q in r['issues'])),'main_issues_preserved':16,'supplemental_new_optional':len(new),'required_issue_ids':['EXT-T10-PREQ','EXT-CURR-BPB'],'major_rewrite_needed':False,'chapter_reorder_needed':False,'rationale':'两个必要问题仅需可见前置接续或指标名称/单位与最短理由；例子、窄任务机制与证据界线整体可追踪，未见章节大改证据。'}
r['coverage']['optional_details_all_read']=True;r['coverage']['optional_details_count']=31;r['coverage']['supplemental_method']='reader.py extras 已封存main；依parent明确授权直接读取本组冻结来源中details，不再next重读已完成main。';r['coverage']['linked_supplementary_reports_or_implementation_read']=False;r['coverage']['prerequisite_routes_unchanged_after_main']=True;r['visual_scope']['optional_sections_have_additional_svg']=False;r['visual_scope']['supplemental_external_report_figures_verified']=False
out=b/'reports/reader-extensions.json';assert not out.exists();out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');assert mainpath.read_bytes()==mainbytes
print(json.dumps({'report':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'summary':r['summary'],'main_hash_unchanged':mainhash},ensure_ascii=False))
