from pathlib import Path
import json, hashlib, datetime
from collections import Counter
b=Path('/workspace/work/tutorial-audit-20261008')
m=json.loads((b/'manifest.json').read_text()); ids=m['groups']['extensions']['pages']; inv={p['page_id']:p for p in m['inventory']['pages']}
trace=[json.loads(x) for x in (b/'traces/reader-extensions.jsonl').read_text().splitlines()]
check={(t['page_id'],t['unit_index']):t for t in trace if t['kind']=='checkpoint' and t['phase']=='main'}
ends={t['page_id']:t for t in trace if t['kind']=='page_end' and t['phase']=='main'}
assert len(ids)==len(ends)==36
pages=[]; issues=[]; exposures=[]
for pid in ids:
 ns=[]
 for f in (b/'notes/extensions').glob(pid+'-u*.json'):
  d=json.loads(f.read_text());ns.append((d['unit_index'],f,d))
 ns.sort(key=lambda t:t[0]); assert [t[0] for t in ns]==list(range(len(ns)))
 assert ns and all((pid,i) in check for i,_,_ in ns)
 rec=ns[-1][2]['page_record'].copy(); rec['status']=rec.get('status',rec.get('result'))
 rec.update({'source':inv[pid]['source'],'selector':inv[pid]['selector'],'source_sha256':inv[pid]['source_sha256'],'title':inv[pid]['title'],'main_units_completed':len(ns),'main_completed_at':ends[pid]['at'],'checkpoints':[]})
 pissues=[]
 for i,f,d in ns:
  rec['checkpoints'].append({'unit_index':i,'path':str(f),'note_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'at':check[(pid,i)]['at'],'unit_sha256':check[(pid,i)]['unit_sha256'],'record':d})
  for q in d.get('issues',[])+d.get('page_record',{}).get('issues',[]):
   if q['id'] not in {x['id'] for x in pissues}:
    q=dict(q);q.update({'page_id':pid,'unit_index':i,'source':inv[pid]['source'],'source_sha256':inv[pid]['source_sha256'],'checkpoint_path':str(f),'discovered_at':check[(pid,i)]['at']});pissues.append(q)
  if d.get('exposure'):
   e=dict(d['exposure']);e.update({'page_id':pid,'unit_index':i,'at':check[(pid,i)]['at'],'checkpoint_path':str(f)});exposures.append(e)
 rec['issues']=pissues;issues.extend(pissues);pages.append(rec)
figs={}
for pid in ids+['7.19','1.12','14.2']:
 for fig,h in inv[pid]['figures_sha256'].items():
  stem=Path(fig).stem
  paths=[str(b/'renders'/f'{stem}-{w}.png') for w in (640,360)]
  assert all(Path(p).exists() for p in paths)
  figs[fig]={'source_sha256':h,'viewed_with':'view_image','widths_px':[640,360],'render_paths':paths,'readability':'亲看两宽，图中材料/箭头/标签可辨；逐图判断记录于对应checkpoint','page_layout_verified':False}
routes=[]
for t in trace:
 if t['kind']=='prerequisite_read':
  d=dict(t);d['actual_read_scope']='main-only'
  if d['page_id']=='first-steps':d['actual_read_scope']='仅W.1与W.7 main正文；context输出先存文件，再只显示这两节，不把整页视为读过'
  routes.append(d)
report={'schema_version':1,'reviewer':'/root/read_extensions','group':'extensions','phase':'main-only sealed first read','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline_commit':m['baseline_commit'],'criteria_read':['SKILL.md','references/review-protocol.md','references/calibration.md'],'criteria_sha256':{k:m['criteria_sha256'][k] for k in ('SKILL.md','references/review-protocol.md','references/calibration.md')},'criteria_not_read':['references/project-context.md','historical issue records','author conclusions outside unavoidable current-source text','other reviewers findings'],'reader_background':'基本数学高中/大学生；概念先备只计实际读过来源','pages':pages,'issues':issues,'summary':{'pages':len(pages),'main_units':len(check),'issues_by_severity':dict(Counter(q['severity'] for q in issues)),'issues_by_rewrite_scale':dict(Counter(q['rewrite_scale'] for q in issues)),'major_rewrite_needed':False,'chapter_reorder_needed':False,'rationale':'必要问题为局部术语/理由与固定训练前置接续，两处均可精准修补；既有例子机制、结果角色及操作路线整体可追踪。'},'coverage':{'canonical_page_ids':ids,'all_main_completed':True,'main_checkpoint_count':len(check),'pages_completed_count':len(ends),'trace_path':str(b/'traces/reader-extensions.jsonl'),'prerequisite_routes':routes,'optional_sections_read_before_seal':False,'historical_links_opened':False,'implementation_read':False,'external_answers_read':False,'actual_commands_executed_from_course':False,'exposures':exposures,'independence_limit':'curriculum unit3起不可宣称完全无历史提示：看到现行教材自带过去審閱的概括摘要，未打开旧審閱；前32页与T10发现早于曝光，原判保持。'},'visual_scope':{'figures':figs,'figure_count':len(figs),'viewed_renders_count':2*len(figs),'actual_webpages_verified':False,'notebook_visual_layout_verified':False,'colab_verified':False,'model_training_or_tests_run':False,'linked_empirical_reports_verified':False,'missing_required_visuals_observed':False,'unverified_claim_policy':'各页保留声称但未核实的实际训练/测试/发布/平台运行；未把声称当本轮验收，未核实本身不自动判缺陷。'}}
out=b/'reports/reader-extensions-main.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'summary':report['summary']},ensure_ascii=False))
