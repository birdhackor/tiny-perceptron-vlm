import json,hashlib,sys
from pathlib import Path
root=Path('.').resolve();base=Path('docs/technical-reviews/artifacts/p7_technical_a');cb=base/'freeze-06-callback'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=base/'group-a-technical-initial-20261007.json';assert sha(old)=='f33ea856fed5aaecf7f3197778ed671c3c0d7fdc2f34c35cee2ef039babb477d';r=json.loads(old.read_text())
manifest=Path('docs/course-revision-20261007-phase7/reviews/freeze-06/manifest.json');m=json.loads(manifest.read_text());assert sha(manifest)=='0bbf4113fbd8288786bca540330fa9e2109e4fe848304a43b0321efa7bd23709'
pagepath=cb/'page-1.9.json';page=json.loads(pagepath.read_text());oldpage=next(p for p in r['pages'] if p['page_id']=='1.9');proof=json.loads((cb/'exact-byte-comparison.json').read_text());assert sum(p['equal_bytes'] for p in proof['pages'])==59
frozen={p['page_id']:p for p in m['pages']}
page['original_saved_assessment']={'path':str(pagepath),'sha256':sha(pagepath)}
page['retained_previous_page_record']={'report':{'path':str(old),'sha256':sha(old)},'page':oldpage,'scope':'原freeze03本人判断/证据保持原样，仅为历史保留，不冒充新版1.9判断'}
r['pages']=[page if p['page_id']=='1.9' else p for p in r['pages']]
for p in r['pages']:assert p['source_sha256']==frozen[p['page_id']]['source_sha256'] and p['figures_sha256']==frozen[p['page_id']]['figures_sha256']
trace=Path('docs/course-revision-20261007-phase7/reviews/freeze-06/traces/technical/a/077d89a8d55544b08d4bbc341a6f44f3.jsonl');rows=[json.loads(x) for x in trace.read_text().splitlines()];assert rows[-1]['event']=='complete' and len(rows)==5
r['retained_original_report']={'path':str(old),'sha256':sha(old),'immutable':True}
r['trace_files'].append({'path':str(trace),'sha256':sha(trace)})
r['manifest_sha256']=sha(manifest);r['verdict']='pass';r['judgment']='本人原技术owner回查freeze06变动页1.9全部三单元，新原句/函数/相称CPU与必要位置均核，无未解决必要技术问题。其余59页实际冻结正文bytes与图SHA全部exact same才沿用本人原证据。判定仅本组当前freeze，不当全书或全stage通过。'
r['initial_scope']=r['scope'];r['scope']={'primary_pages':60,'newly_reread_pages':['1.9'],'unchanged_exact_bytes_retained_pages':59,'range':'入口/暖身/第1–5章，group a 精确primary顺序','mandatory_boundary_context_pages':[],'initial_session':'86ddcaf7b50f4afeafb3f5b041a7162c','callback_session':'077d89a8d55544b08d4bbc341a6f44f3','initial_actual_checkpoints':230,'new_actual_checkpoints':3,'total_preserved_checkpoints':233,'callback_started_at':rows[0]['recorded_at'],'callback_completed_at':rows[-1]['recorded_at'],'reader_four_questions':'technical question_refs=[]；同owner回查非新盲读，五栏完整保存'}
r['exact_byte_retention_proof']={'path':str(cb/'exact-byte-comparison.json'),'sha256':sha(cb/'exact-byte-comparison.json'),'scope':'本人读实际snapshot bytes比较、逐份hash核manifest；其它59文件没有decode或重新读正文'}
r['procedural_notes'].extend(['freeze06 WAIT期间未start或读新版正文；parent真实followup登记/预览GO后按page1.9与recheck-of原trace启动，原start返回session077d89a8d55544b08d4bbc341a6f44f3，三单元每读后本人五栏/next。没有虚构issue、rechecks均空。','初始reviewer_context fresh指最初独立起点；这次同owner回查不是新fresh盲读；原230checkpoint、所有原判断/unknown/历史stdin与未验范围限制照旧保留。新版1.9仅加3真实checkpoint，不重造其它59正文阅读。','新版1.9本人重读实际原SciPy定位277–315/374–404/583–620和原PyTorchComputingGradients71–87。原官方完整文本放localcache，新正式source只必要短摘录与原URL/version/SHA/locator；初始不可覆写正式证据保持不动。','本回查真正CPU脚本/执行的冻结唯一block原码/command/Python版本/device/输出/SHA均保存。没有GPU或长训练；h.001 w1/5/3小核与极小h误差边界，不用root6短source-match结果或别人的理解作本人证据。','新版1.9只有数表无SVG；本人真view1280/390表与补充位置四图才写receipt。手机table截图下方原代码末尾截断未宣称视觉看到，但完整正文code已逐段读并执行。未重新观看其它59版面，原34页required-false未验范围不升级。','新版报告完整60页仅代表1.9新回查+其它59 exact byte保留；原1.9完整技术record也嵌入retained_previous_page_record供追溯，不作新版pass补答案。旧report/traces/pages/artifacts均不改。','本次group-preflight仍只是metadata，不证明科学truth或全stagegate；actual proof另存新路径，之后实际完整FINAL才收件。'])
for q in rows[1:-1]:r['reading_checkpoints'].append({k:q[k] for k in ['event_index','recorded_at','page_id','unit_index','unit_sha256','source_sha256','understanding','materials_and_labels','expected_change','confusion_and_quote','missing_visuals','issues','rechecks']})
# Schema audit before immutable unique group report creation; no automatic truth or judgments.
sys.path.insert(0,str(root/'scripts'));import check_technical_reviews as t
for p in r['pages']:
 errors=[];aa=t._artifacts(root,p['artifacts'],errors);ss=t._sources(root,p['sources'],aa,errors);t._claims(p['claims'],ss,aa,errors)
 if errors:raise RuntimeError((p['page_id'],errors))
p=base/'group-a-technical-freeze06-callback-20261007.json'
with p.open('x',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(p),'sha256':sha(p),'pages':len(r['pages']),'preserved_checkpoints':len(r['reading_checkpoints']),'new_trace_sha256':sha(trace)},ensure_ascii=False))
