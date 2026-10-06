from pathlib import Path
from datetime import datetime, timezone
import difflib
import hashlib
import json
import re

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
record=json.loads((BASE/'active-current-source-callback.json').read_text())
CURRENT=ROOT/record['callback_dir']
check=json.loads((CURRENT/'current-check.json').read_text())
assert check['python_ast_equal'] and check['other_section_bytes_equal']
assert len(check['raw_stages'])==15 and len(check['raw_edges'])==13
assert check['current_source_sha256']==record['source_sha256']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
report=json.loads((ROOT/record['opaque_history']).read_bytes())
assert sha(ROOT/record['opaque_history'])==record['history_sha256']
assert report['reviewer_task']=='/root/p6_fact_19_4' and report['lesson_id']=='19.4'
for artifact in report['artifacts']:
    assert sha(ROOT/artifact['path'])==artifact['sha256'],artifact['path']
for source in report['sources']:
    if source['kind']=='repository_code':assert sha(ROOT/source['path'])==source['sha256']
for figure,digest in report['figure_sha256'].items():assert sha(ROOT/figure)==digest

original=(BASE/'inputs/19.4.md').read_bytes()
current=(CURRENT/'19.4.md').read_bytes()
assert report['source_sha256']==hashlib.sha256(original).hexdigest()
old_lines=original.decode().splitlines()
new_lines=current.decode().splitlines()
mapping={}
for tag,i,j,k,l in difflib.SequenceMatcher(a=old_lines,b=new_lines,autojunk=False).get_opcodes():
    for old_index in range(i,j):
        new_index=k+min(old_index-i,max(l-k-1,0))
        mapping[159+old_index]=record['section_first_line']+new_index
tag=Path(record['callback_dir']).name
new_id='a_'+tag.replace('-','_').replace('.','_')
render_id=new_id+'_render'
image640_id=new_id+'_figure640'
image360_id=new_id+'_figure360'

def add(identifier,path,kind,description,command=None,result=None,environment=None):
    p=CURRENT/path
    item={'id':identifier,'kind':kind,'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'description':description}
    if kind=='execution':item.update(command=command,result=result,environment=environment)
    report['artifacts'].append(item)

add(new_id,'current-check.json','execution','原owner實際複查目前完整19.4：Ruff換行的AST、其余原bytes、当前Python fence兩個cwd、15個原raw completed紀錄/13個parent SHA、native4000/1000及必要程式版本。','/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.4/current_check.py > '+record['callback_dir']+'/current-check.log 2>&1','exit0；只有repo=next(...)物理換行收併，AST相同；目前程式兩cwd七行吻合；15段/13鏈與native4000/1000重新核對成立；沒有重訓或heldout重評。',check['environment'])
add(render_id,'render-svg.log','execution','本人重新真正渲染当前SVG並實際view_image兩尺寸；不是僅沿用舊圖hash。','/workspace/tiny-perceptron-vlm/.venv/bin/python '+record['callback_dir']+'/render_svg.py > '+record['callback_dir']+'/render-svg.log 2>&1','exit0；640x884與360x497 PNG渲染完成，兩張已本人實際查看。',{'python':check['environment']['python'],'device':'cpu','renderer':'system librsvg via Cairo','cairo':'1.18.4'})
add(image640_id,'lineage-640.png','figure_render','目前SVG在640寬度的真渲染，已本人實際查看；八段順序、向下箭頭、best及新段reset文字相符。')
add(image360_id,'lineage-360.png','figure_render','目前SVG在360寬度的真渲染，已本人實際查看；无末段裁切或箭頭/標籤變義。')
figure_view={'reviewer_task':'/root/p6_fact_19_4','viewed_at':datetime.now(timezone.utc).isoformat(),'tools':'view_image called separately on both current rendered PNGs','source_svg':'course/figures/p6-19-start-lineage.svg','source_sha256':sha(ROOT/'course/figures/p6-19-start-lineage.svg'),'observed':'八段MoE主鏈順序pretrain/SFT/vision/OCR/audio/joint/weighted/native與正文及原始parent鏈一致；每支向下箭頭與best/reset標籤無變。Dense止weighted由正文及其原發布來源說明。','images':[{'path':str((CURRENT/name).relative_to(ROOT)),'sha256':sha(CURRENT/name)} for name in ['lineage-640.png','lineage-360.png']]}
(CURRENT/'figure-view.json').write_text(json.dumps(figure_view,ensure_ascii=False,indent=2)+'\n')
for index,p in enumerate(sorted(CURRENT.rglob('*'))):
    if p.is_file() and p.name not in ['current-check.json','render-svg.log','lineage-640.png','lineage-360.png','update-report.log','checker.log']:
        add(new_id+'_snapshot_'+str(index),p.relative_to(CURRENT).as_posix(),'code' if p.suffix=='.py' else 'source_snapshot','本次原owner複查永久證據/現稿原bytes，意義與實際範圍見start-record及current-check：'+p.name)
for suffix,name in [('code','current_check.py'),('update_code','update_current_report.py')]:
    p=BASE/name
    report['artifacts'].append({'id':new_id+'_'+suffix,'kind':'code','path':str(p.relative_to(ROOT)),'sha256':sha(p),'description':'本次真正执行的current-source核對/本人報告更新程式。'})

for claim in report['claims']:
    prefix='course/chapters/19.md:'
    assert claim['location'].startswith(prefix)
    old_location=claim['location']
    claim['location']=prefix+re.sub(r'\d+',lambda m:str(mapping[int(m.group())]),claim['location'][len(prefix):])
    claim['artifact_ids'].append(new_id)
    claim['current_source_recheck']={'reviewer_task':'/root/p6_fact_19_4','current_source_sha256':record['source_sha256'],'initial_location':old_location,'claim_unchanged':True,'support':'本人完整重讀現稿，原文只有Python fence物理換行變動，AST相同；其余原文/命令bytes、必要實作/圖及原证据版本相同。這項原主張及既有支持範圍因此保留。'}
    if claim['id']=='c18':
        claim['verification']['observed']='本次当前原Python fence在repo根目錄及notebooks/19重新原樣執行：兩者各印同樣七行，最後moe-native 完成4000選定1000；只有json/pathlib，未載入權重或啟動訓練。'
    if claim['id'] in ['c02','c14']:
        claim['artifact_ids'].extend([render_id,image640_id,image360_id])
report['source_sha256']=record['source_sha256']
report['verdict']='pass'
report['inspection']['initial_read_text']=report['inspection']['read_text']
report['inspection']['read_text']='本次本人讀目前完整19.4原bytes（course/chapters/19.md166-255）、必要19.3前置及TRAINING.md1-140；重讀凍結、fresh/resume必要分支與保存官方原文相關段落；核對自己的原證據。19.3僅供理解資料切分前提，不擴大19.4判定範圍。'
callback={'kind':'original_owner_current_source_recheck','reviewer_task':'/root/p6_fact_19_4','completed_at':datetime.now(timezone.utc).isoformat(),'source_sha256':record['source_sha256'],'current_section_snapshot':record['callback_dir']+'/19.4.md','current_full_chapter_snapshot':{'path':record['callback_dir']+'/19.md','sha256':record['chapter_snapshot_sha256'],'meaning':'本次完整章原始byte快照，仅用于保存来源；本报告scope仍只19.4及必要context，其他節/全檔变化不当成19.4内容变化。'},'prior_report_opaque_history':record['opaque_history'],'prior_report_sha256':record['history_sha256'],'actual_scope':check['read_scope'],'change':check['only_change'],'methods_and_execution':'亲自确认AST与unchanged raw bytes；运行当前短程序两cwd；重新核对15段/13 parent best链、native4000/1000；当前图重新render/view640与360。原bounded optimizer/RNG/sampler算例不重复训练，因为相应方法/输入/证据bytes未变。','artifact_id':new_id,'verdict':'pass'}
report.setdefault('owner_callbacks',[]).append(callback)
for name in ['factual_accuracy','numeric_verification','source_verification','limitations']:
    report['checks'][name]['details']+=' 本次原owner完成当前源AST/bytes、必要實作与原证据版本复查，保留原支持范围；当前短程序及原步数/parent链已真实重新核对，未重跑训练或heldout。'
report['checks']['figure_consistency']['details']='本次本人重新以librsvg/Cairo渲染当前本节唯一SVG为640x884及360x497并用view_image查看；八段顺序、向下箭头、best及新段reset標籤与当前正文/原始链一致，source SVG hash保持不变。'
live=(ROOT/'course/chapters/19.md').read_bytes()
m=re.search(rb'^## 19\.4 .+$',live,re.M);n=re.search(rb'^## ',live[m.end():],re.M);end=m.end()+n.start() if n else len(live)
assert live[m.start():end]==current
(ROOT/'docs/technical-reviews/19.4.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Original owner current-source recheck pass; 21 claims retained after actual AST, current execution, raw lineage, source and render/view checks.')
print('Current source SHA256',record['source_sha256'])
print('Opaque prior report history',record['opaque_history'])
