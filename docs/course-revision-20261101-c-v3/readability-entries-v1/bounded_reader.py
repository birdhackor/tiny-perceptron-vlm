"""原判收件後，依位元組差異逐單位有界重讀 source21；保留舊判。"""
import difflib
import hashlib
import importlib.util
import json
import re
import sys
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'recheck-source21-v1'
spec=importlib.util.spec_from_file_location('reader_api',ROOT/'docs/course-revision-20261016/tools/reader.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(path.read_bytes())
def ref(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path.read_bytes())}
def save(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
def folds(body):return re.findall(r'<details\b[^>]*>.*?</details>',body,flags=re.S)

op,*rest=sys.argv[1:]
plan_path=OUT/'plan.json'; state_path=OUT/'state.json'
if op=='start':
 oldfreeze=read(ROOT/'docs/course-revision-20261101-c-v3/source-freeze-v19.json')
 newpath=ROOT/'docs/course-revision-20261101-c-v3/source-freeze-v21.json'
 assert sha(newpath.read_bytes())=='d9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea'
 newfreeze=read(newpath);old={p['page_id']:p for p in oldfreeze['pages']};new={p['page_id']:p for p in newfreeze['pages']}
 specs=[]
 for pid in ['readme','course','training','asset-storage','curriculum','validation','index']:
  specs.append({'page_id':pid,'canonical':True,'source':new[pid]['source'],'old_snapshot':old[pid]['snapshot'],'new_snapshot':new[pid]['snapshot'],'old_sha256':old[pid]['source_sha256'],'source_sha256':new[pid]['source_sha256'],'figures_sha256':new[pid]['figures_sha256']})
 for pid,source in [('readme-en','README_en.md'),('omni-products','docs/omni-products/README.md'),('decision-product','docs/decision-product/README.md')]:
  oldstate=read(OUT.parent/'auxiliary'/pid/'state.json');raw=(ROOT/source).read_bytes();target=OUT/'auxiliary-sources'/f'{pid}.md';target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(raw)
  specs.append({'page_id':pid,'canonical':False,'source':source,'old_snapshot':oldstate['snapshot'],'new_snapshot':target.relative_to(ROOT).as_posix(),'old_sha256':oldstate['source_sha256'],'source_sha256':sha(raw),'figures_sha256':{}})
 units=[]
 for s in specs:
  a=(ROOT/s['old_snapshot']).read_bytes();b=(ROOT/s['new_snapshot']).read_bytes();assert sha(a)==s['old_sha256'] and sha(b)==s['source_sha256']
  before=reader.units(a.decode());after=reader.units(b.decode());ah=[sha(x.encode()) for x in before];bh=[sha(x.encode()) for x in after]
  operations=difflib.SequenceMatcher(a=ah,b=bh,autojunk=False).get_opcodes();selected=set()
  for tag,i1,i2,j1,j2 in operations:
   if tag!='equal':
    selected.update(range(j1,j2))
    if j1==j2:
     if j1>0:selected.add(j1-1)
     if j1<len(after):selected.add(j1)
  s['opcodes']=operations;s['old_unit_count']=len(before);s['new_unit_count']=len(after);s['selected_new_unit_indices']=sorted(selected)
  for j in sorted(selected):units.append({'page_id':s['page_id'],'canonical':s['canonical'],'kind':'main-unit','new_unit_index':j,'unit_sha256':bh[j],'source_sha256':s['source_sha256']})
  fa=folds(a.decode());fb=folds(b.decode());assert len(fa)==len(fb)
  s['folded_blocks_unchanged']=all(x==y for x,y in zip(fa,fb))
  for j,(x,y) in enumerate(zip(fa,fb)):
   if x!=y:units.append({'page_id':s['page_id'],'canonical':s['canonical'],'kind':'folded-block','new_unit_index':j,'unit_sha256':sha(y.encode()),'source_sha256':s['source_sha256']})
 plan={'kind':'bounded_recheck_not_first_read','reviewer_task':'/root/c_v3_reader_entries','created_at':datetime.now(UTC).isoformat(),'source_freeze':ref(newpath),'original_report':ref(OUT.parent/'reader-original-v1.json'),'sources':specs,'reading_units':units,'scope_note':'先前整頁正文/全部折疊+位元組相同部分沿真實既讀證據；只對新快照差異逐單位重讀。附頁不計native370。'}
 save(plan_path,plan);state={'next_index':0,'started_at':datetime.now(UTC).isoformat(),'plan':ref(plan_path)};save(OUT/'opening.json',state);save(state_path,state)
 print(json.dumps({'changed_units_total':len(units),'source_units':{s['page_id']:s['selected_new_unit_indices'] for s in specs},'folds_all_unchanged':all(s['folded_blocks_unchanged'] for s in specs)},ensure_ascii=False))
else:
 plan=read(plan_path);state=read(state_path);i=state['next_index'];u=plan['reading_units'][i];n=json.loads(rest[0]);assert n['unit_index']==i
 n.update(u,reviewer_task=plan['reviewer_task'],saved_at=datetime.now(UTC).isoformat(),reading_kind='bounded_recheck_after_original_collected')
 save(OUT/f'note-{i:03}.json',n);state['next_index']+=1;state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
i=state['next_index'];print(json.dumps({'unit_index':i,'units_total':len(plan['reading_units']),'complete':i==len(plan['reading_units'])},ensure_ascii=False))
if i<len(plan['reading_units']):
 u=plan['reading_units'][i];s=next(s for s in plan['sources'] if s['page_id']==u['page_id']);raw=(ROOT/s['new_snapshot']).read_bytes();assert sha(raw)==s['source_sha256'];chunks=reader.units(raw.decode()) if u['kind']=='main-unit' else folds(raw.decode());body=chunks[u['new_unit_index']];assert sha(body.encode())==u['unit_sha256'];print(json.dumps(u,ensure_ascii=False));print(body)
