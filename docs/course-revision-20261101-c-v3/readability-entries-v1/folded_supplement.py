"""原判真收件後，逐個自洽折疊區補讀；不回填正文首讀。"""
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'supplements'/'folded-v1'
FREEZE=ROOT/'docs/course-revision-20261101-c-v3/source-freeze-v19.json'
freeze=json.loads(FREEZE.read_bytes())
pages={p['page_id']:p for p in freeze['pages']}
units=[]
for pid in ['readme','course','training','asset-storage','curriculum','validation','index']:
 p=pages[pid]
 raw=(ROOT/p['snapshot']).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==p['source_sha256']
 for i,m in enumerate(re.finditer(r'<details\b[^>]*>.*?</details>',raw.decode(),re.S)):
  label=re.search(r'<summary[^>]*>(.*?)</summary>',m[0],re.S)[1]
  units.append({'page_id':pid,'detail_index':i,'label':label,'source_sha256':p['source_sha256'],'block':m[0],'unit_sha256':hashlib.sha256(m[0].encode()).hexdigest()})

def write_new(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')

operation,*rest=sys.argv[1:]
statepath=OUT/'state.json'
if operation=='start':
 state={'kind':'post_original_folded_supplement','reviewer_task':'/root/c_v3_reader_entries','started_at':datetime.now(UTC).isoformat(),'next_index':0,'units_total':len(units),'original_report_sha256':'27115271a2313bfc55b4ce146e961c660ef222841517f26043e990e5c6a3a239','parent_collection':'actual-reader-originals-collected-02.json (parent message; not read peer report)','not_original_main_first_read':True}
 write_new(OUT/'opening.json',state);write_new(statepath,state)
else:
 state=json.loads(statepath.read_bytes());i=state['next_index'];n=json.loads(rest[0]);assert n['unit_index']==i
 u=units[i]
 n.update({k:u[k] for k in ['page_id','detail_index','label','source_sha256','unit_sha256']},saved_at=datetime.now(UTC).isoformat(),reviewer_task=state['reviewer_task'],kind='post_original_folded_supplement',not_original_main_first_read=True)
 write_new(OUT/f'note-{i:03}.json',n)
 state['next_index']+=1;statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
i=state['next_index']
print(json.dumps({'unit_index':i,'units_total':len(units),'complete':i==len(units)},ensure_ascii=False))
if i<len(units):
 u=units[i];print(json.dumps({k:u[k] for k in ['page_id','detail_index','label','source_sha256','unit_sha256']},ensure_ascii=False));print(u['block'])
