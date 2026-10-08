from pathlib import Path
import json,hashlib,re,argparse,importlib.util
from datetime import datetime,timezone
B=Path(__file__).resolve().parent
M=json.loads((B/'manifest.json').read_text());P={p['page_id']:p for p in M['inventory']['pages']}
spec=importlib.util.spec_from_file_location('audit_units','/workspace/tiny-perceptron-vlm/docs/review-tools/incremental_reader.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def now():return datetime.now(timezone.utc).isoformat()
def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def page(pid,mode='main'):
 raw=(B/'freeze/sources'/f'{pid}.md').read_text()
 if mode=='full':return raw
 def hide(m):
  label=re.search(r'<summary[^>]*>(.*?)</summary>',m[0],re.S)
  return '\n[原位置選讀折疊區：'+(re.sub('<[^>]+>','',label[1]).strip() if label else '選讀')+'；正文路線完成後補讀]\n'
 return re.sub(r'<details\b[^>]*>.*?</details>',hide,raw,flags=re.S)
def append(group,event):
 with (B/'traces'/f'reader-{group}.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
def state(group):
 path=B/'sessions'/f'{group}.json'
 if not path.exists():
  s={'group':group,'page_index':0,'unit_index':0,'reviewer':M['groups'][group]['reviewer'],'phase':'main','began_at':now()};path.write_text(json.dumps(s))
  append(group,{'kind':'begin','at':s['began_at'],'reviewer':s['reviewer'],'criteria_sha256':M['criteria_sha256'],'route':'grouped sequential main-only; read prerequisites recorded separately'})
 return json.loads(path.read_text()),path
def show(s):
 ids=M['groups'][s['group']]['pages']
 if s['page_index']>=len(ids):
  print(json.dumps({'complete':True,'phase':s['phase'],'pages':len(ids),'next':'Seal main-only findings before reading optional sections.'}));return
 pid=ids[s['page_index']];text=page(pid,'full' if s['phase']=='extras' else 'main');units=mod.units(text)
 print(json.dumps({'page_id':pid,'source':P[pid]['source'],'selector':P[pid]['selector'],'source_sha256':P[pid]['source_sha256'],'phase':s['phase'],'unit_index':s['unit_index'],'units_total':len(units),'page_index':s['page_index'],'pages_total':len(ids)},ensure_ascii=False))
 if units:
  body=units[s['unit_index']];print(body)
  print('Current-image references:',json.dumps(re.findall(r'!\[[^\]]*\]\(([^)]+)\)',body),ensure_ascii=False))
 else:print('[Empty introduction]')
p=argparse.ArgumentParser();p.add_argument('command',choices=['show','next','context','seal','extras']);p.add_argument('group',choices=list(M['groups']));p.add_argument('--note',type=Path);p.add_argument('--page');a=p.parse_args();s,path=state(a.group)
if a.command=='show':show(s)
elif a.command=='next':
 assert a.note and a.note.exists(),'save your own checkpoint first'
 note=json.loads(a.note.read_text());assert note.get('reviewer')==s['reviewer'],'actual reviewer identity required'
 ids=M['groups'][a.group]['pages'];pid=ids[s['page_index']];units=mod.units(page(pid,'full' if s['phase']=='extras' else 'main'))
 assert note.get('page_id')==pid and note.get('unit_index')==s['unit_index'],'note must match current source'
 for field in ['understanding','materials_and_labels','expected_change','confusion_and_quote','missing_visuals']:
  assert isinstance(note.get(field),str) and note[field].strip(), 'missing checkpoint '+field
 append(a.group,{'kind':'checkpoint','at':now(),'phase':s['phase'],'page_id':pid,'unit_index':s['unit_index'],'unit_sha256':sha(units[s['unit_index']]) if units else sha(''),'source_sha256':P[pid]['source_sha256'],'note':note})
 s['unit_index']+=1
 if s['unit_index']>=len(units):
  append(a.group,{'kind':'page_end','at':now(),'phase':s['phase'],'page_id':pid,'source_sha256':P[pid]['source_sha256']});s['page_index']+=1;s['unit_index']=0
 path.write_text(json.dumps(s));show(s)
elif a.command=='context':
 assert a.page in P,'unknown canonical prerequisite'
 assigned=M['groups'][a.group]['pages']
 if s['phase']=='main' and a.page in assigned:
  assert assigned.index(a.page)<s['page_index'],'context may not expose current or future assigned page; use show/next'
 append(a.group,{'kind':'prerequisite_read','at':now(),'page_id':a.page,'source_sha256':P[a.page]['source_sha256'],'phase':s['phase']})
 print(json.dumps(P[a.page],ensure_ascii=False));print(page(a.page))
elif a.command=='seal':
 assert s['page_index']>=len(M['groups'][a.group]['pages']) and s['phase']=='main','finish main-only route first'
 report=B/'reports'/f'reader-{a.group}-main.json';assert report.exists(),'save actual main report first'
 digest=hashlib.sha256(report.read_bytes()).hexdigest()
 receipt=B/'reports'/f'reader-{a.group}-main.seal.json'
 assert not receipt.exists(),'do not replace original main seal'
 receipt.write_text(json.dumps({'reviewer':s['reviewer'],'at':now(),'path':str(report),'sha256':digest},ensure_ascii=False,indent=2)+'\n')
 append(a.group,{'kind':'sealed_main_report','at':now(),'path':str(report),'sha256':digest})
 print(json.dumps({'main_sealed':True,'sha256':digest,'next':'Wait for root global main gate before extras.'}))
elif a.command=='extras':
 assert (B/'checks/reader-main-all-sealed.json').exists(),'wait for all group main seals and root global gate'
 receipt=json.loads((B/'reports'/f'reader-{a.group}-main.seal.json').read_text())
 assert hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest()==receipt['sha256'],'main report drift'
 assert s['page_index']>=len(M['groups'][a.group]['pages']) and s['phase']=='main','finish main-only route first'
 report=B/'reports'/f'reader-{a.group}-main.json';assert report.exists(),'save and seal main-only page findings first'
 raw=report.read_bytes();append(a.group,{'kind':'sealed_main_report','at':now(),'path':str(report),'sha256':hashlib.sha256(raw).hexdigest()})
 s.update({'phase':'extras','page_index':0,'unit_index':0});path.write_text(json.dumps(s));print('Main-only judgments sealed. Optional sections may now be checked; preserve initial records.');show(s)
