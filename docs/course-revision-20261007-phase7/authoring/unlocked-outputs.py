"""Return actual CPU outputs for code already unlocked by the coordinator; no future text or outputs."""
import argparse,hashlib,importlib.util,json,re,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('session');p.add_argument('--page-id');a=p.parse_args();root=Path('.').resolve();sys.path.insert(0,str(root));assert re.fullmatch('[0-9a-f]{32}',a.session)
spec=importlib.util.spec_from_file_location('phase7_materials',root/'docs/review-tools/phase7_review.py');tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
state=tool.read_json(root/'outputs/grouped-review'/a.session/'state.json');visible=state['units'][:min(state['cursor']+1,len(state['units']))]
pid=a.page_id or visible[-1]['page_id'];units=[u for u in visible if u['page_id']==pid];assert units,'Page has not been unlocked.'
manifest=tool.load_manifest(root,root/state['manifest']);entry=next(x for x in manifest['pages'] if x['page_id']==pid)
if 'notebook' not in entry:print(json.dumps({'page_id':pid,'available':False,'reason':'This guide is not an executed lesson notebook.'}));raise SystemExit()
source=root/entry['notebook'];assert tool.sha(source.read_bytes())==entry['notebook_sha256'],'Source notebook changed after this freeze.'
executed=root/'outputs'/entry['notebook'];assert executed.is_file(),'No stored CPU execution exists.'
src=tool.read_json(source);nb=tool.export_course.checked_notebook(src,executed)
codes=[]
for u in units:
 for body in re.findall(r'^```python\s*\n(.*?)^```',u['text'],flags=re.M|re.S):codes.append((u,body.strip()))
last=-1;results=[]
for u,body in codes:
 matches=[(i,c) for i,c in enumerate(nb['cells']) if i>last and c['cell_type']=='code' and tool.export_course.joined(c['source']).strip()==body]
 assert matches,'Visible code does not match the executed notebook.'
 idx,cell=matches[0];last=idx
 record={'kind':'actual_unlocked_code_outputs_not_author_explanation','page_id':pid,'unit_index':u['unit_index'],'unit_sha256':u['unit_sha256'],'notebook':entry['notebook'],'source_notebook_sha256':entry['notebook_sha256'],'executed_notebook_sha256':tool.sha(executed.read_bytes()),'code_cell_index':idx,'code_sha256':hashlib.sha256(body.encode()).hexdigest(),'outputs':cell.get('outputs',[])}
 folder=root/'docs/course-revision-20261007-phase7/reviews/artifacts/unlocked-outputs';folder.mkdir(parents=True,exist_ok=True);raw=(json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode();target=folder/(hashlib.sha256(raw).hexdigest()+'.json')
 if not target.exists():target.write_bytes(raw)
 text=[]
 for out in record['outputs']:
  if out['output_type']=='stream':text.append(tool.export_course.joined(out['text']))
  elif out['output_type'] in ('display_data','execute_result') and 'text/plain' in out.get('data',{}):text.append(tool.export_course.joined(out['data']['text/plain']))
  elif out['output_type']=='error':text.append('Execution recorded an error: '+str(out.get('evalue','')))
 results.append({'unit_index':u['unit_index'],'code_cell_index':idx,'artifact':{'path':tool.relative(root,target),'sha256':tool.sha(target.read_bytes())},'actual_stdout_or_plaintext':text})
print(json.dumps({'page_id':pid,'available':bool(results),'scope':'Only code in already unlocked units. These are actual stored outputs, not newly executed or author-supplied answers.','results':results},ensure_ascii=False,indent=2))
