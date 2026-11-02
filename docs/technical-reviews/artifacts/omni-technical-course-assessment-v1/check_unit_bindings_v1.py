"""Bind existing reader notes to exact opened units; compare disclosed prose deltas."""
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re

ROOT=Path(__file__).resolve().parents[4]
CAT='docs/course-revision-20261029-omni/reviews/reader-evidence-catalogue-technical-fix07-v1.json'
cache={}
def load(p):
    if p not in cache:cache[p]=json.loads((ROOT/p).read_text())
    return cache[p]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pointer(d,p):
    for k in p.strip('/').split('/'):
        d=d[int(k)] if isinstance(d,list) else d[k.replace('~1','/').replace('~0','~')]
    return d
def binding(r):
    assert sha(ROOT/r['path'])==r['sha256']
    return load(r['path'])

spec=importlib.util.spec_from_file_location('actual_reader_units',ROOT/'docs/course-revision-20261016/tools/reader.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)

def main():
    cat=load(CAT);current=binding(cat['current_source_freeze']);inventory={p['page_id']:p for p in current['pages']}
    failures=[];sessions=0;notes=0;deltas={}
    for row in cat['rows']:
        if 'basis' not in row:continue
        report=binding(row['basis']['report']);bp=pointer(report,row['basis']['json_pointer']);se=bp['session_evidence']
        opening=binding(se['opening']);sessions+=1
        if opening['page_id']!=row['page_id'] or opening['reviewer_task']!=row['reviewer_task'] or opening['source_sha256']!=bp['source_sha256']:
            failures.append({'page_id':row['page_id'],'reason':'opening owner/page/source mismatch'})
        frozen=binding(opening['source_freeze']);page=next(p for p in frozen['pages'] if p['page_id']==row['page_id'])
        text=(ROOT/page['snapshot']).read_text();assert sha(ROOT/page['snapshot'])==page['source_sha256']
        chunks=reader.units(text)
        got=[]
        for ref in se['notes']:
            note=binding(ref);notes+=1;i=note['unit_index'];got.append(i)
            if note['page_id']!=row['page_id'] or note['reviewer_task']!=row['reviewer_task'] or i>=len(chunks) or note['unit_sha256']!=hashlib.sha256(chunks[i].encode()).hexdigest():
                failures.append({'page_id':row['page_id'],'note':ref['path'],'reason':'unit/page/owner mismatch'})
        if got!=list(range(len(chunks))):failures.append({'page_id':row['page_id'],'reason':'note index coverage mismatch','got':got,'unit_count':len(chunks)})
        if row['page_id'] in ['1.11','first-steps']:
            now=(ROOT/inventory[row['page_id']]['snapshot']).read_text()
            fenced=lambda s:re.findall(r'```[^\n]*\n(.*?)```',s,re.S)
            deltas[row['page_id']]={'old_snapshot':page['snapshot'],'old_sha256':page['source_sha256'],'current_snapshot':inventory[row['page_id']]['snapshot'],'current_sha256':inventory[row['page_id']]['source_sha256'],
              'fenced_code_count_old':len(fenced(text)),'fenced_code_count_current':len(fenced(now)),'all_fenced_code_equal':fenced(text)==fenced(now),
              'diff':''.join(difflib.unified_diff(text.splitlines(True),now.splitlines(True),fromfile=page['snapshot'],tofile=inventory[row['page_id']]['snapshot'],n=3))}
    manifest=binding(cat['manifest']);closures={}
    for pid in ['1.2','19.2','7.26','16.15']:
        proof=next(p for p in manifest['pages'] if p['page_id']==pid)['reader']['checkpoint_closure']
        doc=binding(proof['report']);entry=pointer(doc,proof['json_pointer'])
        closures[pid]={'reference':proof,'owner':doc.get('reviewer_task'),'entry':entry}
    out={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_technical_coordinator/technical_course',
      'environment':{'python':platform.python_version(),'device':'CPU metadata comparison'},
      'scope':'Exact source/opening/note unit binding and disclosed delta comparison only, not semantic reading of every unit.',
      'reader_tool':{'path':'docs/course-revision-20261016/tools/reader.py','sha256':sha(ROOT/'docs/course-revision-20261016/tools/reader.py')},
      'sessions_checked':sessions,'notes_checked':notes,'binding_failures':failures,'disclosed_deltas':deltas,'selected_closures':closures,
      'python_floor_example':{'expression':'-7 // 2','observed':-7//2,'expected':-4},
      'author_patch_coefficient_ratio':{'expression':'2/0.5','observed':2/0.5,'scope':'Technical check of current wording; does not relabel the author patch as an owner fresh reread.'}}
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
