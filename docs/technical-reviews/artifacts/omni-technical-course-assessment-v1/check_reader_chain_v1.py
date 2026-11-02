"""Bounded metadata binding audit, deliberately not a semantic full-book review."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re

ROOT=Path(__file__).resolve().parents[4]
CAT='docs/course-revision-20261029-omni/reviews/reader-evidence-catalogue-technical-fix07-v1.json'
SELECT={'1.2','1.11','11.16','12.17','16.15','19.2','20.10','7.26','9.11','course','first-steps','glossary','A.2'}
cache={}; checked={}; failures=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(path):
    if path not in cache:cache[path]=json.loads((ROOT/path).read_text())
    return cache[path]
def ref(r):
    p=ROOT/r['path']; actual=sha(p) if p.is_file() else None
    checked[r['path']]=actual
    if actual!=r['sha256']:failures.append({'path':r['path'],'expected':r['sha256'],'actual':actual})
    return load(r['path'])
def ptr(d,p):
    if p in ('','/'):return d
    for k in p.strip('/').split('/'):
        k=k.replace('~1','/').replace('~0','~');d=d[int(k)] if isinstance(d,list) else d[k]
    return d

def main():
    cat=load(CAT);freeze=ref(cat['current_source_freeze']);manifest=ref(cat['manifest'])
    inventory={p['page_id']:p for p in freeze['pages']}
    assert set(inventory)=={r['page_id'] for r in cat['rows']}
    counts=Counter();owner_counts=Counter();note_count=0;selected={};exceptions=[]
    for row in cat['rows']:
        pid=row['page_id'];counts[row['status']]+=1
        page=inventory[pid];snapshot=ROOT/page['snapshot']
        if sha(snapshot)!=page['source_sha256'] or page['source_sha256']!=row['current_source_sha256']:
            failures.append({'page_id':pid,'reason':'current freeze/snapshot/catalogue mismatch'})
        if 'current_report' not in row:
            exceptions.append({'page_id':pid,'status':row['status'],'reason':'preserved inherit; no new reader report asserted'})
            continue
        current=ref(row['current_report']);cpage=ptr(current,row['current_pointer'])
        basis=ref(row['basis']['report']);bpage=ptr(basis,row['basis']['json_pointer'])
        owner_counts[current.get('reviewer_task','missing')]+=1
        expected_same=row['status']=='current_owner_source_identity'
        if expected_same and (cpage['source_sha256']!=page['source_sha256'] or bpage['source_sha256']!=page['source_sha256']):
            failures.append({'page_id':pid,'reason':'current or basis source differs from current identity claim'})
        if cpage.get('verdict')!='pass' or bpage.get('verdict')!='pass':
            failures.append({'page_id':pid,'reason':'current/basis verdict is not pass'})
        if current.get('reviewer_task')!=row['reviewer_task'] or basis.get('reviewer_task')!=row['reviewer_task']:
            failures.append({'page_id':pid,'reason':'owner differs from actual report'})
        session=bpage.get('session_evidence',{})
        notes=session.get('notes',[])
        opening=ref(session['opening']) if 'opening' in session else None
        unit_indices=[];note_sources=[];note_keys=set()
        for nr in notes:
            n=ref(nr);note_count+=1;unit_indices.append(n.get('unit_index'));note_keys.update(n)
            note_sources.append(n.get('source_sha256'))
        if pid in SELECT:
            selected[pid]={'current_report':row['current_report'],'current_pointer':row['current_pointer'],'basis':row['basis'],
                'status':row['status'],'current_source_sha256':page['source_sha256'],'reviewed_source_sha256':cpage['source_sha256'],
                'basis_source_sha256':bpage['source_sha256'],'session':bpage.get('session'),'session_evidence':session,
                'note_unit_indices':unit_indices,'note_source_values':sorted(set(str(v) for v in note_sources)),
                'actual_note_keys':sorted(note_keys),'opening_keys':list(opening) if opening else [],
                'basis_reader_summary':bpage.get('reader_summary'),'basis_check_count':len(bpage.get('checks',[]))}
        if not expected_same:
            exceptions.append({'page_id':pid,'status':row['status'],'current_sha256':page['source_sha256'],'reviewed_sha256':cpage['source_sha256'],'relied_author_patch_ids':row.get('relied_author_patch_ids',[]),'fenced_code_unchanged_claim':row.get('all_fenced_code_unchanged_from_prior_reviewed_source')})
    seal_path='docs/course-revision-20261029-omni/reviews/assembly-reader-01/reader-seal.json'
    seal=load(seal_path)
    out={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_technical_coordinator/technical_course',
         'environment':{'python':platform.python_version(),'device':'CPU metadata reads only'},
         'scope':'All catalogue source/report/basis/opening/note digests and pointer metadata; selected semantic reading performed separately. This output does not prove explanations sufficient or claim full-book reading.',
         'catalogue':{'path':CAT,'sha256':sha(ROOT/CAT)},'freeze':cat['current_source_freeze'],'manifest':cat['manifest'],
         'canonical_pages':len(inventory),'counts':dict(counts),'owner_counts':dict(owner_counts),'basis_note_references_checked':note_count,
         'unique_report_or_note_files_checked':len(checked),'checked_reference_set_sha256':hashlib.sha256(json.dumps(checked,sort_keys=True).encode()).hexdigest(),
         'binding_failures':failures,'exceptions':exceptions,'selected_pages':selected,
         'seal':{'path':seal_path,'sha256':sha(ROOT/seal_path),'role':seal['role'],'status':seal['status'],'started_at':seal['started_at'],'completed_at':seal['completed_at'],'scope':seal.get('scope'),'root_confirmation_pending':seal.get('root_confirmation_pending'),'report_sha256s_count':len(seal['report_sha256s']),'collection_index_count':len(seal['collection_index'])},
         'manifest_review_policy':manifest['review_policy'],'manifest_criteria':manifest['criteria']}
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
