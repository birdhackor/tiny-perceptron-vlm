"""Current course collector checks, bounded CPU and read-only delivery inspection.

Full machine metadata is retained outside Git; stdout is a compact claim receipt.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.request

import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from scripts import export_course
from scripts import build_course

ART=ROOT/'docs/technical-reviews/artifacts/omni-technical-course-v1'
CACHE=Path('/workspace/work/technical-course-current-v1')
CACHE.mkdir(parents=True,exist_ok=True)
FREEZE='docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'
BUNDLE='docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v3.json'
CAT='docs/course-revision-20261029-omni/reviews/reader-evidence-catalogue-technical-fix07-v1.json'
PROPOSAL='docs/course-revision-20261029-omni/reviews/reader-current-proof-proposal-fix13-v1.json'
cache={};hashes={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(path):
    if path not in cache:cache[path]=json.loads((ROOT/path).read_text())
    return cache[path]
def bind(ref):
    p=ROOT/ref['path'];actual=sha(p);assert actual==ref['sha256'],ref['path'];hashes[ref['path']]=actual
    return load(ref['path'])
def pointer(d,p):
    if not p:return d
    for k in p.strip('/').split('/'):
        d=d[int(k)] if isinstance(d,list) else d[k.replace('~1','/').replace('~0','~')]
    return d
def ref(path):return {'path':path,'sha256':sha(ROOT/path)}

def main():
    torch.set_num_threads(1)
    freeze=load(FREEZE);pages={p['page_id']:p for p in freeze['pages']};course=pages['course']
    body=(ROOT/course['snapshot']).read_text();assert sha(ROOT/course['snapshot'])==course['source_sha256']==sha(ROOT/course['source'])
    assert course['figures_sha256']=={}
    bundle=load(BUNDLE);assert sha(ROOT/BUNDLE)=='d8c8015797dff2e030b211499822873f236d54f5c6025f0ac9e08fce4218adba'
    bind(bundle['canonical_source_freeze']);numbered=bind(bundle['section_freeze']);numbered={p['page_id']:p for p in numbered['pages']}
    collection=bind(bundle['actual_collection_snapshot']);issue_index=bind(bundle['issue_inventory'])
    section_rows=[];owners=[];section_details={}
    for row in bundle['sections']:
        current=bind(row['report']);original=bind(row['original_report']);old_freeze=bind(row['original_section_freeze']);old_page=next(p for p in old_freeze['pages'] if p['page_id']==row['lesson_id'])
        assert current['reviewer_task']==original['reviewer_task'];owners.append(current['reviewer_task'])
        assert original['source_sha256']==old_page['source_sha256']==sha(ROOT/old_page['snapshot'])
        assert current['source_sha256']==numbered[row['lesson_id']]['source_sha256']==sha(ROOT/numbered[row['lesson_id']]['snapshot'])
        assert current['figure_sha256']==numbered[row['lesson_id']]['figures_sha256']
        for role,report_ref,report in [('original',row['original_report'],original),('current',row['report'],current)]:
            receipt=row[role+'_collection']
            hits=[r for r in collection['collection_index'] if r['path']==report_ref['path'] and r['sha256']==report_ref['sha256'] and r['received_at']==receipt['received_at']]
            assert len(hits)==1,(row['lesson_id'],role,hits)
            assert datetime.fromisoformat(report['created_at'])<=datetime.fromisoformat(receipt['received_at'])
        if row['lesson_id'].startswith('R.'):
            sid=row['lesson_id'];m=re.search(r'^## '+re.escape(sid)+r' .+$',body,re.M);end=re.search(r'^## ',body[m.end():],re.M);part=body[m.start():m.end()+end.start()] if end else body[m.start():]
            assert hashlib.sha256(part.encode()).hexdigest()==current['source_sha256']
            section_rows.append({'lesson_id':sid,'owner':current['reviewer_task'],'original':row['original_report'],'current':row['report'],'original_created_at':original['created_at'],'current_created_at':current['created_at'],'original_received_at':row['original_collection']['received_at'],'current_received_at':row['current_collection']['received_at'],'source_sha256':current['source_sha256'],'current_verdict':current['verdict'],'original_issue_count':len(original['issues']),'current_unresolved_issues':[i for i in current['issues'] if i['status']!='resolved']})
            section_details[sid]={'original_issues':original['issues'],'current_issues':current['issues'],'original_issue_resolutions':current.get('original_issue_resolutions',[])}
            if sid=='R.1':
                prior=next(s for s in original['sources'] if s['id']=='s_export');new=next(s for s in current['sources'] if s['id']=='s_export')
                snapshot=next(x for x in row['original_source_snapshots'] if x['source']=='scripts/export_course.py')['snapshot']
                assert sha(ROOT/snapshot['path'])==snapshot['sha256']==prior['sha256']=='792f6b269d0c1afd34619d2ea10e1db52a6c19a67ce3365539a612f6ef625874'
                assert new['sha256']==sha(ROOT/'scripts/export_course.py')=='662355e293d75c4a559a7f5bf60b319054de20c63ed6fded6edf82a577f379bf'
    assert len(owners)==26 and len(set(owners))==26
    assert '/root/omni_technical_coordinator/technical_course' not in owners and '/root/omni_reader_entries' not in owners

    index=load('course/lesson-index.json');sync=subprocess.run([sys.executable,'scripts/build_course.py','--check'],cwd=ROOT,capture_output=True,text=True);assert sync.returncode==0,sync.stderr
    stored=ROOT/'outputs/omni-notebooks-technical-fix13-current';site=ROOT/'outputs/omni-review-site-technical-fix13'
    all_source_same=0;nb_same=0;code_cells=0;teaching_cells=0;output_count=0;cpu_html=0;footers=0;details=[]
    for item in index:
        p=pages[item['id']];raw=(ROOT/p['snapshot']).read_text();assert sha(ROOT/p['snapshot'])==p['source_sha256']
        source=(ROOT/item['source']).read_text();match=re.search(r'^## '+re.escape(item['id'])+r' .+$',source,re.M);nxt=re.search(r'^## ',source[match.end():],re.M);live=source[match.start():match.end()+nxt.start()] if nxt else source[match.start():]
        assert live==raw;all_source_same+=1
        notebook=load(item['notebook']);executed=export_course.checked_notebook(notebook,stored/Path(item['notebook']).relative_to('notebooks'))
        download=site/item['notebook'];assert download.read_bytes()==(ROOT/item['notebook']).read_bytes();nb_same+=1
        html=(site/(item['id']+'.html')).read_text()
        assert 'https://colab.research.google.com/' in html and 'https://colab.research.google.com/github/' not in html
        assert '尚未發布的預覽版' in html and 'Upload notebook' in html and 'File → Upload notebook' in html and '雲端環境未經本課實測' in html
        footers+=1;cpu_html+='實際執行結果 · CPU' in html
        for cell in executed['cells']:
            if cell['cell_type']=='code':
                code_cells+=1;output_count+=len(cell.get('outputs',[]))
                if not cell.get('metadata',{}).get('course_setup') and 'course_figure' not in cell.get('metadata',{}):teaching_cells+=1
        details.append({'id':item['id'],'source_sha256':p['source_sha256'],'notebook_sha256':sha(ROOT/item['notebook']),'executed_sha256':sha(stored/Path(item['notebook']).relative_to('notebooks'))})
    links=[]
    for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',body):
        if '://' in target:continue
        rel,_,fragment=target.partition('#');p=((ROOT/course['source']).parent/rel).resolve() if rel else ROOT/course['source'];assert p.is_file()
        if fragment:assert fragment in re.findall(r'^## ([A-Z\d]+\.\d+) ',p.read_text(),re.M)
        links.append(target)

    guards=[]
    def guard(name,items,revision,expected):
        try:export_course.verify_colab_notebooks(items,revision);result='accepted';msg=''
        except ValueError as e:result='rejected';msg=str(e)
        assert result==expected,(name,result,msg)
        guards.append({'case':name,'result':result,'error_first_line':msg.splitlines()[0] if msg else None})
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    guard('symbolic main refused',index,'main','rejected');guard('missing 40hex commit refused',index,'0'*40,'rejected')
    one=next(i for i in index if i['id']=='1.1');assert subprocess.check_output(['git','show',head+':'+one['notebook']],cwd=ROOT)==(ROOT/one['notebook']).read_bytes()
    guard('real commit exact one-notebook subset', [one],head,'accepted')
    guard('old P1 all331 lacks current content',index,'80c5db17e8b077b2c0a5e4e9baa48e760ccb0532','rejected')
    embedded={}
    for pid in ['9.12','9.15']:
        p=pages[pid];blocks=re.findall(r'```python\n(.*?)```',(ROOT/p['snapshot']).read_text(),re.S);ns={};s=io.StringIO()
        with redirect_stdout(s):
            for code in blocks:exec(compile(code,p['snapshot'],'exec'),ns)
        embedded[pid]={'source_sha256':p['source_sha256'],'code_sha256s':[hashlib.sha256(c.encode()).hexdigest() for c in blocks],'stdout':s.getvalue(),'scope':'Exact embedded CPU demonstration; no training or empirical model quality.'}

    cat=load(CAT);proposal=load(PROPOSAL);bind(proposal['source_freeze']);bind(proposal['original_immutable_reader_manifest'])
    for r in proposal['actual_intakes']:bind(r)
    overlays={r['page_id']:r['reader'] for r in proposal['pages']};reader_rows=[];refcount=0
    for row in cat['rows']:
        pid=row['page_id']
        if row['status']=='preserved_inherit_not_new_read':reader_rows.append({'page_id':pid,'status':row['status']});continue
        proof=overlays.get(pid)
        cref=proof['report'] if proof else row['current_report'];cptr=proof['json_pointer'] if proof else row['current_pointer'];bref=proof['basis'] if proof else row['basis']
        cdoc=bind(cref);cp=pointer(cdoc,cptr);bd=bind(bref['report']);bp=pointer(bd,bref['json_pointer']);refcount+=2
        if row['status']=='current_owner_source_identity':assert cp['source_sha256']==bp['source_sha256']==pages[pid]['source_sha256'],pid
        assert cdoc['reviewer_task']==bd['reviewer_task']==row['reviewer_task'];assert cp['verdict']==bp['verdict']=='pass'
        se=bp.get('session_evidence',{})
        if se:
            opening=bind(se['opening']);assert opening['source_sha256']==bp['source_sha256'];refcount+=1
            for nr in se['notes']:bind(nr);refcount+=1
        reader_rows.append({'page_id':pid,'status':row['status'],'source_sha256':cp['source_sha256'],'current':cref,'basis':bref,'overlay_used':pid in overlays})

    http=[]
    for path in ['course.html','1.1.html','notebooks/12/12.17.ipynb']:
        response=urllib.request.urlopen('http://127.0.0.1:8774/'+path,timeout=15);data=response.read();assert data==(site/path).read_bytes()
        http.append({'path':path,'status':response.status,'sha256':hashlib.sha256(data).hexdigest(),'matches_site_bytes':True})
    original=load('docs/course-revision-20261029-omni/reviews/technical/technical_course/technical-original-v1.json')
    for a in original['artifacts']:assert sha(ROOT/a['path'])==a['sha256']
    raw={'notebooks':details,'reader_rows':reader_rows,'section_details':section_details,'verified_reference_hashes':hashes}
    rawpath=CACHE/'current-audit-full-v1.json'
    with rawpath.open('x') as f:json.dump(raw,f,ensure_ascii=False,indent=2);f.write('\n')
    result={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_technical_coordinator/technical_course','environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu; read-only source/Git/HTTP; no model training'},
      'source_freeze':ref(FREEZE),'course_source_sha256':course['source_sha256'],'sidecar':ref(BUNDLE),'sidecar_section_count':len(bundle['sections']),'unique_section_owners':len(set(owners)),'section_rows':section_rows,'collection_status':collection['status'],'issue_inventory':bundle['issue_inventory'],
      'exporter':ref('scripts/export_course.py'),'historical_exporter_snapshot_scope':'792f verified only against original R.1 preserved snapshot; all current checks use6623.',
      'builder_check':{'returncode':sync.returncode,'stdout':sync.stdout.strip()},'notebooks':{'count':len(index),'frozen_source_matches':all_source_same,'current_stored_cpu_matches':nb_same,'site_download_matches':nb_same,'recorded_code_cells':code_cells,'teaching_cells':teaching_cells,'recorded_outputs':output_count,'html_pages_with_nonempty_cpu_output':cpu_html,'upload_footers_verified':footers},
      'local_links_verified':len(links),'colab_release_guard_cases':guards,'guard_scope':'Real Git checks only. Positive is a one-notebook subset, not a matching all331 release; rejected old pin is not a publication result.',
      'embedded_current_demos':embedded,'reader_proof_proposal':ref(PROPOSAL),'reader_overlay_page_ids':sorted(overlays),'reader_reference_checks':refcount,'reader_identity_exceptions':[{'page_id':r['page_id'],'status':r['status']} for r in reader_rows if r['status']!='current_owner_source_identity'],
      'build_info':load('outputs/omni-review-site-technical-fix13/build-info.json'),'http_checks':http,'original_artifacts_unchanged':True,
      'raw_machine_metadata_outside_git':{'path':str(rawpath),'sha256':sha(rawpath),'bytes':rawpath.stat().st_size},'scope':'Metadata/source/delivery facts and two exact short CPU demos. Neither 331 semantic rereads nor 331 new executions, all-book visual acceptance, published site, real cloud execution or new model quality.'}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
