from pathlib import Path
import hashlib,json,sys,subprocess,re
import torch
OUT=Path(__file__).resolve().parent;PRIOR=OUT.parent;ROOT=OUT.parents[5]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
old=json.loads((PRIOR/'report.json').read_text());reads=json.loads((PRIOR/'input-read-receipts.json').read_text())
assert sha(PRIOR/'report.json')=='d9482cdfc38766e00160b823235f819090b24d7e512c88103cf6d94a25abb5c5'
record={'reviewer_task':'/root/v4_review_coordinator/factual_whole_entries','environment':{'python':sys.version,'torch':torch.__version__,'device':'cpu','cuda_available':str(torch.cuda.is_available())},'prior_report':{'path':str((PRIOR/'report.json').relative_to(ROOT)),'sha256':sha(PRIOR/'report.json'),'verdict':old['verdict'],'document_sha256':old['current_document_sha256']},'limits':'Rechecks actual current source and dependency bytes and a small notebook inventory; previous independently executed CPU/empirical/remote metadata results are reused explicitly, with no implied rerun, GPU training, fresh download or cloud Colab execution.'}
assert torch.__version__=='2.14.1+cpu' and not torch.cuda.is_available()
record['complete_documents']={}
for f in ['README.md','README_en.md','course/README.md','assets/training/README.md']:
 data=(ROOT/f).read_bytes();(OUT/(f.replace('/','__')+'.read-snapshot')).write_bytes(data)
 record['complete_documents'][f]={'sha256':sha(ROOT/f),'lines':len(data.decode().splitlines()),'fully_read':True,'scope_note':'First complete factual English reading in this round' if f=='README_en.md' else 'Complete current-source reread'}
record['canonical_read_dependencies']=[]
for f,h in reads['canonical_read_inputs'].items():
 actual=sha(ROOT/f);record['canonical_read_dependencies'].append({'path':f,'previous_sha256':h,'current_sha256':actual,'exact_equal':actual==h})
 assert actual==h or f=='README.md'
record['prior_artifacts_unchanged']=[]
for a in old['artifacts']:
 actual=sha(ROOT/a['path']);assert actual==a['sha256']
 record['prior_artifacts_unchanged'].append({'id':a['id'],'path':a['path'],'sha256':actual,'exact_equal':True})
record['prior_original_authorities_unchanged']=[]
for s in old['sources']:
 if 'raw_retrieval_path' in s:
  actual=sha(ROOT/s['raw_retrieval_path']);assert actual==s['retrieval_sha256']
  record['prior_original_authorities_unchanged'].append({'id':s['id'],'path':s['raw_retrieval_path'],'sha256':actual,'exact_equal':True})
record['prior_fixed_generation_records_unchanged']=[]
emp=json.loads((PRIOR/'empirical-record.json').read_text())
for key,a in emp['capstone'].items():
 actual=sha(ROOT/a['original_path']);assert actual==a['sha256']
 record['prior_fixed_generation_records_unchanged'].append({'variant':key,'path':a['original_path'],'sha256':actual,'reused_recomputed':a['recomputed']})
for p,h in [(emp['natural_selection']['source_path'],emp['natural_selection']['source_sha256']),('docs/validation-artifacts/release-compatibility-ci.json',emp['ci_record']['source_sha256']),('outputs/lfs-remote-validation.json',emp['original_anonymous_asset_record_audit']['record_sha256'])]:
 assert sha(ROOT/p)==h;record['prior_fixed_generation_records_unchanged'].append({'path':p,'sha256':h,'exact_equal':True})
record['notebook_inventory']=[]
for p in sorted((ROOT/'notebooks').glob('*/*.ipynb')):
 n=json.loads(p.read_text());code=[i for i,c in enumerate(n['cells']) if c['cell_type']=='code'];first=n['cells'][0]
 setup=bool(code and ''.join(n['cells'][code[0]]['source']).startswith('# @title 準備本節的工具'))
 if code:assert setup
 record['notebook_inventory'].append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'cells':len(n['cells']),'first_cell_type':first['cell_type'],'first_cell_source':''.join(first['source']),'first_code_cell_index':code[0] if code else None,'first_code_is_setup':setup})
assert len(record['notebook_inventory'])==266
record['notebook_summary']={'total':266,'first_cell_markdown':sum(n['first_cell_type']=='markdown' for n in record['notebook_inventory']),'with_code_setup':sum(n['first_code_is_setup'] for n in record['notebook_inventory']),'without_code':[n['path'] for n in record['notebook_inventory'] if n['first_code_cell_index'] is None]}
assert record['notebook_summary']=={'total':266,'first_cell_markdown':266,'with_code_setup':264,'without_code':['notebooks/20/20.2.ipynb','notebooks/20/20.8.ipynb']}
# Verify the reused actual notebook execution and figure are the same bytes as the prior pinned source.
prior_revision='f8ca78bc3ffef82b0faa352c464909b76bfd8976'
record['reuse_exact_byte_comparisons']=[]
for f in ['notebooks/01/1.1.ipynb','course/figures/character_ids.svg']:
 oldbytes=subprocess.check_output(['git','show',f'{prior_revision}:{f}'],cwd=ROOT);data=(ROOT/f).read_bytes();assert data==oldbytes
 record['reuse_exact_byte_comparisons'].append({'path':f,'current_sha256':sha(ROOT/f),'previous_pinned_sha256':hashlib.sha256(oldbytes).hexdigest(),'exact_equal':True})
record['current_counts']={}
headings=re.compile(r'^## ([A-Z\d]+\.\d+) ',re.M)
chapters=sorted((ROOT/'course/chapters').glob('*.md'))
front={f:len(headings.findall((ROOT/'course'/f).read_text())) for f in ['README.md','first-steps.md','training.md','glossary.md']}
record['current_counts']={'chapter_files':len(chapters),'chapter_lessons':sum(len(headings.findall(p.read_text())) for p in chapters),'front':front,'total_numbered':sum(len(headings.findall(p.read_text())) for p in chapters)+sum(front.values()),'notebooks':len(record['notebook_inventory'])}
assert record['current_counts']==json.loads((PRIOR/'cpu-record.json').read_text())['counts']
record['prior_archive_bytes_unchanged']=[]
manifest=json.loads((ROOT/'assets/training/manifest.json').read_text())
for a in manifest['assets']:
 actual=sha(ROOT/a['archive']);assert actual==a['archive_sha256']
 record['prior_archive_bytes_unchanged'].append({'path':a['archive'],'sha256':actual,'bytes':(ROOT/a['archive']).stat().st_size,'exact_equal':True})
record['necessary_prerequisite_sections_unchanged']=[]
def section_bytes(text,id):
 headings=list(re.finditer(r'^## .+$',text,re.M))
 for i,h in enumerate(headings):
  if h[0].startswith('## '+id+' '):return text[h.start():headings[i+1].start() if i+1<len(headings) else len(text)].encode()
 raise AssertionError(id)
for f,ids in [('course/chapters/01.md',['1.1']),('course/chapters/03.md',['3.1']),('course/chapters/19.md',['19.2','19.12']),('course/chapters/20.md',['20.1','20.8','20.13'])]:
 text=(ROOT/f).read_text();prior=subprocess.check_output(['git','show',f'f8ca78bc3ffef82b0faa352c464909b76bfd8976:{f}'],cwd=ROOT).decode()
 for id in ids:
  data=section_bytes(text,id);original=section_bytes(prior,id);assert data==original
  record['necessary_prerequisite_sections_unchanged'].append({'path':f,'section':id,'current_section_sha256':hashlib.sha256(data).hexdigest(),'previous_read_pinned_sha256':hashlib.sha256(original).hexdigest(),'exact_equal':True,'read_scope':'Previously personally read complete prerequisite section; exact current bytes unchanged.'})
oldtext=(PRIOR/'README.md.read-snapshot').read_text();current=(ROOT/'README.md').read_text()
record['former_c05']={'old_location':'README.md:17','old_assertion':'影片目前提供逐幀切片的輔助函式','old_assertion_present': '影片目前提供逐幀切片的輔助函式' in oldtext,'current_assertion_absent': '影片目前提供逐幀切片的輔助函式' not in current,'current_line17':current.splitlines()[16],'closure_judgment':'Prior unsupported existing-helper assertion was removed. This does not establish any video helper implementation.'}
assert record['former_c05']['old_assertion_present'] and record['former_c05']['current_assertion_absent']
(OUT/'current-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('Actual current torch',torch.__version__, 'device CPU; prior final preserved',record['prior_report']['sha256'])
print('Four full-file snapshots',record['complete_documents'])
print('Actual notebook inventory',record['notebook_summary'])
print('Unchanged reuse dependencies',len(record['canonical_read_dependencies']),len(record['prior_original_authorities_unchanged']),len(record['prior_fixed_generation_records_unchanged']))
