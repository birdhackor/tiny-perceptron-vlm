from pathlib import Path
import hashlib,json,re,sys
import torch
OUT=Path(__file__).resolve().parent;PRIOR=OUT.parent;ROOT=OUT.parents[5];R2=PRIOR/'round2'
sys.path.insert(0,str(ROOT))
from scripts.build_course import BOOTSTRAP

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
old=read(PRIOR/'report.json');last=read(R2/'report.json');prior=read(R2/'current-record.json')
assert sha(PRIOR/'report.json')=='d9482cdfc38766e00160b823235f819090b24d7e512c88103cf6d94a25abb5c5'
assert sha(R2/'report.json')=='8037f2307fda5919c639162d8532b454baf42379349386c2c063a9871ba66310'
record={'reviewer_task':'/root/v4_review_coordinator/factual_whole_entries','review_round':3,'environment':{'python':sys.version,'torch':torch.__version__,'device':'cpu','cuda_available':str(torch.cuda.is_available())},'limits':'Actual current full-source snapshots, hashes/counts/ordered-cell inventory only. Previously executed bounded CPU and fixed original GPU-record audits are explicitly reused after exact-byte checks. No prior probe rerun, Git command, setup install, model/data download, training or cloud Colab execution.'}
assert torch.__version__=='2.14.1+cpu' and not torch.cuda.is_available()
record['preserved_reports']=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'verdict':read(p)['verdict'],'source_hashes':read(p)['current_document_sha256']} for p in [PRIOR/'report.json',R2/'report.json']]
expected={'README.md':'0754cb4ce67ebad205f4db4edc28a9a0f8e611f1e9cddca11c10b925e0e59b73','README_en.md':'430d503f88f966aae21314a0c9c14f2be4be29e808a0293bfe08b7c2218527ca','course/README.md':'2434fa73f4c2fc800f4d336523dd3ce68a6d9dcaaac6bba372b8dc9d03a19fa4','assets/training/README.md':'675d58b211e7eb14752586cb6244da6ac9ed2b8b58341d7033d04ac65900a10d'}
record['complete_documents']={}
for f,h in expected.items():
 data=(ROOT/f).read_bytes();assert sha(ROOT/f)==h
 (OUT/(f.replace('/','__')+'.read-snapshot')).write_bytes(data)
 record['complete_documents'][f]={'sha256':h,'lines':len(data.decode().splitlines()),'fully_read':True,'prior_round_sha256':last['current_document_sha256'][f],'exact_equal_to_prior_round':h==last['current_document_sha256'][f]}
record['canonical_read_dependencies']=[]
for a in prior['canonical_read_dependencies']:
 actual=sha(ROOT/a['path']);same=actual==a['current_sha256'];assert same or a['path']=='README.md'
 record['canonical_read_dependencies'].append({'path':a['path'],'previous_sha256':a['current_sha256'],'current_sha256':actual,'exact_equal':same})
record['previous_evidence_unchanged']=[]
for a in last['artifacts']:
 actual=sha(ROOT/a['path']);assert actual==a['sha256']
 record['previous_evidence_unchanged'].append({'id':a['id'],'path':a['path'],'sha256':actual,'exact_equal':True})
record['original_authorities_unchanged']=[]
for s in last['sources']:
 if 'raw_retrieval_path' in s:
  actual=sha(ROOT/s['raw_retrieval_path']);assert actual==s['retrieval_sha256']
  record['original_authorities_unchanged'].append({'id':s['id'],'path':s['raw_retrieval_path'],'sha256':actual,'exact_equal':True,'inspection':'Previously personally inspected original authority text; no duplicate retrieval or new claim of reading another reviewer.'})
record['fixed_records_unchanged']=[]
for a in prior['prior_fixed_generation_records_unchanged']:
 actual=sha(ROOT/a['path']);assert actual==a['sha256'];record['fixed_records_unchanged'].append({'path':a['path'],'sha256':actual,'exact_equal':True,'reused_recomputation':a.get('reused_recomputed')})
record['archive_bytes_unchanged']=[]
for a in prior['prior_archive_bytes_unchanged']:
 actual=sha(ROOT/a['path']);assert actual==a['sha256'];record['archive_bytes_unchanged'].append({'path':a['path'],'sha256':actual,'bytes':(ROOT/a['path']).stat().st_size,'exact_equal':True})
record['notebook_inventory']=[];oldnb={n['path']:n for n in prior['notebook_inventory']}
for p in sorted((ROOT/'notebooks').glob('*/*.ipynb')):
 n=read(p);code=[i for i,c in enumerate(n['cells']) if c['cell_type']=='code'];name=str(p.relative_to(ROOT));h=sha(p)
 setup=bool(code and ''.join(n['cells'][code[0]]['source']).rstrip('\n')==BOOTSTRAP.rstrip('\n') and n['cells'][code[0]]['metadata'].get('course_setup') is True)
 if code:assert setup
 record['notebook_inventory'].append({'path':name,'sha256':h,'previous_sha256':oldnb[name]['sha256'],'exact_equal_to_previous':h==oldnb[name]['sha256'],'cell_count':len(n['cells']),'first_cell_type':n['cells'][0]['cell_type'],'first_cell_source':''.join(n['cells'][0]['source']),'first_code_cell_index':code[0] if code else None,'first_code_exact_generator_bootstrap':setup})
record['notebook_summary']={'total':len(record['notebook_inventory']),'first_cell_markdown':sum(n['first_cell_type']=='markdown' for n in record['notebook_inventory']),'with_exact_first_code_setup':sum(n['first_code_exact_generator_bootstrap'] for n in record['notebook_inventory']),'without_code':[n['path'] for n in record['notebook_inventory'] if n['first_code_cell_index'] is None],'changed_notebook_paths':[n['path'] for n in record['notebook_inventory'] if not n['exact_equal_to_previous']]}
assert record['notebook_summary']['total']==266 and record['notebook_summary']['with_exact_first_code_setup']==264 and record['notebook_summary']['without_code']==['notebooks/20/20.2.ipynb','notebooks/20/20.8.ipynb']
headings=re.compile(r'^## ([A-Z\d]+\.\d+) ',re.M);chapters=sorted((ROOT/'course/chapters').glob('*.md'));front={f:len(headings.findall((ROOT/'course'/f).read_text())) for f in ['README.md','first-steps.md','training.md','glossary.md']}
record['current_counts']={'chapter_files':len(chapters),'chapter_lessons':sum(len(headings.findall(p.read_text())) for p in chapters),'front':front,'total_numbered':sum(len(headings.findall(p.read_text())) for p in chapters)+sum(front.values()),'notebooks':len(record['notebook_inventory'])};assert record['current_counts']==prior['current_counts']
def section(text,id):
 hs=list(re.finditer(r'^## .+$',text,re.M))
 for i,h in enumerate(hs):
  if h[0].startswith('## '+id+' '):return text[h.start():hs[i+1].start() if i+1<len(hs) else len(text)].encode()
 raise AssertionError(id)
record['necessary_prerequisites']=[]
for a in prior['necessary_prerequisite_sections_unchanged']:
 data=section((ROOT/a['path']).read_text(),a['section']);h=hashlib.sha256(data).hexdigest();assert h==a['current_section_sha256']
 record['necessary_prerequisites'].append({'path':a['path'],'section':a['section'],'current_section_sha256':h,'previous_read_section_sha256':a['current_section_sha256'],'exact_equal':True,'inspection':'Previously personally read complete section, current exact bytes reused.'})
data=section((ROOT/'course/chapters/19.md').read_text(),'19.10');(OUT/'prerequisite19-10.read-snapshot').write_bytes(data)
record['necessary_prerequisites'].append({'path':'course/chapters/19.md','section':'19.10','current_section_sha256':hashlib.sha256(data).hexdigest(),'inspection':'Actually read current complete prerequisite section this round because its generated notebook changed. No separate formal whole-lesson verdict is asserted. Assigned student-count/scope claims still match unchanged original results.'})
record['figure_reuse']={'path':'course/figures/character_ids.svg','current_sha256':sha(ROOT/'course/figures/character_ids.svg'),'previous_sha256':prior['reuse_exact_byte_comparisons'][1]['current_sha256']};assert record['figure_reuse']['current_sha256']==record['figure_reuse']['previous_sha256']
cn=(ROOT/'README.md').read_text().splitlines();en=(ROOT/'README_en.md').read_text().splitlines()
record['prior_issue_reassessment']={'c05':{'prior_assertion':'影片目前提供逐幀切片的輔助函式','current_line17':cn[16],'assertion_remains_absent':'影片目前提供逐幀切片的輔助函式' not in '\n'.join(cn),'judgment':'Removal remains factually sufficient; no claim that a helper was added.'},'c28':{'old_round2_english_line9':(R2/'README_en.md.read-snapshot').read_text().splitlines()[8],'current_english_line9':en[8],'current_chinese_line9':cn[8],'judgment':'Current both-language statements expressly condition setup on notebooks with code and locate it in the first code cell. Exact264first-code bootstrap matches and exclusion of2prose-only notebooks support the corrected wording. Colab execution remains explicitly unverified.'}}
assert 'In notebooks with code, the first code cell prepares the repository and packages.' in en[8]
assert '有程式的Notebook會在第一個程式格準備專案與套件' in cn[8]
(OUT/'current-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('Current CPU',torch.__version__,'CUDA',torch.cuda.is_available())
print('Current notebook inventory',record['notebook_summary'])
print('Preserved genuine reports',[(p['sha256'],p['verdict']) for p in record['preserved_reports']])
print('Current counts',record['current_counts'])
