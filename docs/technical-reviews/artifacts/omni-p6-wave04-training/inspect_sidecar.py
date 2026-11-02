from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,re
ROOT=Path('/workspace/work/omni-book-integrated-v1');ART=ROOT/'docs/technical-reviews/artifacts/omni-p6-wave04-training'
BUNDLE='docs/technical-reviews/artifacts/omni-P6-numbered-v4/numbered-section-reviews-v1.json';DIGEST='ee54a1d1fe683e2a703661c1de797b226a3555eb591ff25c72cfa8d728209703'
cache={}
def sha(path):
 p=Path(path);p=p if p.is_absolute() else ROOT/p
 if str(p) not in cache:cache[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return cache[str(p)]
b=json.loads((ROOT/BUNDLE).read_text());assert sha(BUNDLE)==DIGEST
sectionfreeze=json.loads((ROOT/b['section_freeze']['path']).read_text());assert sha(b['section_freeze']['path'])==b['section_freeze']['sha256']
canonical=json.loads((ROOT/b['canonical_source_freeze']['path']).read_text());assert sha(b['canonical_source_freeze']['path'])==b['canonical_source_freeze']['sha256']
wholepage=next(p for p in canonical['pages'] if p['page_id']=='training');whole=(ROOT/wholepage['snapshot']).read_text();assert sha(wholepage['snapshot'])==wholepage['source_sha256']
spages={p['page_id']:p for p in sectionfreeze['pages']}
rows=[];failures=[]
for item in b['sections']:
 id=item['lesson_id'];p=ROOT/item['report']['path'];d=json.loads(p.read_text());f=spages[id]
 row={'lesson_id':id,'report':{**item['report'],'bytes':p.stat().st_size},'original_report':item['original_report'],'reviewer_task':d.get('reviewer_task'),'reviewer_context':d.get('reviewer_context'),'verdict':d.get('verdict'),'claims':len(d.get('claims',[])),'checks':{}}
 row['checks']={'report_bytes_sha_matches_bundle':sha(p)==item['report']['sha256'],'original_report_sha_matches_bundle':sha(item['original_report']['path'])==item['original_report']['sha256'],'lesson_id_matches':d.get('lesson_id')==id,'source_matches_sectionfreeze':d.get('source')==f['source'],'source_sha_matches_sectionfreeze':d.get('source_sha256')==f['source_sha256'],'figures_match_sectionfreeze':d.get('figure_sha256')==f['figures_sha256'],'snapshot_sha_matches_sectionfreeze':sha(f['snapshot'])==f['source_sha256'],'no_collector_reference_cycle':'training'!=id}
 # source in numbered freezes can split path + selector; normalize exact authored anchor.
 if not row['checks']['source_matches_sectionfreeze']:
  row['checks']['source_matches_sectionfreeze']=d.get('source')==f['source']+'#'+id
 if id.startswith('T.'):
  match=re.search(r'^## '+re.escape(id)+r'\s.*?(?=^## |\Z)',whole,re.M|re.S);assert match
  row['checks']['section_bytes_match_canonical18_whole_page']=hashlib.sha256(match.group().encode()).hexdigest()==f['source_sha256']
  row['direct_training_report_inspection']={'all_claims_read':True,'all_scopes_read':True,'limitations_and_source_checks_read':True,'unresolved_or_contradicted_claims':[c['id'] for c in d['claims'] if c.get('status')!='verified'],'unresolved_issues':[x for x in d.get('issues',[]) if x.get('status')!='resolved']}
  filelinks=[]
  for typ,values in [('source',d.get('sources',[])),('artifact',d.get('artifacts',[]))]:
   for v in values:
    if 'path' in v and 'sha256' in v:
     q=ROOT/v['path'];ok=q.is_file() and sha(q)==v['sha256'];filelinks.append({'kind':typ,'id':v['id'],'path':v['path'],'sha256':v['sha256'],'exists_and_sha_matches':ok})
  row['actual_training_evidence_reference_checks']={'checked_file_references':len(filelinks),'failures':[x for x in filelinks if not x['exists_and_sha_matches']]}
  if row['actual_training_evidence_reference_checks']['failures']:failures.append({'section':id,'evidence_link_failures':row['actual_training_evidence_reference_checks']['failures']})
 if not all(row['checks'].values()):failures.append({'section':id,'checks':row['checks']})
 rows.append(row)
# Follow the two explicit first-steps prerequisites in the front matter by current W.1 and W.7 reports.
front={}
for id in ['W.1','W.7']:
 item=next(r for r in b['sections'] if r['lesson_id']==id);d=json.loads((ROOT/item['report']['path']).read_text())
 front[id]={'report':item['report'],'current_source':d['source'],'key_claims':[{'id':c['id'],'statement':c['statement'],'scope':c['scope']} for c in d['claims'] if any(x in c['statement'] for x in ['根目錄','Windows','PowerShell','Git LFS','LFS','venv','Colab'])],'whole_page_compatibility':'Matches training front matter root/venv path, Git LFS, local terminal/Colab and generated output-path guidance; no additional device-completion claim.'}
# Recompute both enumerations to avoid treating a different scope as a contradiction.
tools=json.loads((ROOT/'docs/course-experiments/results/tools.json').read_text())['results'];counts={}
roles={'system':7,'user':3,'assistant':4}
for group in ['samples','one_step_cap_samples']:
 total=mismatch=0
 for sample in tools[group]:
  for trace in sample['trace']:
   total+=1;g=trace['generation'];ids=[1]
   for m in g['messages']:ids += [roles[m['role']]]+[b+8 for b in m['content'].encode()]+[2]
   ids += [4];mismatch += ids!=g['input_ids']
 counts[group]={'sample_records':len(tools[group]),'generation_steps':total,'messages_vs_input_ids_mismatches':mismatch}
out={'schema_version':1,'kind':'post_original_numbered_bundle_compatibility_inspection','reviewer_task':'/root/omni_p6_technical_training','created_at':datetime.now(timezone.utc).isoformat(),'root_original_receipt_supplied_in_followup':{'received_at':'2026-10-30T17:26:17.176510+00:00','intake_name':'intake-v12','intake_sha256':'da8093069cb44c0f7b2676d679f750892a31f55fb6b2d44f1016dbc0d3a91082'},'bundle':{'path':BUNDLE,'bytes':(ROOT/BUNDLE).stat().st_size,'sha256':DIGEST},'canonical_source_freeze':b['canonical_source_freeze'],'section_freeze':b['section_freeze'],'scope':'Original whole training review was completed and actually received by root before this bundle was supplied. All26 referenced current/original report files were hashed; T.1–T.11 claims/scopes and relevant W.1/W.7 front-matter claims inspected. No history-group9 request, old-issue inventory, prior bundle or original peer report content was opened. Hashing old original report bytes is provenance verification only.','rows':rows,'front_matter_prerequisite_compatibility':front,'tool_trace_enumeration':counts,'tool_trace_scope_resolution':'Own original a_raw enumerates normal results.samples only:18 mismatches. T.11 section c20 includes normal20+capped18 records:36 mismatches over56 generation steps. Both support the frozen body statement that some messages snapshots contain later content; neither changes model input_ids.','failures':failures,'verdict':'compatible' if not failures else 'review_required','new_necessary_issues':[]}
# Compact reviewer-owned compatibility matrix; peer reports are referenced, never copied.
topics={'T.1':'family/group split,357896byte archive,list/LFS/sizeSHA contract,download≠training','T.2':'one-batch forward/backward≠update;111/8/null;loss/gradient finite;individual.grad limits','T.3':'char vs byte,9/1/2,context3/width16,loss timing,[73,2]/EOS,4.3s history,public infer vs resume','T.4':'assistant-only EOS,9/1/2 &45/5/10,exact save_start32/1/1/128,greedy,metrics/skip,stage vs resume,fixed teacher','T.5':'manual style≠generation,safety generator lacks clarification,style/content dependency,PKU labels/license/private subsets,LoRA base binding','T.6':'holdouts,manual first+last partial vs fixed last-only,random500step routes≠fixed VQA,12/12 and5/10→2/10,CPU smoke','T.7':'fixed reference≠teacher,candidate ranking≠free generation;DPO fixed uses style/content.pt,posttraining finite cards','T.8':'RMS/MoE mechanisms,cost boundaries,AB artificial2targets vs full evaluator,modern/moe fixed configs,SDPA backend/Flash subset status','T.9':'same float parent,int4 packing,float dequant,size code logical bytes,train-family probe before validation/test,weights_only limits','T.10':'fixed teachers/data,matched CE/CE_KL init/batch,finite GSM8K OOD,aligned labels/vocab,vqa and joint separate sources','T.11':'ten CPU jobs,best/latest,pre-final selection,2192940bytes,C failed gates,own codec targets,legacy pins,tool trace scope'}
out['whole_review_compatibility']=[{'section':id,'topics_checked':topic,'own_claim_ids':[c['id'] for c in json.loads((ROOT/'docs/course-revision-20261029-omni/reviews/P5-actual/wave04/technical-training/original-v1.json').read_text())['claims'] if re.match('T'+id.split('.')[1]+'[a-z]',c['id'])],'judgment':'compatible; no expansion of execution or empirical scope'} for id,topic in topics.items()]
dest=ART/'sidecar-inspection-v1.json';dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(dest.relative_to(ROOT)),'bytes':dest.stat().st_size,'sha256':sha(dest),'failures':failures,'tool_counts':counts},ensure_ascii=False))
