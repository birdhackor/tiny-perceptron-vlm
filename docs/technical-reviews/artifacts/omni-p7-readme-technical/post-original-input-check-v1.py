"""Post-original reference/hash checks only; no fresh lesson or model execution."""
from pathlib import Path
import collections,datetime,hashlib,json,platform
ROOT=Path('/workspace/work/omni-book-integrated-v1');BASE=ROOT/'docs/technical-reviews/artifacts/omni-p7-readme-technical';WAVE=ROOT/'docs/course-revision-20261029-omni/reviews/P5-actual/wave05'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def load(p):return json.loads(p.read_text())
p=WAVE/'technical-readme/original-v1.json';assert sha(p)=='515f1ad20004f8bf9e2019abc2c4e3237819d8305ec6edaaac73548e90af58ba';own=load(p)
for artifact in own['artifacts']:assert sha(ROOT/artifact['path'])==artifact['sha256']
for source in own['sources']:
 if source['kind']=='repository_code':assert sha(ROOT/source['path'])==source['sha256']
snapshots=load(BASE/'source-snapshots.json')
for x in snapshots:assert sha(ROOT/x['snapshot'])==x['snapshot_sha256'] and sha(ROOT/x['path'])==x['sha256']
assert sha(ROOT/'README.md')==own['source_sha256']
for path,h in own['figure_sha256'].items():assert sha(ROOT/path)==h
names=['original-v1.json','basis-v1.json','current-v1.json','checkpoint-closure-v1.json','reading-proof-v1.json','ui-receipt-v1.json','w1-context-read-v1.json','supplement-readme-v1.json'];inputs={n:ref(WAVE/'reader-readme'/n) for n in names}
proof=load(WAVE/'reader-readme/reading-proof-v1.json')['pages'][0]['reader'];session=proof['sessions'][0];notes=[];concepts=[]
for r in session['notes']:
 p=ROOT/r['path'];assert sha(p)==r['sha256'];a=load(p);assert a['unit_index']==r['unit_index'];notes.append(ref(p))
 for i,c in enumerate(a['concept_checks']):concepts.append({'unit_index':a['unit_index'],'concept_index':i,'concept':c['concept'],'original_verdict':c['verdict'],'note':ref(p),'json_pointer':f'/concept_checks/{i}'})
closure=load(WAVE/'reader-readme/checkpoint-closure-v1.json')['pages'][0];qualifying={(x['unit_index'],x['concept_index']):x for x in concepts if x['original_verdict']!='已足夠'}
resolved={(x['unit_index'],x['concept_index']):x for x in closure['checkpoint_resolutions']}
assert len(notes)==24 and len(concepts)==52 and set(qualifying)==set(resolved) and len(resolved)==49
for k,x in resolved.items():assert x['original_verdict']==qualifying[k]['original_verdict'] and x['concept']==qualifying[k]['concept'] and x['note_sha256']==qualifying[k]['note']['sha256']
statuses=dict(collections.Counter(x['final_status'] for x in resolved.values()));assert statuses=={'deferred':5,'nonblocking':35,'resolved':9};assert closure['status_counts']['open']==0
for n in ['report','original_report']:
 r=proof[n];assert sha(ROOT/r['path'])==r['sha256']
for n in ['basis','checkpoint_closure']:
 r=proof[n]['report'];assert sha(ROOT/r['path'])==r['sha256']
context=load(WAVE/'reader-readme/w1-context-read-v1.json')['source'];b=(ROOT/'course/first-steps.md').read_bytes();assert hashlib.sha256(b).hexdigest()==context['full_context_source_sha256'];excerpt=b[context['byte_range_start_inclusive']:context['byte_range_end_exclusive']];assert len(excerpt)==7634 and hashlib.sha256(excerpt).hexdigest()==context['excerpt_sha256']
images=[]
for n,h in [('mobile-toc-jump.png','8cd880d899d4fd0a18478078dd0c70af2f7d0aaa89479d3d5d4c05f94bf501b1'),('mobile-toc-dismissed.png','230e5b00dd1c230853a4e5a7873e231f276eb1d4c4fa6bad7d1b89d9f4efd571')]:
 p=Path('/workspace/work/omni-p7-reader-readme-ui-v1')/n;assert sha(p)==h;images.append({'external_path':str(p),'sha256':h})
intake=WAVE/'technical-collection-intake-v1.json';assert sha(intake)=='0f0cedfa109a7797912138f1391daab64a9123b575fff89aa23f6e57cd9d9fdc'
join=WAVE/'reader-current-proof-join-v2.json';assert sha(join)=='e19341ac70c9a429e798ffc4b38d97287594b27ded3b257fa82bded0bca3336f'
result={'reviewer_task':'/root/omni_p7_technical_readme','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'own_original':ref(WAVE/'technical-readme/original-v1.json'),'root_received_at':'2026-10-30T21:34:30.400516+00:00','root_intake':ref(intake),'root_intake_scope':'Only the intake path/SHA was checked; no other reviewer or author decision body read. It is provenance, not a stable technical source.','peer_inputs':inputs,'reader_current_proof_join':ref(join),'reader_task':proof['reviewer_task'],'raw_notes':notes,'raw_concepts':concepts,'concept_count':len(concepts),'closure_qualifying_count':len(resolved),'closure_status_counts':{**statuses,'open':0},'W1_context':{'full_source_sha256':context['full_context_source_sha256'],'byte_range':[504,8138],'bytes':len(excerpt),'excerpt_sha256':context['excerpt_sha256'],'scope':'Bounded later W.1 only; no full first-steps first-reading claim.'},'secondary_image_refs':images,'original_bound_artifacts_checked':len(own['artifacts']),'original_repository_sources_checked':sum(x['kind']=='repository_code' for x in own['sources']),'unchanged_source_snapshots_checked':len(snapshots),'command':'python docs/technical-reviews/artifacts/omni-p7-readme-technical/post-original-input-check-v1.py','environment':{'python':platform.python_version(),'platform':platform.platform(),'device':'CPU; stdlib JSON/hash only'},'status':'pass','method_limit':'Mechanical integrity/count checking only. Manual substantive comparisons are recorded separately; no model, kernel, training or publication gate was invoked.'}
(BASE/'post-original-input-check-v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['created_at','concept_count','closure_qualifying_count','closure_status_counts','original_bound_artifacts_checked','original_repository_sources_checked','unchanged_source_snapshots_checked','status']},ensure_ascii=False,indent=2))
