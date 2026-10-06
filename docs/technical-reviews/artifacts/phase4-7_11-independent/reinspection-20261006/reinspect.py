import hashlib,importlib.util,json,pathlib,platform,re,sys
ROOT=pathlib.Path('/workspace/tiny-perceptron-vlm');A=ROOT/'docs/technical-reviews/artifacts/phase4-7_11-independent';N=A/'reinspection-20261006'
PRIOR=ROOT/'docs/technical-reviews/history/phase4-7_11-own-prior-before-reinspection-2a5a262420055f5f6b0bba09fab7637511ea0ebe18f32b10afb73d675a926a33.json'
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(p):return sha(pathlib.Path(p).read_bytes())
report=json.loads(PRIOR.read_text());assert digest(PRIOR)=='2a5a262420055f5f6b0bba09fab7637511ea0ebe18f32b10afb73d675a926a33'
checks=[]
for item in report['artifacts']:
 p=ROOT/item['path'];checks.append({'id':item['id'],'path':item['path'],'expected_sha256':item['sha256'],'actual_sha256':digest(p),'equal':digest(p)==item['sha256']})
assert all(c['equal'] for c in checks)
code=[]
for p in ['tiny_perceptron/data.py','tiny_perceptron/model.py','tiny_perceptron/attention.py','tiny_perceptron/modern.py','tiny_perceptron/training.py','scripts/train.py','scripts/course_experiments/common.py','scripts/course_experiments/text.py','scripts/prepare_data.py']:
 current=(ROOT/p).read_bytes();frozen=(A/'inputs'/p).read_bytes();assert current==frozen;code.append({'path':p,'current_sha256':sha(current),'own_frozen_sha256':sha(frozen),'unchanged':True})
spec=importlib.util.spec_from_file_location('facts',ROOT/'docs/review-tools/section_facts.py');facts=importlib.util.module_from_spec(spec);spec.loader.exec_module(facts)
current,_,first=facts.original_section(ROOT/'course/chapters/07.md','7.11');assert sha(current)=='a5f2ab4d56da0a8cdf6ba988f0cfca92d0ea902b2f1a4760ac4bf67724c60872';assert current==(N/'7.11.md').read_bytes()
original=(A/'original/section.md').read_bytes();oldf=facts.fences(original,first);newf=facts.fences(current,first);assert [f['raw'] for f in oldf]==[f['raw'] for f in newf]
images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',current.decode());assert images==[]
oldjson=json.loads((A/'inputs/original-sft.json').read_text());newjson=json.loads((ROOT/'docs/course-experiments/results/sft.json').read_text())
def pointer(obj,p):
 for part in p.split('/'):obj=obj[part]
 return obj
pointers=['revision','seed','step_scale','device','torch_version','results/data','results/training','results/before','results/after','results/pretrain_then_sft']
measurements=[]
for p in pointers:
 a=pointer(oldjson,p);b=pointer(newjson,p);assert a==b;measurements.append({'json_pointer':'/'+p,'equal':True,'canonical_selected_value_sha256':sha(json.dumps(b,ensure_ascii=False,sort_keys=True).encode())})
counts={}
for name,p in [('direct_sft','results/after/test'),('text_before_sft','results/pretrain_then_sft/before_sft/test'),('text_after_sft','results/pretrain_then_sft/after_sft/test')]:
 ev=pointer(newjson,p);samples=ev['samples'];matches=sum(s['generated']==s['expected'] for s in samples);eos=sum(2 in s['generated_ids'] for s in samples)
 assert len(samples)==ev['records']==ev['examples']==10 and matches==ev['matches'];assert matches/10==ev['exact_match']
 counts[name]={'measurement_pointer':'/'+p,'matches':matches,'records':10,'eos_count':eos,'target_tokens':ev['effective_tokens']}
assert [counts[name]['matches'] for name in ['direct_sft','text_before_sft','text_after_sft']]==[5,1,7]
ctx12=(N/'7.12.md').read_text();ctx17=(N/'7.17.md').read_text();target=current.decode()
assert '成績見[7.17](#7.17)' in target and '下一節[7.12](#7.12)先用四格材料說明題目家族的切分規則' in target
assert '最後十題從對話微調前1題答對' in ctx17 and '微調後7題答對' in ctx17 and '直接SFT的5／10' in ctx17
assert '沒有訓練模型或產生答題成績' in ctx12 and all(s in ctx12 for s in ['紅、圓','藍、圓','藍、方','紅、方'])
page=json.loads((N/'page-render-receipt.json').read_text());assert page['dom']['target_images']==0
assert [l['href'].split('/')[-1] for l in page['dom']['links']]==['7.17.html','7.12.html']
rec={'schema_version':1,'kind':'same_owner_actual_reinspection','reviewer_task':report['reviewer_task'],'reviewed_on':'2026-10-06','prior_report_path':PRIOR.relative_to(ROOT).as_posix(),'prior_report_sha256':digest(PRIOR),'prior_raw_section_sha256':sha(original),'current_source':'course/chapters/07.md#7.11','current_raw_section_sha256':sha(current),'current_section_snapshot':(N/'7.11.md').relative_to(ROOT).as_posix(),'changed_claims':['C7','C8'],'actual_change':'Only supplemental guide: empirical scores now point to7.17;7.12 is explicitly the four-cell family-splitting demonstration. No technical recipe/code or numerical measurement changed.','whole_current_section_personally_read':True,'necessary_context_personally_read':['7.1 complete','7.12 complete: four cells and family split/no score','7.17 complete: target score supplement and shared-base diagram'], 'original_authority_support_reinspected':['InstructGPTv1 §3.1 Step1 and same-architecture introduction','TRLv0.23.1 fixed-commit SFT CE/shift/assistant-only settings'], 'unchanged_claim_support':'C1-C8 original primary sources, helper/code, original measurement and own executed evidence verified byte-for-byte; own first-run support retained. New C7 guidance authenticated by current target contexts plus current measurements and actual rendered page.','python_fence_sha256':[sha(f['raw']) for f in newf],'python_fences_unchanged':True,'target_figure_sha256':{},'dependent_context_figure':{'path':'course/figures/rewrite-07-17-prepost-routes.svg','sha256':digest(N/'rewrite-07-17-prepost-routes.svg'),'render_path':(N/'context-7.17.png').relative_to(ROOT).as_posix(),'render_sha256':digest(N/'context-7.17.png'),'personally_viewed':True,'scope':'Necessary linked7.17 context; not a figure in7.11. General shared base to separately updated/evaluated tasks, consistent with original InstructGPT architecture/SFT support; not a local score chart.'},'prior_artifact_integrity':checks,'live_code_vs_own_frozen':code,'measurement_raw_file_sha256':digest(ROOT/'docs/course-experiments/results/sft.json'),'measurement_frozen_raw_sha256':digest(A/'inputs/original-sft.json'),'measurement_pointers':measurements,'changed_crossreference_measurements_reaggregated':counts,'rendered_page':{'url':page['dom']['url'],'viewport':[1280,1000],'details_opened':True,'screenshot_path':(N/'page-current-paragraph.png').relative_to(ROOT).as_posix(),'screenshot_sha256':digest(N/'page-current-paragraph.png'),'personally_viewed':True,'dom_links':page['dom']['links'],'target_images':0,'failed_initial_cli_render':'Initial30s Chromium CLI screenshot timed out without an image; exact log retained. Bounded20s CDP render with separate profile succeeded; screenshot personally viewed.'},'command':str(ROOT/'.venv/bin/python')+' '+str(N/'reinspect.py'),'environment':{'python':platform.python_version(),'python_executable':sys.executable,'device':'cpu; pure JSON/hash reaggregation, no model execution','runtime':'Node24.19.0 Chromium CDP for page; Inkscape for context SVG'},'original_cpu_evidence_reused_not_rerun':True,'new_model_cpu_execution':False,'gpu_or_training_execution':False,'new_external_primary_source_fetch':False,'verdict':'pass','unresolved_questions':[]}
(N/'reinspection-receipt.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'verdict':rec['verdict'],'current_source_sha256':sha(current),'receipt_path':(N/'reinspection-receipt.json').relative_to(ROOT).as_posix(),'receipt_sha256':digest(N/'reinspection-receipt.json'),'scores':counts,'verified_prior_artifacts':len(checks),'verified_live_code_files':len(code),'unresolved_questions':[]},ensure_ascii=False,indent=2))
