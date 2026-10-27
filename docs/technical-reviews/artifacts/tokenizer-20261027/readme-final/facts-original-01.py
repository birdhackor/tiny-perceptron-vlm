import json,hashlib,re
from pathlib import Path
from datetime import datetime,UTC
from scripts.reading_time import load_estimates
r=Path.cwd(); out=r/'docs/technical-reviews/artifacts/tokenizer-20261027/readme-final'
def load(p): return json.loads((r/p).read_bytes())
def digest(p): return hashlib.sha256((r/p).read_bytes()).hexdigest()
f=load('docs/course-review-20261027-tokenizer/freeze-03.json'); idx=load('course/lesson-index.json'); estimates=load_estimates(r/'course/reading-times.json',f,require_complete=True)
assert len(idx)==313 and len(f['pages'])==351 and len(estimates)==351
assert sum(1 for p in f['pages'] if p['kind']=='course_guide')==7
sections=sum(len(re.findall(r'^## (?:\d+|[A-Z])\.\d+(?=\s)',(r/p).read_text(),re.M)) for p in ['course/README.md','course/first-steps.md','course/training.md','course/glossary.md'])
assert sections==26
s=load('docs/selftrained/v2-manifest.json'); v=load('docs/selftrained/results/v2-final-public-results.json'); p=load('docs/lesson-products/results/final-report-v1.json'); pub=load('docs/course-experiments/public-models.json'); cap=load('docs/course-experiments/capstone-public.json')
weights=lambda fs:[x for x in fs if x['path'].endswith(('.pt','.safetensors'))]
models={k:{'parameters':a['capacity']['parameters'],'count':a['final_test']['count'],'selected_source':a['selected_source'],'tool_roundtrip':a['final_test']['per_task_final_reply']['tool_call']['tool_roundtrip'],'required_tool_roundtrip':a['final_test']['thresholds']['tool_roundtrip']} for k,a in v['architectures'].items()}
assert models['moe']['parameters']==5447107 and models['dense']['parameters']==2288067
assert all(x['count']==3734 and x['tool_roundtrip']['rate']<x['required_tool_roundtrip'] for x in models.values())
counts={'notebooks':len(idx),'additional_numbered_sections':sections,'total_numbered_sections':len(idx)+sections,'canonical_pages':len(f['pages']),'estimates':len(estimates),'v2_package_bytes':s['package']['bytes'],'v2_package_files':len(s['records'])+len(s['assets']),'public_experiments':len(pub['models']),'public_weight_files':sum(len(weights(m['files'])) for m in pub['models']),'capstone_weight_files':sum(len(weights(m['files'])) for m in cap['models'])}
assert counts['public_experiments']==30 and counts['public_weight_files']==120 and counts['capstone_weight_files']==11
source_list=['course/lesson-index.json','course/reading-times.json','scripts/reading_time.py','docs/selftrained/v2-manifest.json','docs/selftrained/results/v2-final-public-results.json','docs/lesson-products/results/final-report-v1.json','docs/course-experiments/public-models.json','docs/course-experiments/capstone-public.json','outputs/tokenizer-20261027/build-info.json']
receipt={'created_at':datetime.now(UTC).isoformat(),'counts':counts,'readme_estimate':estimates['readme'],'models':models,'data_scope':v['data_scope'],'looped_at_5':{k:p['b']['5'][k] for k in ['boxes','sum']},'judge':p['c'],'wording_shift':p['shifts'],'source_hashes':{x:digest(x) for x in source_list},'scope':'New counting and original saved empirical records checked; no cloud model training or heldout evaluation repeated. 351 estimates checked for current source/figure binding and completeness, not 351-page technical re-review or human timing.'}
(out/'facts-original-01.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'counts':counts,'readme_estimate':estimates['readme'],'models':models,'wording_shift_keys':list(p['shifts'])},ensure_ascii=False))
