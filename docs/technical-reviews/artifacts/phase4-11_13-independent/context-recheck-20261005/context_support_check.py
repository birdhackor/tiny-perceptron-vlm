from pathlib import Path
from datetime import UTC,datetime
from collections import Counter,defaultdict
import ast,hashlib,json,sys
q=Path(__file__).resolve().parent;r=q.parent;repo=r.parents[3]
prior=json.loads((q/'prior-pass-opaque.json').read_bytes())
assert prior['reviewer_task']=='/root/phase4_factual_coordinator/factual_11_13' and prior['verdict']=='pass'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
raw_path=repo/'docs/course-experiments/results/ocr.json';raw=json.loads(raw_path.read_bytes())
proof=[{'id':a['id'],'path':a['path'],'expected':a['sha256'],'observed':sha(repo/a['path']),'unchanged':sha(repo/a['path'])==a['sha256']} for a in prior['artifacts']]
assert all(a['unchanged'] for a in proof)
current_code={}
for path in ('scripts/course_experiments/modalities.py','tiny_perceptron/data.py','tiny_perceptron/multimodal.py'):
 observed=sha(repo/path);frozen=sha(r/'inputs'/path);historical=raw['code_sha256'][path]
 assert observed==frozen==historical
 current_code[path]={'current_sha256':observed,'original_frozen_sha256':frozen,'historical_result_sha256':historical,'all_equal':True}
assert sha(raw_path)==sha(r/'inputs/docs/course-experiments/results/ocr.json')
assert (q/'11.13.md').read_bytes()==(r/'inputs/section.md').read_bytes()
selected=[];families={};results={}
for split in ('train','validation','test'):
 recordset=raw['results']['data']['splits'][split]
 rows=recordset['records']
 # Read raw data identities and transformations only, never descriptive result/scope strings.
 measures=[{k:row[k] for k in ('answer','digits','family','offset','modality','question')} for row in rows]
 assert len(measures)==recordset['count']
 assert all(row['answer']==row['digits']==row['family'] and row['modality']=='vision' and row['question']=='read digits' for row in measures)
 counts=Counter(row['family'] for row in measures);offsets=defaultdict(set)
 for row in measures:offsets[row['family']].add(row['offset'])
 assert set(counts.values())=={3} and all(v=={-1,0,1} for v in offsets.values())
 families[split]=set(counts)
 results[split]={'count':recordset['count'],'family_count':len(counts),'digit_characters':sorted({c for row in measures for c in row['answer']}),'families':sorted(counts,key=int),'position_variants_per_family':3,'offset_values':[-1,0,1]}
 selected += [f'/results/data/splits/{split}/count']+[f'/results/data/splits/{split}/records/*/{k}' for k in ('answer','digits','family','offset','modality','question')]
assert results['train']['count']==240 and results['train']['family_count']==80
assert results['validation']['count']==results['test']['count']==30
assert results['validation']['family_count']==results['test']['family_count']==10
assert results['train']['digit_characters']==list('0123456789')
assert set(results['test']['digit_characters']).issubset(results['train']['digit_characters'])
assert all(not families[a]&families[b] for a,b in (('train','validation'),('train','test'),('validation','test')))
assert set.union(*families.values())==set(map(str,range(100)))
method=repo/'scripts/course_experiments/modalities.py';tree=ast.parse(method.read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_ocr')
metadata={'checked_at':datetime.now(UTC).isoformat(),'environment':{'python':sys.version,'executable':sys.executable,'device':'raw-data arithmetic only; no torch import/model operation'},'raw_json_path':'docs/course-experiments/results/ocr.json','raw_json_sha256':sha(raw_path),'json_pointers':selected+['/code_sha256/scripts~1course_experiments~1modalities.py','/code_sha256/tiny_perceptron~1data.py','/code_sha256/tiny_perceptron~1multimodal.py'],'code_ast_locator':{'path':'scripts/course_experiments/modalities.py','function':'run_ocr','function_span':[node.lineno,node.end_lineno],'actually_read_lines':[833,856],'sha256':sha(method)},'splits':results,'whole_string_families_disjoint':True,'test_characters_seen_in_training':True,'code_proof_current_equal':current_code,'original_artifact_hash_verification':proof,'original_11_13_proof_unchanged':True,'scope':'Only data-family/character-set bookkeeping supports the changed prerequisite scope. The archived metric/exact/EOS/DP/probability/render evidence remains intact and is honestly reused. No prior model evaluation, CPU fence rerun, training, GPU, weight load/save, or network request.'}
(q/'context-support-results.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in metadata.items() if k not in ('original_artifact_hash_verification',)},ensure_ascii=False,indent=2))
