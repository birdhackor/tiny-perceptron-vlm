"""Independent bounded T.11 evidence inspection; no training, download, or retained weights."""
from pathlib import Path
import sys,json,hashlib,tempfile,subprocess,os,math
from fractions import Fraction
from collections import Counter
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
sys.path.insert(0,str(ROOT))
import torch
from scripts import evaluate as evaluator
from scripts.course_experiments import applications as apps
from scripts.course_experiments.common import split_records,records_sha256
from tiny_perceptron.model import TinyLM,ModelConfig
from tiny_perceptron.training import save_checkpoint
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.capstone import calculator_runtime,parse_action

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
ENV={'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),'device':'cpu','cuda_build':str(torch.version.cuda),'checkout_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
(OUT/'environment.json').write_text(json.dumps(ENV,indent=2)+'\n')
shapes=json.loads((OUT/'raw-shapes.json').read_text()); inspections=[]
def original(rel):
 meta=shapes[rel];path=ROOT/meta['copy']; raw=path.read_bytes(); assert hashlib.sha256(raw).hexdigest()==meta['sha256']; assert (ROOT/rel).read_bytes()==raw
 return json.loads(raw)
def inspect(d,rel,ptr):
 v=d
 for part in ptr.strip('/').split('/'):v=v[int(part)] if isinstance(v,list) else v[part]
 inspections.append({'file':rel,'pointer':ptr,'value':v});return v

def rate_match(metric,n,d):
 assert metric['numerator']==n and metric['denominator']==d
 assert (metric['rate'] is None and d==0) or math.isclose(metric['rate'],n/d,rel_tol=0,abs_tol=1e-12)

example={'old_bytes':80000,'new_bytes':60000,'old_correct':4,'new_correct':3,'questions':5}
assert example['new_bytes']<example['old_bytes'] and not(example['new_correct']>=example['old_correct'])
example.update(size_ratio=str(Fraction(60000,80000)),old_rate=str(Fraction(4,5)),new_rate=str(Fraction(3,5)),both_conditions_satisfied=False)
a=[True,True,True,True,False]; b=[True,True,True,False,True]
paired={'old_correct':sum(a),'new_correct':sum(b),'newly_wrong':sum(x and not y for x,y in zip(a,b)),'newly_right':sum(not x and y for x,y in zip(a,b))}
assert paired=={'old_correct':4,'new_correct':4,'newly_wrong':1,'newly_right':1}
assert Fraction(4,4)!=Fraction(4,5)
hashes=[hashlib.sha256(s).hexdigest() for s in [b'answer=wrong\n',b'answer=wrong\n',b'answer=right\n']];assert hashes[0]==hashes[1]!=hashes[2]

split_checks={};summary={}
for name,generated in [('rag',apps._rag_splits(42)),('tools',split_records(apps._tool_records(),42)),('reasoning',split_records(apps._reasoning_records(),42))]:
 rel=f'docs/course-experiments/results/{name}.json';d=original(rel);r=d['results']
 for field in ['revision','seed','device','gpu','python_version','torch_version']:
  inspect(d,rel,'/'+field)
 for split,rows in generated.items():
  stored=inspect(d,rel,f'/results/split/{split}')
  assert stored['records']==len(rows) and stored['families']==len({row['family'] for row in rows}) and stored['sha256']==records_sha256(rows)
 families=[{row['family'] for row in generated[s]} for s in ['train','validation','test']]
 assert not(families[0]&families[1] or families[0]&families[2] or families[1]&families[2])
 split_checks[name]={s:len(rows) for s,rows in generated.items()}
 if name=='rag':
  samples=r['samples']; modes=sorted({x['mode'] for x in samples})
  for mode in modes:
   rows=[x for x in samples if x['mode']==mode]
   for field in ['exact_match','fact_answer_correct','citation_valid','supported_by_cited_source','unknown']:
    metric=inspect(d,rel,f'/results/metrics/{mode}/{field}');rate_match(metric,sum(x[field] for x in rows),len(rows))
   for row in rows:
    method=apps._grounding(row['generation']['samples'][0],row['documents'],row['expected'])
    assert all(row[k]==v for k,v in method.items())
  for ptr in ['/results/samples/0/family','/results/samples/0/documents','/results/samples/0/expected','/results/samples/0/generation/messages','/results/samples/0/generation/samples/0/generated','/results/samples/0/generation/samples/0/eos']:
   inspect(d,rel,ptr)
  summary[name]={'total_saved_samples':len(samples),'modes':len(modes),'per_mode_denominator':len(samples)//len(modes)}
 elif name=='tools':
  samples=r['samples']; assert len(samples)==len(generated['test']);rate_match(inspect(d,rel,'/results/accuracy'),sum(x['correct'] for x in samples),len(samples))
  executed=[(i,j,e) for i,x in enumerate(samples) for j,e in enumerate(x['trace']) if e.get('executed')]
  assert sum(x['actual_tool_calls'] for x in samples)==r['actual_tool_calls']==len(executed)
  for i,j,event in executed:
   assert event['tool_result']==apps.call_tool(event['parsed'])
   if j+1<len(samples[i]['trace']):
    assert any(m['content']==f"TOOL_RESULT:{event['tool_result']}" for m in samples[i]['trace'][j+1]['generation']['messages'])
  i,j,event=executed[0]
  for ptr in [f'/results/samples/{i}/question',f'/results/samples/{i}/expected',f'/results/samples/{i}/answer',f'/results/samples/{i}/trace/{j}/generation/samples/0/generated',f'/results/samples/{i}/trace/{j}/tool_result',f'/results/samples/{i}/trace/{j+1}/generation/samples/0/generated']:
   inspect(d,rel,ptr)
  summary[name]={'saved_episodes':len(samples),'actual_executions':len(executed),'inspected_example_index':i}
 elif name=='reasoning':
  for mode in ['direct','steps']:
   budgets=r['comparison'][mode]['budgets']
   for index,budget in enumerate(budgets):
    samples=budget['samples']; candidates=[c for x in samples for c in x['candidates']]
    assert len(samples)==len(generated['test']) and len(candidates)==len(samples)*budget['candidate_count']
    for row in samples:
     for candidate in row['candidates']:
      verified=apps._verify_reasoning(candidate,row,mode);assert all(candidate[k]==v for k,v in verified.items())
    for field,key,rows in [('candidate_final_accuracy','final_correct',candidates),('fully_verified_candidates','fully_verified',candidates),('majority_accuracy','majority_correct',samples),('oracle_coverage','oracle_coverage',samples)]:
     rate_match(inspect(d,rel,f'/results/comparison/{mode}/budgets/{index}/{field}'),sum(x[key] for x in rows),len(rows))
   for leaf in ['question','truth','candidates/0/generated','candidates/0/final_answer']:
    inspect(d,rel,f'/results/comparison/{mode}/budgets/0/samples/0/{leaf}')
  summary[name]={'questions_per_budget':len(generated['test']),'budgets':[1,2,4,8],'modes':['direct','steps']}

# Quantization evidence: stored provenance and comparison scope, without reading commentary or weights.
rel='docs/course-experiments/results/quantization.json';q=original(rel)
for ptr in ['/seed','/device','/revision','/results/data/source','/results/data/sha256','/results/data/split_unit','/results/data/counts','/results/data/family_intersections','/results/teacher_provenance/sha256','/results/training/steps','/results/training/effective_supervised_tokens','/results/same_fp32_source','/results/reloaded_packed_checkpoints_before_evaluation']:
 inspect(q,rel,ptr)
for name in ['fp32','packed4','packed8']:
 v=q['results']['runs'][name]
 inspections.append({'file':rel,'pointer':f'/results/runs/{name}','keys_types':{k:type(val).__name__ for k,val in v.items()}})
# Parent lineage: full hashes, declared chronology and original validation/test row counts.
s=original('docs/course-experiments/results/capstone_sft.json');j=original('docs/course-experiments/results/capstone_joint.json')
assert s['results']['inference_export']['sha256']==j['results']['parent_checkpoint_sha256']
for rel,d in [('docs/course-experiments/results/capstone_sft.json',s),('docs/course-experiments/results/capstone_joint.json',j)]:
 for ptr in ['/results/inference_export/sha256','/results/parent_checkpoint_sha256','/results/test_evaluated','/results/data_manifest/version','/results/data_manifest/counts','/results/data_manifest/sha256']:
  inspect(d,rel,ptr)
sel=original('docs/course-experiments/capstone-selection.json')
for ptr in ['/selected_stage','/selected_at_utc','/selected_before_test_generation','/candidates/joint','/candidates/dpo']:
 inspect(sel,'docs/course-experiments/capstone-selection.json',ptr)
for rel in ['docs/course-experiments/capstone-evidence/sft/validation.json','docs/course-experiments/capstone-evidence/deployment/test-joint.json']:
 d=original(rel); rows=d['records'];assert d['count']==len(rows);assert d['end_to_end_correct']==sum(x['end_to_end_correct'] for x in rows)
 for k in ['count','end_to_end_correct']:inspect(d,rel,'/'+k)
 for row in rows:
  if row['runtime'] is not None and row['runtime']['status']=='ok':
   assert row['runtime']==calculator_runtime(row['parsed_action'])
 summary[Path(rel).stem]={'records':len(rows),'tool_records':sum(x['runtime'] is not None for x in rows)}

# Original API contract variations, synthetic fixture only.
tok=ByteTokenizer(); answer=tok.encode('A')
assert evaluator.answer_sample(tok,answer+[tok.eos_id],'A')['completed_exact_match']
assert evaluator.answer_sample(tok,answer,'A')['exact_match'] and not evaluator.answer_sample(tok,answer,'A')['completed_exact_match']
assert not evaluator.answer_sample(tok,[tok.pad_id]+answer+[tok.eos_id],'A')['exact_match']
assert calculator_runtime(parse_action({'raw':'TOOL:calculator:1+2','eos':True}))=={'status':'ok','result':'3'}
assert calculator_runtime(parse_action({'raw':'TOOL:calculator:1+2','eos':True}),available=False)['status']=='error'
cli=[]
offline={'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
with tempfile.TemporaryDirectory(prefix='phase4-t11-fixture-') as folder:
 temp=Path(folder);torch.manual_seed(42);model=TinyLM(ModelConfig(width=8,heads=1,layers=1,max_length=128));save_checkpoint(temp/'fixture.pt',model)
 data=[{'messages':[{'role':'user','content':'Q'},{'role':'assistant','content':'A'}]},{'messages':[{'role':'user','content':'Q'*200},{'role':'assistant','content':'A'}]}]
 (temp/'fixture.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in data))
 for label,args in [('quantize',['scripts/quantize.py',str(temp/'fixture.pt'),'--bits','4','--output',str(temp/'fixture-int4.pt')]),('evaluate',['scripts/evaluate.py',str(temp/'fixture.pt'),'--data',str(temp/'fixture.jsonl'),'--mode','sft','--tokens','2','--limit','all','--split-label','test','--device','cpu','--output',str(temp/'evaluation.json')]),('infer',['scripts/infer.py',str(temp/'fixture-int4.pt'),'--chat','--prompt','Q','--tokens','2','--device','cpu','--json'])]:
  command=[str(ROOT/'.venv/bin/python'),*args];result=subprocess.run(command,cwd=ROOT,env={**os.environ,**offline},capture_output=True,text=True,timeout=30)
  cli.append({'fixture_label':'untrained contract fixture; not model capability evidence','label':label,'command_argv':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr});assert result.returncode==0
 report=json.loads((temp/'evaluation.json').read_text());assert report['declared_split']=='test' and report['records_read']==2 and report['records_selected']==2 and report['generation_evaluated_records']==1 and report['generation_skipped_records']==1 and report['metric_denominators']['completed_exact_match']==1
 fixture={'records_read':2,'generated':1,'skipped':1,'declared_split':'test','cli_exit_codes':[x['exit_code'] for x in cli],'temporary_weights_deleted_at_context_exit':True}
assert not temp.exists()
(OUT/'cli-commands-and-results.json').write_text(json.dumps(cli,ensure_ascii=False,indent=2)+'\n')
(OUT/'named-original-leaves.json').write_text(json.dumps(inspections,ensure_ascii=False,indent=2)+'\n')
proof={'arithmetic_example':example,'paired_example':paired,'split_checks':split_checks,'original_record_contracts':summary,'parent_lineage_matches':True,'selection_evidence_scope':'Original selection record declares selection before test. No claim of independently proving absence of all earlier test access. T.11 recommends the correct procedure rather than asserting a timestamp.','fixture_contract':fixture,'figure_scope':'T.11 has no images; necessary context figures 19.7 and 20.1 rendered and personally viewed separately.','training_performed':False,'new_capability_score_generated':False}
(OUT/'proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print(json.dumps(proof,ensure_ascii=False,indent=2))
