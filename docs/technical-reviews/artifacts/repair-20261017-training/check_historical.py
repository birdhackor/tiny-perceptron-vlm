"""Read original reports/configuration and published raw scores; never train."""
import collections
import concurrent.futures
import gzip
import hashlib
import json
import platform
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).parent
WORK=Path('/workspace/work/textbook-repair-20261017/training-technical/historical')
WORK.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(path): return json.loads((ROOT/path).read_text())
def compact(value):
 if isinstance(value,dict):
  return {k:compact(v) for k,v in value.items() if k not in ('samples','generated_samples','before','after','teacher_validation','teacher_test','teacher_style','teacher_storage','teacher_cache','ultrachat_pilot','ultrafeedback_pilot','before_sft','after_sft','loss_trace','history','runtime','validation_routing','router_gradients_before','router_gradients_after','heldout','arithmetic','preference','storage','timing','held_out_wording')}
 if isinstance(value,list): return [compact(x) for x in value]
 return value
names=('simple_models','sft','style','safety','lora','encoders','projector','vqa','joint','dpo','posttraining','modern','moe','efficiency','precision','flash_probe','quantization','distillation','multimodal_distillation','ocr','rag','tools','reasoning')
receipt={'environment':{'python':platform.python_version(),'device':'CPU metadata only'},'scope':'Read historical originals; no rerun or inference of historical quality. GPU fields below describe original saved runs.','reports':{}}
for name in names:
 path=Path(f'docs/course-experiments/results/{name}.json'); d=load(path)
 r=d.get('results',d)
 chosen={k:compact(r[k]) for k in ('experiment','checkpoint','scope','sections','teacher_variant','reference_unchanged','reference_sha256','same_fp32_source','reloaded_packed_checkpoints_before_evaluation','alignment_rule','teacher_signal','matched_students','status','schedule_completed','verification_passed','supported_routes') if k in r}
 if name=='simple_models':chosen['run_names']=list(r['runs'])
 if name=='quantization':chosen['training_steps']=r['training']['steps']
 if name=='safety':chosen['pku']={k:r['pku_pilot'][k] for k in ('source_records','safe_selected_records','license','private_only','scope')}
 if name=='flash_probe':chosen['routes']={k:v['status'] for k,v in r['routes'].items()}
 receipt['reports'][name]={'path':str(path),'sha256':sha(ROOT/path),'environment':{k:d.get(k) for k in ('device','gpu','seed','torch_version','python_version','evidence_status','step_scale')},'results_keys':list(r),'checked_fields':chosen}
dist=load('docs/course-experiments/results/distillation.json')['results']
receipt['matched_distillation']={}
for task,value in dist['tasks'].items():
 runs=value['runs']; pairs={}
 for key,row in runs.items():
  if not key.endswith('_ce'): continue
  peer=key+'_kl'; a,b=row['training'],runs[peer]['training']
  pairs[key]={'initialization_equal':a['initialization_sha256']==b['initialization_sha256'],'batch_plan_equal':a['batch_plan_sha256']==b['batch_plan_sha256'],'steps':[a['steps'],b['steps']],'effective_tokens':[a['effective_supervised_tokens'],b['effective_supervised_tokens']]}
 receipt['matched_distillation'][task]={'teacher':value['teacher_provenance'],'data':value['data'],'pairs':pairs}
selection=dist['tasks']['attributes']['out_of_domain_gsm8k']['selection']
receipt['gsm8k_selection']={k:v for k,v in selection.items() if k!='skipped'}
v2=load('docs/selftrained/results/v2-final-public-results.json')
receipt['v2_summary']={'sha256':sha(ROOT/'docs/selftrained/results/v2-final-public-results.json'),'hf_revision':v2['hf_revision'],'architectures':{k:{'training_revision':v['training_revision'],'final_test_count':v['final_test']['count'],'final_test_revision':v['final_test']['revision'],'capability_acceptance_passed':v['capability_acceptance_passed']} for k,v in v2['architectures'].items()}}
receipt['mechanism_configs']={}
for name in ('fixed3','odd135'):
 p=Path(f'docs/lesson-products/results/mechanism-{name}-runner-result.json');d=load(p);t=d['execution']['training']
 receipt['mechanism_configs'][name]={'path':str(p),'sha256':sha(ROOT/p),'gpu':d['execution']['gpu_name'],**{k:t[k] for k in ('mode','last_step','seed','batch_size','initial_weights_sha256','work','reference_depth')}}
report=load('docs/lesson-products/results/final-report-v1.json')
receipt['products']={'report_sha256':sha(ROOT/'docs/lesson-products/results/final-report-v1.json'),'b5':report['b']['5'],'c':report['c'],'gates':report['gates'],'c_wording':report['shifts']['c_wording'],'failures':load('docs/lesson-products/results/observed-failure-examples-v1.json')}
base='https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016/'
manifest=load('docs/lesson-products/results/publication-package-v1.json')['files']
targets=('b/training-receipt.json','c/training-receipt.json','raw/final-report-v1.predictions.jsonl.gz')
def fetch(name):
 request=urllib.request.Request(base+name,headers={'User-Agent':'Tiny Perceptron technical source check'})
 path=WORK/name
 if not path.exists():
  with urllib.request.urlopen(request,timeout=60) as r: data=r.read()
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 actual={'sha256':sha(path),'bytes':path.stat().st_size}
 assert actual==manifest[name],(name,actual,manifest[name])
 return name,{'url':base+name,'path_external':str(path),**actual,'matches_manifest':True}
receipt['published_originals']=dict(concurrent.futures.ThreadPoolExecutor(3).map(fetch,targets))
for product in ('b','c'):
 d=json.loads((WORK/f'{product}/training-receipt.json').read_text())
 receipt['products'][product+'_training_receipt']={k:d[k] for k in ('random_initialization','last_step','selected_step','seed','weights_changed','environment','depth_weights','mode','initial_weights_sha256','selected_weights_sha256','work')}
raw=WORK/targets[-1]
counts=collections.Counter(); first=[]; total=0; scores=collections.defaultdict(lambda:collections.Counter())
examples={x['id']:x for x in receipt['products']['failures']['examples']}; matched=[]
with gzip.open(raw,'rt') as stream:
 for line in stream:
  row=json.loads(line);total+=1
  if len(first)<1:first.append({k:v for k,v in row.items() if k!='scores_by_depth'})
  counts[str((row.get('product'),row.get('split'),row.get('question'),row.get('loops')))]+=1
  if row['product']=='b' and row['split']=='test':
   key='b5_'+row['task'];scores[key]['count']+=1;scores[key]['correct']+=row['answers_by_depth']['5']==row['gold']
  if row['product']=='c' and row['split']=='test':
   key='c_'+row['question'];scores[key]['count']+=1;scores[key]['correct']+=row['selected']==row['gold']
   if row['question']=='choice':scores[key]['reorder_consistent']+=row['reverse_selected']==row['selected']
  if row.get('id') in examples and row.get('product')=='c' and row.get('split')=='wording_shift' and row.get('question')=='choice':
   ex=examples[row['id']]
   predicted=row['candidates'][max(range(len(row['raw_scores'])),key=lambda i:row['raw_scores'][i])]
   assert row['raw_scores']==ex['raw_scores'] and row['gold']==ex['gold'] and predicted==ex['predicted']
   matched.append(row['id'])
receipt['raw_prediction_index']={'rows':total,'groups':dict(counts),'first_three':first}
receipt['recomputed_product_counts']={k:dict(v) for k,v in scores.items()}
receipt['wording_failure_examples_raw_matched']=matched
assert len(matched)==3
assert scores['b5_boxes']=={'count':2956,'correct':2928}
assert scores['b5_sum']=={'count':42,'correct':2}
assert scores['c_choice']['correct']==11797 and scores['c_choice']['count']==11797
for name in ('info','rubric'):assert scores['c_'+name]['correct']==10890
tools=load('docs/course-experiments/results/tools.json')['results']; n=0; differences=0
roles={'system':7,'user':3,'assistant':4}
for key in ('samples','one_step_cap_samples'):
 for episode in tools[key]:
  first_event=next(e['generation'] for e in episode['trace'] if 'generation' in e)
  current=[dict(m) for m in first_event['messages'][:2]]
  for event in episode['trace']:
   if 'generation' not in event:continue
   g=event['generation'];ids=[1]
   for message in current:ids += [roles[message['role']]]+[v+8 for v in message['content'].encode()]+[2]
   ids += [4]
   assert ids==g['input_ids'] and len(ids)==g['input_tokens']
   n+=1;differences+=g['messages']!=current
   if event.get('executed') and event.get('finite_result'):
    current += [{'role':'assistant','content':g['samples'][0]['generated']},{'role':'user','content':f"TOOL_RESULT:{event['tool_result']}"}]
receipt['tool_snapshot_reconstruction']={'steps':n,'metadata_differences':differences,'all_original_input_ids_equal':True,'models_executed':False}
(OUT/'historical-check.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('recomputed_product_counts','wording_failure_examples_raw_matched','tool_snapshot_reconstruction')},ensure_ascii=False,indent=2))
