"""Read original measurement files, retain small portable projections, and replay their arithmetic."""
from pathlib import Path
import argparse,datetime,hashlib,json,platform,sys
import torch
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).parent
CORE=Path('/workspace/work/omni-core-completed-40988311100/verified');FINAL=Path('/workspace/work/omni-final-completed-41004229262/verified')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def load(p):return json.loads(p.read_bytes())
def selected(v,keys):return {k:v[k] for k in keys if k in v}
def extract():
 proof={'created_at':datetime.datetime.now(datetime.UTC).isoformat(),'reviewer_task':'/root/omni_p5_technical_training_whole','environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'CPU'},'scope':'Existing originals read only; no model forward or training; only small measurement/state projections retained. Replay uses this portable proof without ignored outputs or outside-Git archives.'}
 expected={'simple_models':4.3,'text_foundation':16.0,'real_text':23.9,'tokenizer':16.2,'sft':31.7,'sft_ablation':23.5,'style':29.5,'safety':22.2,'lora':24.5,'encoders':10.2,'projector':15.9,'vqa':83.8,'dpo':18.7,'posttraining':2.1,'modern':44.6,'moe':65.7,'efficiency':32.3,'precision':15.0,'flash_probe':4.5,'quantization':7.8,'distillation':90.1,'joint':33.4,'multimodal_distillation':53.0}
 proof['historical_timings']=[]
 def config_rows(value,path='results'):
  rows=[]
  if isinstance(value,dict):
   keep={k:value[k] for k in ('steps','optimizer_updates','batch_size','effective_tokens','effective_supervised_tokens','records','count','families','training_examples','parameters','config','context') if k in value and not isinstance(value[k],list)}
   if keep:rows.append({'locator':path,**keep})
   for k,v in value.items():
    if k not in ('samples','generated_samples','history','artifacts','rows','trace','loss_trace','code_sha256','config'):rows+=config_rows(v,path+'/'+k)
  return rows
 for name,shown in expected.items():
  p=ROOT/'docs/course-experiments/results'/f'{name}.json';v=load(p)
  proof['historical_timings'].append({'name':name,'source':ref(p),'display_seconds':shown,**selected(v,['elapsed_seconds','timing_scope','device','gpu','seed','python_version','torch_version','modal','evidence_status']),'configuration_locations':config_rows(v['results'])})
 proof['formal_jobs']=[]
 for directory in sorted((CORE/'core').iterdir()):
  rp=directory/'training/training-result.json';jp=directory/'job-result.json';v=load(rp);j=load(jp)
  row={'job':directory.name,'training_result':ref(rp),'job_result':ref(jp),'observed':selected(v,['complete','updates','config','trainable_parameters','initial_parameter_sha256','latest_parameter_sha256','optimizer_updates_observed','best','final_opened','runtime','source_unchanged_during_training']),'job_origin':selected(j,['kind','stage','success','attempt','started_at','finished_at','runtime','execution_host','final_opened','Modal_dispatched','formal_recipe_sha256','data_definition_sha256','metrics_sha256']),'files':{}}
  for name in ('weights.safetensors','best-continuation.pt','latest-continuation.pt'):
   p=directory/'training'/name;assert sha(p)==v['files'][name]['sha256'] and p.stat().st_size==v['files'][name]['bytes']
   d={'original':ref(p),'matches_training_receipt':True}
   if name.endswith('.pt'):
    state=torch.load(p,map_location='cpu',weights_only=True)
    d['state_fields']={'keys':sorted(state),'step':state['step'],'optimizer_state_entries':len(state['optimizer']['state']),'has_model':isinstance(state['model'],dict) and bool(state['model']),'has_python_rng':'python_rng' in state,'torch_rng_numel':state['torch_rng'].numel(),'sampler_rng_numel':state['sampler_rng'].numel(),'metadata_keys':sorted(state['metadata'])}
   row['files'][name]=d
  proof['formal_jobs'].append(row)
 selection=load(CORE/'frozen/model-selection.json');opening=load(FINAL/'frozen/final-opening.json')
 proof['pre_final_selection']={'original':ref(CORE/'frozen/model-selection.json'),'repository':ref(ROOT/'docs/omni-products/v1/model-selection.json'),'selection':selected(selection,['created_at','final_opened','selected','temperature','threshold','policy_feasible','model_files','formal_recipe_sha256','data_definition_sha256','metrics_sha256','scope']),'opening_original':ref(FINAL/'frozen/final-opening.json'),'opening':selected(opening,['model_selection_sha256','created_at','runtime']),'final_frozen_selection_sha256':sha(FINAL/'frozen/model-selection.json')}
 proof['selected_four_weights']={kind:{'job':selection['selected'][kind],**ref(CORE/'core'/selection['selected'][kind]/'training/weights.safetensors')} for kind in ('av','choice','codec','talker')}
 cpath=FINAL/'measurements/choice/results.json';c=load(cpath)['primary'];metrics=load(FINAL/'frozen/metrics.json')['C']
 proof['C_final']={'source':ref(cpath),'metrics_source':ref(FINAL/'frozen/metrics.json'),'selection_T':selection['temperature'],'selection_tau':selection['threshold'],'criteria':selected(metrics,['final_families','choice_minimum_correct','info_minimum_correct','rubric_minimum_correct','minimum_accepted','maximum_empirical_accepted_risk','accepted_rule','accepted_error']),'reported_types':{kind:value['primary'] for kind,value in c['types'].items()},'reported_policy':{k:v for k,v in c['policy'].items() if k!='rows'},'rows':[]}
 for row in c['rows']:
  proof['C_final']['rows'].append({'family':row['family'],'tasks':{kind:{k:v[k] for k in ('answer','gold','p_max')} for kind,v in row['tasks'].items()},'choice_info_answer':row['choice_info']['answer'],'world_supported':row['world_supported'],'choice_information_sufficient':row['choice_information_sufficient']})
 proof['final_report']={'source':ref(FINAL/'final-report.json'),'value':load(FINAL/'final-report.json')}
 vpath=ROOT/'docs/selftrained/results/v2-final-public-results.json';v=load(vpath)
 proof['historical_v2']={'source':ref(vpath),'hf_repo':v['hf_repo'],'hf_revision':v['hf_revision'],'architectures':{name:{'initialization':row['capacity']['initialization'],'architecture':row['capacity']['config']['architecture'],'safe_weights':row['safe_weights'],'final_test':selected(row['final_test'],['run_id','count','evaluation_complete','criteria_met','all_original_criteria_met'])} for name,row in v['architectures'].items()}}
 proof['historical_lesson_products']={}
 for name in ('b','c'):
  p=ROOT/f'docs/lesson-products/results/initial-{name}-execution.json';v=load(p)
  proof['historical_lesson_products'][name]={'source':ref(p),'run_id':v['run_id'],'status':v['status'],'job':v['job'],'training':selected(v['training'],['random_initialization','weights_changed','completed_steps','steps','best_step','initial_weights_sha256','selected_weights_sha256','initialization','updates'])}
 return proof

def replay(proof):
 out={'created_at':datetime.datetime.now(datetime.UTC).isoformat(),'historical_display_roundings':{},'formal_job_count':len(proof['formal_jobs']),'training_complete':True,'all_continuation_states_present':True}
 for row in proof['historical_timings']:
  ok=round(row['elapsed_seconds'],1)==row['display_seconds'];assert ok,row['name'];out['historical_display_roundings'][row['name']]={'observed':row['elapsed_seconds'],'rounded':round(row['elapsed_seconds'],1),'display':row['display_seconds'],'matches':ok}
 counts={}
 for row in proof['formal_jobs']:
  obs=row['observed'];counts[row['job_origin']['kind']]=counts.get(row['job_origin']['kind'],0)+1
  assert obs['complete'] and obs['updates']==obs['config']['updates'] and obs['optimizer_updates_observed'] and not obs['final_opened']
  assert obs['runtime']['device']=='cpu' and row['job_origin']['runtime']['device']=='cpu' and row['job_origin']['success']
  assert obs['initial_parameter_sha256']!=obs['latest_parameter_sha256']
  for name in ('best-continuation.pt','latest-continuation.pt'):
   s=row['files'][name]['state_fields'];assert s['has_model'] and s['optimizer_state_entries']>0 and s['has_python_rng'] and s['torch_rng_numel']>0 and s['sampler_rng_numel']>0
 out['formal_kind_counts']=counts;assert out['formal_job_count']==10
 p=proof['pre_final_selection'];assert p['original']['sha256']==p['repository']['sha256']==p['final_frozen_selection_sha256']==p['opening']['model_selection_sha256']
 assert not p['selection']['final_opened'] and p['selection']['created_at']<p['opening']['created_at'];out['selection_precedes_final_opening']=True
 total=sum(row['bytes'] for row in proof['selected_four_weights'].values());assert total==2192940
 out['four_weight_bytes']=total;out['four_weight_MiB']=total/1024**2
 c=proof['C_final'];rows=c['rows'];assert len(rows)==c['criteria']['final_families']==60
 correct={kind:sum(r['tasks'][kind]['answer']==r['tasks'][kind]['gold'] for r in rows) for kind in ('choice','info','rubric')}
 for kind,n in correct.items():assert n==c['reported_types'][kind]['correct']
 accepts=[r for r in rows if r['tasks']['choice']['p_max']>=c['selection_tau'] and r['tasks']['choice']['answer']!='none' and r['choice_info_answer']=='yes']
 errors=sum(r['tasks']['choice']['answer']!=r['tasks']['choice']['gold'] or not r['world_supported'] or not r['choice_information_sufficient'] for r in accepts)
 assert len(accepts)==c['reported_policy']['accepted'] and errors==c['reported_policy']['errors']
 risk=errors/len(accepts);assert risk==c['reported_policy']['empirical_risk']
 out['C_recomputed']={'family_denominator':len(rows),'correct':correct,'accepted':len(accepts),'errors':errors,'accepted_risk':risk,'choice_pass':correct['choice']>=c['criteria']['choice_minimum_correct'],'info_pass':correct['info']>=c['criteria']['info_minimum_correct'],'rubric_pass':correct['rubric']>=c['criteria']['rubric_minimum_correct'],'fixed_policy_pass':len(accepts)>=c['criteria']['minimum_accepted'] and risk<=c['criteria']['maximum_empirical_accepted_risk']}
 assert not out['C_recomputed']['choice_pass'] and not out['C_recomputed']['fixed_policy_pass']
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');args=p.parse_args()
 path=OUT/'measurement-projection-v1.json'
 if not args.replay:
  value=extract();path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 value=load(path);result=replay(value);target=OUT/('measurement-replay-v1.json' if args.replay else 'measurement-results-v1.json');target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'output':str(target.relative_to(ROOT)),'bytes':target.stat().st_size,'sha256':sha(target),'formal_jobs':result['formal_job_count'],'four_weights':result['four_weight_bytes'],'C':result['C_recomputed']}))
