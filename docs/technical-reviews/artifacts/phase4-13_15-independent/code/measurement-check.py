import json,hashlib,math
from pathlib import Path
A=Path(__file__).resolve().parents[1]
p=A/'sources/posttraining-raw.json'; j=json.loads(p.read_text()); r=j['results']; pointers=[]
def read(pointer):
 pointers.append(pointer); obj=j
 for key in pointer.strip('/').split('/'): obj=obj[int(key)] if isinstance(obj,list) else obj[key]
 return obj
out={'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'same_as_published_immutable_file':p.read_bytes()==(A/'sources/published-posttraining.json').read_bytes()}
assert out['same_as_published_immutable_file']
out['provenance']={k:read('/'+k) for k in ['device','seed','torch_version','python_version','elapsed_seconds','revision','code_sha256']}
for name,h in out['provenance']['code_sha256'].items(): assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
out['config']=read('/results/config'); out['parameters']=read('/results/parameters'); out['splits']={}
for name in ['train','validation','test']:
 out['splits'][name]={k:read('/results/splits/'+name+'/'+k) for k in ['families','contexts','preference_pairs','sha256']}
familysets={k:set(v['families']) for k,v in out['splits'].items()}
assert all(not familysets[a]&familysets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
out['sft']={k:read('/results/sft/'+k) for k in ['demonstrations','processed_demonstration_draws','seconds','history']}
out['reward']={k:read('/results/reward/'+k) for k in ['unique_training_pairs','processed_pair_draws','seconds','history']}
out['ppo']={k:read('/results/ppo/'+k) for k in ['policy_updates','value_updates','sampled_actions','reused_action_draws','seconds','history']}
out['effective_tokens']=read('/results/effective_tokens'); out['schedule_completed']=read('/results/schedule_completed')
assert out['sft']['demonstrations']==44 and out['sft']['processed_demonstration_draws']==60*32
assert out['reward']['unique_training_pairs']==660 and out['reward']['processed_pair_draws']==300*64
assert (out['ppo']['policy_updates'],out['ppo']['value_updates'],out['ppo']['sampled_actions'],out['ppo']['reused_action_draws'])==(360,360,7680,23040)
assert out['reward']['history'][-1]['step']==300 and out['ppo']['history'][-1]['rollout_batch']==120
out['test_success']={}; test_rows=read('/results/evaluations/test/rows')
# Inspect only named raw sample fields, never label_source/candidate_source or annotation values.
# The raw rows container is accessed for iteration, but no row is printed wholesale.
for policy in ['sft','ppo']:
 correct=0; per_mode={}; sampledata=[]
 for i,row in enumerate(test_rows):
  base=f'/results/evaluations/test/rows/{i}'; mode=read(base+'/mode'); expected=read(base+'/expected_action'); probabilities=read(base+f'/policies/{policy}/probabilities'); choice=read(base+f'/policies/{policy}/chosen_action'); success=read(base+f'/policies/{policy}/full_request_success')
  independent={'number':0,'explain':1,'missing':3}[mode]; assert expected==independent and choice==max(range(4),key=lambda a:probabilities[a]); assert abs(sum(probabilities)-1)<2e-7 and success==(choice==independent)
  correct+=success; per_mode.setdefault(mode,[0,0]); per_mode[mode][0]+=success; per_mode[mode][1]+=1
  sampledata.append({'index':i,'family':read(base+'/family'),'mode':mode,'expected_action':expected,'chosen_action':choice,'success':success})
 aggregate=read(f'/results/evaluations/test/policies/{policy}/greedy_full_request_success'); assert aggregate['numerator']==correct and aggregate['denominator']==len(test_rows)==18
 assert math.isclose(aggregate['rate'],correct/18,abs_tol=4e-8)
 out['test_success'][policy]={'recomputed_numerator':correct,'denominator':len(test_rows),'by_mode':per_mode,'aggregate':aggregate,'raw_sample_fields':sampledata}
assert [out['test_success'][k]['recomputed_numerator'] for k in ['sft','ppo']]==[6,12]
trace=read('/results/ppo/first_rollout_trace'); out['first_rollout']={'epochs':len(trace),'samples_per_epoch':[],'max_advantage_error':0.0,'max_ratio_error':0.0,'fixed_old_records':True}
fields=['epoch','train_context_indices','sampled_actions','old_selected_log_probabilities','old_values','normalized_rm_rewards','fixed_advantages','log_probabilities_before_update','ratios_before_update','old_log_probability_sha256','reference_state_sha256']
selected=[{k:read(f'/results/ppo/first_rollout_trace/{i}/{k}') for k in fields} for i in range(len(trace))]
for t in selected:
 assert all(len(t[k])==64 for k in fields if isinstance(t[k],list)); out['first_rollout']['samples_per_epoch'].append(len(t['sampled_actions']))
 for a,v,reward in zip(t['fixed_advantages'],t['old_values'],t['normalized_rm_rewards']): out['first_rollout']['max_advantage_error']=max(out['first_rollout']['max_advantage_error'],abs(a-(reward-v)))
 for new,old,ratio in zip(t['log_probabilities_before_update'],t['old_selected_log_probabilities'],t['ratios_before_update']): out['first_rollout']['max_ratio_error']=max(out['first_rollout']['max_ratio_error'],abs(math.exp(new-old)-ratio))
 for k in ['train_context_indices','sampled_actions','old_selected_log_probabilities','old_values','normalized_rm_rewards','fixed_advantages','old_log_probability_sha256','reference_state_sha256']: assert t[k]==selected[0][k]
assert out['first_rollout']['max_advantage_error']<2e-7 and out['first_rollout']['max_ratio_error']<2e-7
assert all(v==1 for v in selected[0]['ratios_before_update'])
out['rounding']={'total_seconds_2dp':round(out['provenance']['elapsed_seconds'],2),'ppo_seconds_2dp':round(out['ppo']['seconds'],2)}
assert out['rounding']=={'total_seconds_2dp':2.12,'ppo_seconds_2dp':0.66}
(A/'execution/measurement-pointers.json').write_text(json.dumps(pointers,indent=2)+'\n')
(A/'execution/measurement-check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2)); print('PASS: raw measurement consistency only; no training or checkpoint inference was run.')
