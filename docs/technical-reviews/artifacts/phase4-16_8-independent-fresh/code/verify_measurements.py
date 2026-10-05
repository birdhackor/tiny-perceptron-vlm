import json,statistics,hashlib,sys
from pathlib import Path
ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/technical-reviews/artifacts/phase4-16_8-independent-fresh'
eff=json.loads((OUT/'inputs/efficiency.json').read_bytes())
flash=json.loads((OUT/'inputs/flash_probe.json').read_bytes())
pointers=[]
def get(obj,path):
 pointers.append(path)
 for k in path.strip('/').split('/'):obj=obj[k]
 return obj
summary={}
probe=get(eff,'/results/manual_vs_sdpa')
config=get(eff,'/results/models/mha/model/config')
assert probe['shape']==[4,53] and probe['dtype']=='torch.float32'
assert config['heads']==config['kv_heads']==4
assert probe['output_max_error']<1e-5 and probe['gradient_max_error']<1e-8
names=probe['backend']['operator_names']
assert 'aten::_scaled_dot_product_efficient_attention' in names and 'aten::_efficient_attention_forward' in names
assert not any('flash' in x for x in names)
times={}
for route in ['manual','sdpa']:
 t=probe['timings'][route]
 recomputed=statistics.median(t['samples_seconds'])
 assert recomputed==t['median_seconds']
 assert len(t['samples_seconds'])==t['measured_calls']==9 and t['warmup_calls']==3
 times[route]={'median_ms':recomputed*1000,'rounded_ms':round(recomputed*1000,3),'sample_count':len(t['samples_seconds']),'warmup':t['warmup_calls']}
assert times['manual']['rounded_ms']==2.990 and times['sdpa']['rounded_ms']==2.346
updates={}
for route,expected_ms in [('ordinary',14.070),('sdpa',12.849)]:
 path='/results/update_variants/'+route+'/training'
 t=get(eff,path)
 selected={k:t[k] for k in ['requested_steps','steps','optimizer_updates','skipped_updates','effective_tokens','batch_size','warm_step_median_seconds']}
 assert t['requested_steps']==t['steps']==t['optimizer_updates']==40
 assert t['skipped_updates']==0 and t['effective_tokens']==2328
 assert round(t['warm_step_median_seconds']*1000,3)==expected_ms
 updates[route]=selected
weight_error=get(eff,'/results/update_variants/sdpa/weight_max_error_from_ordinary_after_updates')
assert f'{weight_error:.5e}'=='3.51369e-05'
summary['efficiency']={'provenance':{k:get(eff,'/'+k) for k in ['revision','device','seed','torch_version','python_version','gpu']},'shape_batch_positions':probe['shape'],'query_and_kv_heads':[config['heads'],config['kv_heads']],'dtype':probe['dtype'],'output_max_error':probe['output_max_error'],'gradient_max_error':probe['gradient_max_error'],'operator_names':names,'timing_full_TinyLM':times,'updates':updates,'weight_max_error_after_updates':weight_error}

configuration=get(flash,'/results/configuration')
assert configuration['shape_B_H_T_D']==[2,4,512,32]
assert configuration['attn_mask'] is None and configuration['is_causal'] is True
assert configuration['dropout_p']==0 and configuration['requested_backend']=='SDPBackend.FLASH_ATTENTION only'
assert get(flash,'/seed')==get(flash,'/results/seed')==42
routes={}
for dtype,expected in [('fp16',(0.001953125,0.000244140625,(0.01,0.01),(0.03,0.03))),('bf16',(0.015625,0.001953125,(0.06,0.04),(0.08,0.08)))]:
 correctness=get(flash,'/results/routes/'+dtype+'/correctness')
 profiles=get(flash,'/results/routes/'+dtype+'/profiles')
 output=correctness['output'];gradients=correctness['gradients']
 assert output['max_absolute_error']==expected[0]
 assert max(g['max_absolute_error'] for g in gradients.values())==expected[1]
 assert (output['atol'],output['rtol'])==expected[2]
 checks=[output]+list(gradients.values())
 for g in gradients.values():assert (g['atol'],g['rtol'])==expected[3]
 # Each measured max error is below atol alone, proving the elementwise
 # allclose bound regardless of unknown reference magnitudes; no reliance
 # on a summary verdict or on GPU reruns.
 assert all(x['finite'] and x['max_absolute_error']<=x['atol'] for x in checks)
 observed={}
 for mode,p in profiles.items():
  operators=p['operator_names'];kernels=p['cuda_kernel_names']
  assert 'aten::_scaled_dot_product_flash_attention' in operators and kernels
  if mode=='forward_backward':assert 'aten::_scaled_dot_product_flash_attention_backward' in operators
  observed[mode]={'relevant_operators':[x for x in operators if 'flash_attention' in x],'cuda_kernel_names':kernels}
 routes[dtype]={'output':output,'gradients':gradients,'gradient_max_absolute_error':max(g['max_absolute_error'] for g in gradients.values()),'profiles':observed,'independent_sufficient_atol_bound_passed':True}
summary['flash_probe']={'provenance':{k:get(flash,'/'+k) for k in ['revision','device','seed','torch_version','python_version','gpu']},'configuration':configuration,'routes':routes,'elements_per_output':2*4*512*32,'elements_in_three_gradients':3*2*4*512*32}
summary['read_json_pointers']=pointers
summary['original_json_sha256']={name:hashlib.sha256((OUT/'inputs'/(name+'.json')).read_bytes()).hexdigest() for name in ['efficiency','flash_probe']}
summary['limitation']='Audits existing measurement scalars, timing samples, declared comparisons and actual operator/kernel trace names. Does not retrain, load weights, recreate GPU tensors, or evaluate model quality.'
(OUT/'measurement_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(OUT/'measurement_stdout.txt').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
