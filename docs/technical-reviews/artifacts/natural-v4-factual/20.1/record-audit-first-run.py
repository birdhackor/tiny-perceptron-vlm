"""Inspect original saved run inputs/results. Recompute provenance/wiring/selection,
not training or semantic grading. No models/datasets are downloaded or loaded.
"""
from pathlib import Path
from collections import Counter
import json,hashlib,math,platform
import torch
OUT=Path('docs/technical-reviews/artifacts/natural-v4-factual/20.1'); inputs=[]
def read(path,lines=False):
 p=Path(path);b=p.read_bytes();inputs.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 return [json.loads(s) for s in b.decode().splitlines()] if lines else json.loads(b)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
root='outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1/review'
train=read(root+'/training.json'); config=read(root+'/adapter/adapter_config.json')
assert train['status']=='complete' or train['status']=='completed'
assert train['model']=='Qwen/Qwen3-VL-2B-Instruct'
assert len(train['history'])==train['completed_steps']==2077
assert [x['step'] for x in train['history']]==list(range(1,2078))
rows=sum(len(x['row_ids']) for x in train['history']); supervised=sum(x['supervised_tokens'] for x in train['history'])
initial=train['initial_adapter_tensors'];final=train['final_adapter_tensors']
parameters=sum(math.prod(v['shape']) for v in initial.values());changed=sum(initial[k]['sha256_values']!=final[k]['sha256_values'] for k in initial)
assert parameters==train['trainable_parameters']==1605632
assert changed==len(initial)==112 and set(initial)==set(final)
assert rows==train['trained_rows']==4154 and supervised==78872
assert train['frozen_parameter_samples_initial']==train['frozen_parameter_samples_final']
assert all('lora_' in n for n in train['optimizer_parameter_names'])
assert config['base_model_name_or_path']==train['model'] and config['peft_type']=='LORA' and config['r']==8
training={'model':train['model'],'revision':train['model_revision'],'official_environment':train['versions'],'device':train['device'],'gpu':train['gpu_name'],'seed':train['seed'],'updates':len(train['history']),'rows':rows,'supervised_targets':supervised,'adapter_parameters':parameters,'changed_of_total_adapter_tensors':f'{changed}/{len(initial)}','sampled_frozen_tensors':len(train['frozen_parameter_samples_initial']),'sampled_frozen_equal':True,'scope':'Original fixed GPU run record recomputation; not own GPU replication or complete base weight comparison'}
proto=read('docs/natural-assistant/v4/validation-protocol-lower-lr.json'); scores=read('docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json');selected=read('docs/natural-assistant/v4/selection.json'); public=read('docs/natural-assistant/v4/public-release.json')
assert sha('docs/natural-assistant/v4/validation-protocol-lower-lr.json')==selected['validation_protocol_sha256']==scores['artifact_binding']['protocol_sha256']
weights=scores['integer_weights'];base=scores['variants']['base']['correct_counts'];selection=[]
vr='outputs/natural-v4/modal-runs/validation-37219466611/natural-natural-v4-validation-37219466611-1/review'
for name,v in scores['variants'].items():
 gens=read(vr+'/generations-'+name+'.json');assert len(gens)==132
 binding=next(x for x in scores['artifact_binding']['generation_files'] if x['variant']==name)
 assert sha(vr+'/'+binding['name'])==binding['sha256']
 incomplete=[x['id'] for x in gens if not(x['stop_reason']=='eos' and x['ended_with_eos'] and x['generated_token_ids'][-1] in x['eos_token_ids'] and not x['truncated'] and not x['completion_unknown'])]
 assert sorted(incomplete)==sorted(v['incomplete_case_ids'])
 numerator=sum(weights[k]*v['correct_counts'][k] for k in weights);assert numerator==v['primary_numerator']
 eligible=not incomplete
 if name!='base':
  for k,value in v['correct_counts'].items():
   margin=0 if k in ['photo_summary','ordered_ocr','voice_typed_reference_chat','voice_actual_asr_chat'] else 1
   eligible=eligible and value>=base[k]-margin
  assert eligible==v['eligible']
 selection.append({'variant':name,'generations':len(gens),'incomplete':len(incomplete),'primary_numerator':numerator,'primary_denominator':75600,'eligible_adapter':eligible if name!='base' else None})
assert not any(x['eligible_adapter'] for x in selection) and selected['selected_variant']==public['selected_variant']=='base'
selection_result={'candidates':selection,'selected':'base','denominators':scores['denominators'],'scope':'Independently recomputed selection from original generation stopping records and committed semantic grade counts; no independent regrade of photos or claim of population improvement'}
studentroot='docs/natural-assistant/evidence/v4-research/student-selected-public-ui/actual-ui'
observer=read(studentroot+'/observer.jsonl',True);requests=read(studentroot+'/browser/requests.json');report=read(studentroot+'/report.json')
begins=[x for x in observer if x['kind']=='generate_begin'];ends=[x for x in observer if x['kind']=='generate_end'];transcripts=[x for x in observer if x['kind']=='transcribe_end'];chats=[x for x in requests if x['route']=='/api/chat'];tr_req=[x for x in requests if x['route']=='/api/transcribe']
assert len(begins)==len(ends)==len(chats)==4 and len(transcripts)==len(tr_req)==1
assert all(c['response']['prediction']==e['result']['prediction'] for c,e in zip(chats,ends))
assert tr_req[0]['response']['asr']==transcripts[0]['result']
history=[]
for begin,chat,end in zip(begins,chats,ends):
 row=begin['row'];assert row['history']==history and row['user']==chat['request']['prompt']
 content=[]
 if row['image']:content.append({'type':'image','image':row['image']})
 content.append({'type':'text','text':row['user']})
 history+=[{'role':'user','content':content},{'role':'assistant','content':[{'type':'text','text':end['result']['prediction']}]}]
assert all(any(part.get('type')=='image' for turn in begin['row']['history'] for part in turn['content']) for begin in begins[2:])
speech=next(c for c in chats if c['request'].get('speech'))
assert speech['response']['asr']['transcript']==transcripts[0]['result']['transcript']
assert speech['response']['asr']['submitted_text']==speech['request']['prompt'] and speech['response']['asr']['corrected'] is True
reset=next(x for x in observer if x['kind']=='operation_end' and x['route']=='/api/reset')
assert reset['state']=={'history_messages':0,'assets':0,'transcriptions':0,'asset_files':[]}
for x in observer[0]['sources']:assert sha(x['path'])==x['sha256']
options=next(x['options'] for x in observer if x['kind']=='load_core_begin');assert options['adapter']=='None' and options['asr_model']==public['asr_model']['repo']
student={'chats':len(chats),'transcriptions':len(transcripts),'each_actual_result_matches_http':True,'history_matches_every_prior_actual_turn':True,'prior_photo_in_speech_and_followup':True,'original_transcript_and_user_edit_separate':True,'reset_clears_history_assets_transcripts':True,'runtime_sources_match_current':True,'pinned_options':options,'official_environment':report['dependencies'],'scope':'Direct original public student smoke record audit, not own pretrained model replication or accuracy benchmark'}
result={'own_recompute_environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'},'training':training,'selection':selection_result,'student':student,'original_input_receipts':inputs}
(OUT/'record-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='original_input_receipts'},ensure_ascii=False,indent=2));print('ORIGINAL_INPUTS',len(inputs))
