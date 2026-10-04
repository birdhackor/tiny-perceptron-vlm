"""Independent bounded CPU/record audit for lesson 19.6; no checkpoint or training."""
from collections import Counter, defaultdict
from pathlib import Path
import contextlib, copy, hashlib, io, json, math, platform, re, sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from tiny_perceptron.capstone import CapstoneModel, build_dataset, modality_tensors, prepare_batch
from tiny_perceptron.multimodal import log_mel, tone, expand_modalities
ART=ROOT/'docs/technical-reviews/artifacts/natural-v4-factual/19.6'
E=ROOT/'docs/course-experiments/capstone-evidence'
torch.set_num_threads(2)
torch.manual_seed(42)
environment={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','cuda_available':str(torch.cuda.is_available()),'threads':str(torch.get_num_threads())}
print('ENVIRONMENT',json.dumps(environment)); assert torch.__version__=='2.14.1+cpu'
# Execute the section's exact fenced Python with no substitutions.
section=(ART/'section-19.6-original.md').read_text()
code=re.search(r'```python\n(.*?)\n```',section,re.S)[1]
namespace={};buf=io.StringIO()
with contextlib.redirect_stdout(buf):exec(compile(code,'course/chapters/19.md#19.6','exec'),namespace)
print('EXACT_LESSON_OUTPUT\n'+buf.getvalue())
(ART/'exact-lesson.py').write_text(code+'\n')
row=namespace['row']; model=namespace['model']; batch=namespace['batch']; labels=namespace['labels']
assert namespace['image'].shape==(3,16,16) and namespace['audio'].shape==(16,)
assert namespace['result']['logits'].shape==(*labels.shape,264)
assert not namespace['result']['logits'].requires_grad
assert all(p.grad is None for p in model.parameters())
# Independent byte-level target alignment and manual 4x4 spatial block averages.
prefix_len=len(batch['ids'][0])-len(row['answer'].encode())
answer_ids=[v+8 for v in row['answer'].encode()]+[2]
assert labels[0,:prefix_len-1].tolist()==[-100]*(prefix_len-1)
assert labels[0,prefix_len-1:].tolist()==answer_ids
assert batch['ids'][0,prefix_len-1].item()==4 and labels[0,-1].item()==2
assert int((batch['ids']==5).sum())==int((batch['ids']==6).sum())==1
image=namespace['image']; pooled=F.adaptive_avg_pool2d(image.unsqueeze(0),(4,4)).flatten(1)
manual=torch.stack([image[c,y:y+4,x:x+4].mean() for c in range(3) for y in range(0,16,4) for x in range(0,16,4)]).unsqueeze(0)
print('POOL_FLOAT32_MAX_ABS_DIFF',float((pooled-manual).abs().max()),'ATOL',1e-6)
assert torch.allclose(pooled,manual,atol=1e-6,rtol=0)
# Independently reconstruct the triangular HTK mel bank and log-power summary.
a=row['audio'];frequency=(440 if a['pitch']=='low' else 880)+a['delta']+(a['variation']-1)*2
wave=tone(frequency,seconds=.04);power=torch.stft(wave,400,160,window=torch.hann_window(400),return_complex=True).abs().square()
mel_points=torch.linspace(0,2595*math.log10(1+8000/700),18)
hz=700*(10**(mel_points/2595)-1); freq_bins=torch.linspace(0,8000,201)
bank=torch.stack([torch.minimum((freq_bins-hz[i])/(hz[i+1]-hz[i]),(hz[i+2]-freq_bins)/(hz[i+2]-hz[i+1])).clamp(min=0) for i in range(16)])
mel=(bank@power).clamp(min=1e-8).log(); manual_audio=mel.mean(-1)
assert torch.allclose(manual_audio,namespace['audio'],atol=1e-6,rtol=0)
assert torch.allclose(mel.mean(-1),mel.flip(-1).mean(-1),atol=2e-6,rtol=0)
# Capture inputs/output of both projectors and verify replacement of the marker positions.
captured={}
handles=[]
for name in ['image_projector','audio_projector']:
 def hook(mod,args,out,name=name):captured[name]={'input':args[0].detach().clone(),'output':out.detach().clone()}
 handles.append(getattr(model,name).register_forward_hook(hook))
def language_hook(mod,args,kwargs):captured['language']=kwargs['embeddings'].detach().clone()
handles.append(model.language.register_forward_pre_hook(language_hook,with_kwargs=True))
before={n:p.detach().clone() for n,p in model.named_parameters()}
with torch.no_grad():model(**batch)
for h in handles:h.remove()
for name,marker,insize in [('image_projector',5,48),('audio_projector',6,16)]:
 assert captured[name]['input'].shape==(1,insize) and captured[name]['output'].shape==(1,64)
 assert torch.equal(captured['language'][0,(batch['ids'][0]==marker).nonzero()[0,0]],captured[name]['output'][0])
assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
# Exercise missing-audio code and markers: batch has harmless zero filler, prefix has no slot.
splits,manifest=build_dataset();color_row=next(r for r in splits['train'] if r['task']=='image_color')
ci,ca=modality_tensors(color_row); cb,cl=prepare_batch([color_row]);assert ca is None and int((cb['ids']==6).sum())==0
with torch.no_grad():cr=model(**cb)
assert cr['logits'].shape[:2]==cl.shape
print('CPU_FEATURE_CHECKS',json.dumps({'row_id':row['id'],'question':row['user'],'answer':row['answer'],'image_shape':list(image.shape),'pooled_items':pooled.numel(),'waveform_samples':len(wave),'frequency_hz':frequency,'stft_power_shape':list(power.shape),'log_mel_shape':list(mel.shape),'audio_items':len(manual_audio),'projection_width':64,'input_positions':batch['ids'].shape[1],'answer_targets':len(answer_ids),'vocabulary':264,'gradient_graph':False,'parameters_unchanged':True,'image_only_audio':None,'image_only_audio_slots':0}))
data=json.loads((E/'deployment/data.json').read_text());assert data['splits']==splits and data['manifest']==manifest
assert len(splits['validation'])==84 and len(splits['test'])==90
assert manifest['families']['validation']['modalities']==['modalities:blue:circle']
assert manifest['families']['test']['modalities']==['modalities:green:square']
assert {r['image']['shape'] for r in splits['train'] if r['image'] and r['image']['color']=='blue'}=={'square'}
# Text byte IDs contain only explicit system/question text and modality markers.
def prompt(r):
 return [1,7]+[v+8 for v in r['system'].encode()]+[2,3]+([5] if r['image'] else [])+([6] if r['audio'] else [])+[v+8 for v in r['user'].encode()]+[2,4]
def trace_check(t):
 ids=t['generated_ids'];raw=bytes(v-8 for v in ids if v>=8).decode('utf8',errors='replace')
 eos=bool(ids and ids[-1]==2 and 2 not in ids[:-1])
 assert t['raw']==raw and t['eos']==eos
 assert (t['stop_reason']=='eos')==eos
 return raw,eos
def audit(path,rows):
 d=json.loads(path.read_text());byid={r['id']:r for r in rows};assert len(byid)==len(rows)
 assert len(d['records'])==len(rows) and {r['id'] for r in d['records']}==set(byid)
 counts=defaultdict(lambda:{'count':0,'action_correct':0,'end_to_end_correct':0});results={}
 for rec in d['records']:
  r=byid[rec['id']];t=rec['action_trace'];raw,eos=trace_check(t)
  assert t['prompt_ids']==prompt(r) and rec['task']==r['task'] and rec['expected_action']==r['answer']
  expected_final=r['answer'].split(':',1)[1]
  if r['answer'].startswith('TOOL:'):
   args=r['answer'].split(':')[-1].split('+');expected_final=str(int(args[0])+int(args[1]))
  assert rec['expected_final']==expected_final
  answer=None
  if eos and ((raw.startswith('DIRECT:') and len(raw)>7) or (raw.startswith('ASK:') and len(raw)>4)):answer=raw.split(':',1)[1]
  elif eos and rec['final_trace'] is not None:
   fr,fe=trace_check(rec['final_trace'])
   if fe and fr.startswith('DIRECT:') and len(fr)>7:answer=fr[7:]
  action_ok=eos and raw==r['answer'];full_ok=action_ok and answer==expected_final
  assert rec['action_correct']==action_ok and rec['end_to_end_correct']==full_ok and rec['answer']==answer
  c=counts[r['task']];c['count']+=1;c['action_correct']+=int(action_ok);c['end_to_end_correct']+=int(full_ok)
  results[r['id']]={'correct':full_ok,'raw':raw,'eos':eos,'expected':r['answer'],'task':r['task']}
 assert dict(counts)==d['by_task'] and d['count']==len(rows)
 assert d['action_correct']==sum(v['action_correct'] for v in counts.values())
 assert d['end_to_end_correct']==sum(v['end_to_end_correct'] for v in counts.values())
 return dict(counts),results
validation={};vr={}
for stage in ['sft','joint','dpo']:
 validation[stage],vr[stage]=audit(E/stage/'validation.json',splits['validation'])
 assert [validation[stage][t]['end_to_end_correct'] for t in ['image_color','image_shape','audio','joint']]=={'sft':[0,0,0,0],'joint':[9,0,6,18],'dpo':[9,0,6,14]}[stage]
for rid,v in vr['joint'].items():
 if v['task']=='image_shape':assert v['raw']=='DIRECT:square' and v['eos']
regressions=[]
for rid,j in vr['joint'].items():
 d=vr['dpo'][rid]
 if j['task']=='joint' and j['correct'] and not d['correct']:
  assert d['raw']=='DIRECT:red,'+j['expected'].split(',')[1] and d['eos']
  regressions.append({'id':rid,'truth':j['expected'],'joint':j['raw'],'dpo':d['raw']})
assert len(regressions)==4
test={};tr={}
for stage in ['joint','dpo']:
 test[stage],tr[stage]=audit(E/'deployment'/f'test-{stage}.json',splits['test'])
 assert [test[stage][t]['end_to_end_correct'] for t in ['image_color','image_shape','audio','joint']]==[9,0,6,18]
for v in tr['joint'].values():
 if v['task']=='image_shape':assert v['raw']=='DIRECT:circle' and v['eos']
# Recreate interventions independently and actually regenerate all affected pixels/features.
image_rows=[];audio_rows=[];input_changes=[]
for r in splits['test']:
 if r['task'] not in ['image_color','image_shape','joint']:continue
 s=copy.deepcopy(r)
 if r['task']=='image_shape':s['image']['shape']='circle';payload='circle'
 else:s['image']['color']='blue';payload='blue'+(','+r['audio']['pitch'] if r['task']=='joint' else '')
 s['answer']='DIRECT:'+payload;s['id']=r['id']+'-image-swap';image_rows.append(s)
 oi,oa=modality_tensors(r);si,sa=modality_tensors(s)
 assert not torch.equal(oi,si) and prompt(r)==prompt(s) and r['user']==s['user']
 assert (oa is None and sa is None) or torch.equal(oa,sa)
 input_changes.append({'id':r['id'],'pixels_changed':int((oi!=si).sum()),'unchanged_audio':True})
 if r['task']=='joint':
  s=copy.deepcopy(r);s['audio']['pitch']='high' if r['audio']['pitch']=='low' else 'low'
  s['answer']='DIRECT:green,'+s['audio']['pitch'];s['id']=r['id']+'-audio-swap';audio_rows.append(s)
  si,sa=modality_tensors(s);assert torch.equal(oi,si) and not torch.equal(oa,sa) and prompt(s)==prompt(r)
iscores,ir=audit(E/'deployment/test-joint-image-swaps.json',image_rows)
ascores,ar=audit(E/'deployment/test-joint-audio-swaps.json',audio_rows)
pairfile=json.loads((E/'deployment/test-joint-image-pairs.json').read_text());pairs={p['original_id']:p for p in pairfile['pairs']}
paired=defaultdict(lambda:{'count':0,'original_correct':0,'swapped_correct':0,'both_correct':0,'changed':0})
for s in image_rows:
 rid=s['id'].removesuffix('-image-swap');o=tr['joint'][rid];v=ir[s['id']];p=pairs[rid]
 assert p['original_expected']==o['expected'] and p['swapped_expected']==v['expected']
 assert p['original_generated']==o['raw'] and p['swapped_generated']==v['raw']
 both=o['correct'] and v['correct'];changed=o['raw']!=v['raw'];assert p['both_end_to_end_correct']==both and p['generated_answer_changed']==changed
 c=paired[s['task']];c['count']+=1;c['original_correct']+=int(o['correct']);c['swapped_correct']+=int(v['correct']);c['both_correct']+=int(both);c['changed']+=int(changed)
assert dict(paired)=={'image_color':{'count':9,'original_correct':9,'swapped_correct':9,'both_correct':9,'changed':9},'image_shape':{'count':9,'original_correct':0,'swapped_correct':9,'both_correct':0,'changed':0},'joint':{'count':18,'original_correct':18,'swapped_correct':18,'both_correct':18,'changed':18}}
assert pairfile['count']==36 and len(pairs)==36 and pairfile['both_end_to_end_correct']==27
assert len(ar)==18 and all(v['correct'] and tr['joint'][rid.removesuffix('-audio-swap')]['correct'] for rid,v in ar.items())
# Independently count distributions, variants and constant-output baselines.
distributions={}
for split in ['train','validation','test']:
 distributions[split]={t:dict(Counter(r['answer'].split(':',1)[1] for r in splits[split] if r['task']==t)) for t in ['image_color','image_shape','audio','joint']}
assert distributions['validation']['image_color']=={'blue':9}
assert distributions['validation']['audio']=={'high':3,'low':3}
assert distributions['validation']['joint']=={'blue,low':9,'blue,high':9}
frequencies={pitch:sorted({(440 if pitch=='low' else 880)+r['audio']['delta']+(r['audio']['variation']-1)*2 for rs in splits.values() for r in rs if r['audio'] and r['audio']['pitch']==pitch}) for pitch in ['low','high']}
# Provenance: fixed GPU source records and stage lineage; record validation only.
run_receipts={};previous=None
for stage,stem in [('pretrain','capstone_pretrain'),('sft','capstone_sft'),('joint','capstone_joint'),('dpo','capstone_preference')]:
 d=json.loads((ROOT/f'docs/course-experiments/results/{stem}.json').read_text());r=d['results'];local=json.loads((E/stage/'train-report.json').read_text())
 assert r==local and r['data_manifest']==manifest and r['schedule_completed'] and not r['test_evaluated'] and d['seed']==42
 assert r['parent_checkpoint_sha256']==previous;previous=r['inference_export']['sha256']
 run_receipts[stage]={k:d[k] for k in ['revision','device','seed','torch_version','python_version','gpu','elapsed_seconds','timing_scope']}
 run_receipts[stage].update({k:r[k] for k in ['steps','effective_tokens','elapsed_training_seconds','parent_checkpoint_sha256','objective']})
 run_receipts[stage]['inference_export_sha256']=previous
 print('RUN_PROVENANCE',stage,json.dumps(run_receipts[stage]))
deployment=json.loads((ROOT/'docs/course-experiments/results/capstone_deployment.json').read_text());dr=deployment['results']
assert dr['recipe_frozen_before_test'] is True and dr['recommended_stage']=='joint'
assert dr['final_model']['sha256']==run_receipts['joint']['inference_export_sha256']
assert json.loads((E/'deployment/test-joint-image-swaps.json').read_text())['source_checkpoint_sha256']==dr['final_model']['sha256']
assert json.loads((E/'deployment/test-joint-audio-swaps.json').read_text())['source_checkpoint_sha256']==dr['final_model']['sha256']
print('DEPLOYMENT_PROVENANCE',json.dumps({k:deployment[k] for k in ['revision','device','seed','torch_version','python_version','gpu','elapsed_seconds','timing_scope']}))
# Necessary prerequisites: many-image-vector expansion and frequency axes.
emb=torch.nn.Embedding(264,8);x,y=expand_modalities(torch.tensor([1,5,4,73,2]),torch.tensor([-100,-100,-100,73,2]),emb,{5:torch.zeros(3,8)},{5})
assert x.shape==(1,6,8) and y.tolist()==[[-100,-100,-100,-100,73,2]]
axes=[]
for f in [440,880]:
 p=torch.stft(tone(f,seconds=.2),400,160,window=torch.hann_window(400),return_complex=True).abs().square()
 peak=int(p.mean(-1).argmax());assert p.shape==(201,21) and peak==f//40
 axes.append({'frequency_hz':f,'power_shape':list(p.shape),'peak_bin':peak,'bin_hz':40})
result={'environment':environment,'lesson_dimensions':{'image':[3,16,16],'audio':[16],'logits':list(namespace['result']['logits'].shape),'answer_targets':len(answer_ids),'prefix_length':prefix_len,'row':row},'validation_scores':validation,'validation_distributions':distributions['validation'],'dpo_joint_regressions':regressions,'test_scores':test,'test_distributions':distributions['test'],'image_pair_scores':dict(paired),'image_input_changes':input_changes,'audio_swap_scores':ascores,'audio_both_correct':18,'frequency_ranges_hz':{k:[min(v),max(v)] for k,v in frequencies.items()},'run_receipts':run_receipts,'deployment_recipe_frozen_before_test':True,'prerequisite_axes':axes,'limitations':'Independent recomputation of retained GPU-generated token traces and deterministic CPU preprocessing only. No weights fetched; no GPU execution, training, model-quality replication, or general natural-image/speech claim.'}
(ART/'verification-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
registered=[ROOT/'tiny_perceptron/capstone.py',ROOT/'tiny_perceptron/multimodal.py',ROOT/'tiny_perceptron/data.py',ROOT/'tiny_perceptron/model.py',ROOT/'scripts/course_experiments/capstone.py',ROOT/'scripts/course_experiments/capstone_deployment.py',E/'deployment/data.json',E/'deployment/test-joint-image-swaps.json',E/'deployment/test-joint-image-pairs.json',E/'deployment/test-joint-audio-swaps.json',ROOT/'docs/course-experiments/results/capstone_deployment.json']
registered += [E/s/'validation.json' for s in ['sft','joint','dpo']]+[E/'deployment'/f'test-{s}.json' for s in ['joint','dpo']]+[ROOT/f'docs/course-experiments/results/{s}.json' for s in ['capstone_pretrain','capstone_sft','capstone_joint','capstone_preference']]+[E/s/'train-report.json' for s in ['pretrain','sft','joint','dpo']]
(ART/'registered-input-sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in registered},indent=2)+'\n')
for key in ['validation_distributions','dpo_joint_regressions','image_pair_scores','audio_both_correct','frequency_ranges_hz','prerequisite_axes']:
 print(key.upper(),json.dumps(result[key],ensure_ascii=False))
for stage in ['sft','joint','dpo']:
 print('VALIDATION',stage,{t:(validation[stage][t]['end_to_end_correct'],validation[stage][t]['count']) for t in ['image_color','image_shape','audio','joint']})
for stage in ['joint','dpo']:print('TEST',stage,{t:(test[stage][t]['end_to_end_correct'],test[stage][t]['count']) for t in ['image_color','image_shape','audio','joint']})
print('AUDIT PASS: exact lesson, all recorded rows and token/EOS flags, frozen data, stage lineage, all 36 image pairs and 18 audio pairs; no training or GPU replication.')
