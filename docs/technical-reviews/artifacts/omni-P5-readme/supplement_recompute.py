"""Pure preserved-data checks and tensor-shape counting on the meta device; zero neural forward."""
import collections,datetime,hashlib,json,math,tarfile
from pathlib import Path
import torch
from tiny_perceptron.selftrained.model import LimitedAssistant,SelftrainedConfig
ROOT=Path('/workspace/work/omni-book-integrated-v1');OUT=ROOT/'docs/technical-reviews/artifacts/omni-P5-readme';RAW=Path('/workspace/work/omni-P5-readme-independent-raw');FINAL=Path('/workspace/work/omni-final-completed-41004229262/verified')
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
p={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_readme','torch':str(torch.__version__),'scope':'No neural forward, checkpoint/tensor load, optimizer update or new model inference. Count parameter shapes on meta; stream existing local data tar; recompute saved waveform-analysis event conditions.'}
# Exact model shapes, no weights allocation/loading and no call to forward().
summary=read(ROOT/'docs/selftrained/results/v2-final-public-results.json');p['parameter_count_meta']={}
for arch in ['moe','dense']:
 config=summary['architectures'][arch]['capacity']['config']
 with torch.device('meta'):model=LimitedAssistant(SelftrainedConfig(**config))
 n=sum(x.numel() for x in model.parameters());p['parameter_count_meta'][arch]={'config':config,'unique_parameter_numel':n,'all_parameters_meta':all(x.device.type=='meta' for x in model.parameters()),'all_buffers_meta':all(x.device.type=='meta' for x in model.buffers()),'neural_forward_calls':0,'saved_capacity_count':summary['architectures'][arch]['capacity']['parameters']}
 assert n==({'moe':5447107,'dense':2288067}[arch])
 del model
# Verify complete existing package bytes plus every member digest and actual row split.
m=read(ROOT/'docs/selftrained/v2-manifest.json');archive=ROOT/m['package']['path'];expected={r['path']:r for r in m['records']+m['assets']};seen=set();by_split=collections.Counter();by_task=collections.Counter();file_rows={};bad=[]
with tarfile.open(archive,'r:gz') as tf:
 for member in tf:
  if not member.isfile():continue
  seen.add(member.name);b=tf.extractfile(member).read();e=expected.get(member.name)
  if e is None or len(b)!=e['bytes'] or hashlib.sha256(b).hexdigest()!=e['sha256']:bad.append(member.name)
  if member.name.endswith('.jsonl'):
   rows=[json.loads(l) for l in b.splitlines() if l.strip()];file_rows[member.name]=len(rows)
   for r in rows:by_split[r['split']]+=1;by_task[(r['split'],r['task'])]+=1
p['v2_data_package']={'archive':digest(archive),'manifest':digest(ROOT/'docs/selftrained/v2-manifest.json'),'expected_files':len(expected),'observed_files':len(seen),'member_mismatches':bad,'missing_files':sorted(set(expected)-seen),'unexpected_files':sorted(seen-set(expected)),'row_splits':dict(by_split),'file_rows':file_rows,'task_split_counts':{':'.join(k):v for k,v in by_task.items()}}
assert not bad and seen==set(expected) and dict(by_split)=={'train':28876,'validation':2435,'test':3734}
# Recompute waveform event acceptance from saved measured events and program, not stored success booleans.
s=read(FINAL/'measurements/sound/results.json')['codec_and_gold_program'];decisions=[]
for row in s['rows']:
 g=row['generation'];a=g['analysis']['measured'];expected=[];pos=0
 for e in row['program']:expected.append({'frequency':e['frequency'],'onset_ms':pos,'offset_ms':pos+e['duration_ms']});pos+=e['duration_ms']+e['rest_ms']
 ok=bool(g['eos'] and not g['capped'] and a['legal'] and len(a['events'])==len(expected))
 if ok:
  for actual,target in zip(a['events'],expected):
   ok &= abs(actual['frequency']-target['frequency'])/target['frequency']<=.05
   ok &= abs(actual['onset_ms']-target['onset_ms'])<=30
   ok &= abs(actual['offset_ms']-target['offset_ms'])<=30
   ok &= abs((actual['offset_ms']-actual['onset_ms'])-(target['offset_ms']-target['onset_ms']))<=30
  for actual,nxt,target,nt in zip(a['events'],a['events'][1:],expected,expected[1:]):ok &= abs((nxt['onset_ms']-actual['offset_ms'])-(nt['onset_ms']-target['offset_ms']))<=30
  ok &= abs(a['samples']/8-pos)<=30
 decisions.append({'family_index':row['family_index'],'recomputed_success':bool(ok),'saved_success':g['program_success']})
p['sound_event_recheck']={'families':len(decisions),'recomputed_success':sum(x['recomputed_success'] for x in decisions),'all_saved_agree':all(x['recomputed_success']==x['saved_success'] for x in decisions),'criteria':'Frequency relative tolerance5%; event onset/offset/duration/inter-event rest/total waveform duration each <=30ms; all events required, EOS true and not capped. Inputs are saved original analysis values, not reanalyzed PCM16 or new neural output.','rows':decisions}
assert p['sound_event_recheck']['recomputed_success']==200 and p['sound_event_recheck']['all_saved_agree']
# Exact producer source versions used by current raw measurements.
x=read(FINAL/'execution-sources.json');p['final_evaluator_versions']={'raw_binding':digest(FINAL/'execution-sources.json'),'revision':x['revision'],'inspected_files':{}}
for name in ['scripts/omni_revision/core_evaluation.py','scripts/omni_revision/core_final.py','tiny_perceptron/omni_products/reliability.py','tiny_perceptron/omni_products/data.py','tiny_perceptron/omni_products/signal_metrics.py']:
 p['final_evaluator_versions']['inspected_files'][name]={'original_sha256':x['files'][name],'current_sha256':digest(ROOT/name)['sha256']};assert x['files'][name]==digest(ROOT/name)['sha256']
p['v2_training']={}
for arch,name in [('moe','moe-native'),('dense','dense-weighted')]:
 base=ROOT/'docs/selftrained/results/training-raw'/name;t=read(base/'raw/train-receipt.json');e=read(base/'raw/execution.json');r=read(base/'receipt.json');manifest=read(RAW/('v2-'+arch+'-inference.raw'))
 p['v2_training'][arch]={'train_receipt':digest(base/'raw/train-receipt.json'),'execution':digest(base/'raw/execution.json'),'producer_receipt':digest(base/'receipt.json'),'steps':t['steps'],'completed_requested_steps':t['completed_requested_steps'],'interrupted':t['interrupted'],'total_parameters':t['total_parameters'],'origin_kind':t['origin']['kind'],'test_used_for_selection':t['test_used_for_selection'],'selection':t['selection'],'inference_selected_step':manifest['selected_step'],'selected_checkpoint_sha256':manifest['selected_checkpoint_sha256'],'inference_manifest':digest(RAW/('v2-'+arch+'-inference.raw')),'command':e['command'],'returncode':e['returncode'],'run_id':e['run_id'],'revision':e['revision']}
 assert t['steps']==({'moe':4000,'dense':10000}[arch]) and t['completed_requested_steps'] and not t['interrupted'] and manifest['selected_step']==1000
p['public_old_bc_training']={}
for product in ['b','c']:
 q=RAW/('old-'+product+'-training.raw');r=read(q)
 p['public_old_bc_training'][product]={'public_receipt':digest(q),'environment':r['environment'],'random_initialization':r['random_initialization'],'resume_source':r['resume_source'],'steps_this_attempt':r['steps_this_attempt'],'selected_step':r['selected_step'],'initial_weights_sha256':r['initial_weights_sha256'],'selected_weights_sha256':r['selected_weights_sha256'],'weights_changed':r['weights_changed'],'parameters':r['parameters']}
p['repository_sources']=[digest(ROOT/q) for q in ['tiny_perceptron/selftrained/model.py','tiny_perceptron/model.py','tiny_perceptron/modern.py','tiny_perceptron/selftrained/tokenizer.py','tiny_perceptron/omni_products/signal_metrics.py']]
with (OUT/'supplement-recompute-v1.json').open('x') as f:json.dump(p,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'created_at':p['created_at'],'parameter_counts':{a:x['unique_parameter_numel'] for a,x in p['parameter_count_meta'].items()},'data_package':{k:v for k,v in p['v2_data_package'].items() if k not in ['task_split_counts','file_rows']},'sound_event_successes':p['sound_event_recheck']['recomputed_success'],'v2_completed_steps':{a:x['steps'] for a,x in p['v2_training'].items()}},ensure_ascii=False,indent=2))
