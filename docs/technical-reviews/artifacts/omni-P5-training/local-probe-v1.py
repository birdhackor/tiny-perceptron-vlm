"""Reviewer-owned bounded CPU checks; no training updates or fixed final-model inference."""
from pathlib import Path
import argparse, ast, copy, datetime, hashlib, json, math, os, platform, re, shlex, subprocess, sys, tempfile
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tiny_perceptron.data import ByteTokenizer, render_chat, split_documents, toy_documents
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM
from scripts.prepare_data import generate_records
OUT=Path(__file__).parent/'local-probe-results-v1.json'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def run(arguments,cwd=ROOT):
 env=dict(os.environ,PYTHONPATH=str(ROOT));p=subprocess.run([sys.executable,*arguments],cwd=cwd,env=env,text=True,capture_output=True,timeout=45)
 if p.returncode: raise AssertionError((arguments,p.returncode,p.stderr[-2000:]))
 return p.stdout
body=(ROOT/'course/training.md').read_text()
report={'created_at':datetime.datetime.now(datetime.UTC).isoformat(),'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'CPU','platform':platform.platform()},'scope':'No optimizer update, no original trained model inference, no workflow/Modal/publish. All created model fixtures are untrained artificial models, temporary and outside Git.','source_sha256':sha(body.encode())}
# Check every fenced shell command's option names against the current entrypoint help.
entries=['scripts/train.py','scripts/train_simple.py','scripts/prepare_data.py','scripts/pretrain_encoders.py','scripts/evaluate.py','scripts/evaluate_modal.py','scripts/infer.py','scripts/infer_modal.py','scripts/quantize.py','scripts/fetch_training_assets.py','scripts/fetch_course_models.py','-m scripts.course_experiments.run','-m scripts.omni_revision.public_download','-m scripts.omni_revision.public_demo']
helps={entry:run(shlex.split(entry)+['--help']) for entry in entries}
report['entrypoint_help']={entry:{'command':f'{sys.executable} {entry} --help','exit':0,'help_sha256':sha(text.encode()),'options':sorted(set(re.findall(r'--[a-z][a-z-]*',text)))} for entry,text in helps.items()}
commands=[]
for block in re.findall(r'```bash\n(.*?)```',body,re.S):
 for line in block.strip().splitlines():
  if '.venv/bin/python ' in line: rest=line.split('.venv/bin/python ',1)[1]
  elif line.startswith('uv run --extra cpu python '): rest=line.split('uv run --extra cpu python ',1)[1]
  else: continue
  parts=shlex.split(rest);entry=' '.join(parts[:2]) if parts[0]=='-m' else parts[0]
  opts=[p.split('=',1)[0] for p in parts if p.startswith('--')]
  missing=[p for p in opts if p not in report['entrypoint_help'][entry]['options']]
  assert not missing,(line,missing)
  commands.append({'command':line,'entrypoint':entry,'options_checked':opts,'unknown_options':missing,'execution':'syntax/help/code inspected; training/evaluation/download command not executed as textbook run'})
report['fenced_commands']=commands
report['command_count']=len(commands)
# Exercise the exact permitted dry runs. Capture full two-line train output.
report['dry_runs']={}
for task in ('text','sft','vision'):
 text=run(['scripts/train.py','--task',task,'--device','cpu']);lines=text.strip().splitlines();v=json.loads(lines[-1]);h=v['history'][0]
 assert len(lines)==2 and v['mode']=='dry-run-no-weight-update' and len(v['history'])==1
 assert math.isfinite(h['loss']) and math.isfinite(h['grad_norm']) and h['grad_norm']>0
 assert (h['effective_tokens'] is None) if task=='vision' else (h['effective_tokens']>0)
 report['dry_runs'][task]={'stdout':text,'exit':0,'weight_updates':0,'code_basis':'train.py optimizer.step and saves both guarded by args.train'}
for model in ('bigram','mlp'):
 args=['scripts/train_simple.py','--model',model,'--seed','42','--device','cpu']
 if model=='mlp':args+=['--context','3','--width','16']
 report['dry_runs'][model]=json.loads(run(args))
for modality in ('vision','audio'):
 report['dry_runs']['encoder-'+modality]=json.loads(run(['scripts/pretrain_encoders.py','--modality',modality,'--device','cpu']))
 assert report['dry_runs']['encoder-'+modality]['holdout_examples']==2
# Data grouping and labels, generated in an outside-Git temporary directory.
with tempfile.TemporaryDirectory(prefix='P5-training-local-') as temporary:
 tmp=Path(temporary);report['generated_data']={}
 for kind in ('toy-text','attributes-sft','style','safety','preference'):
  v=json.loads(run(['scripts/prepare_data.py','--kind',kind,'--seed','42','--output',str(tmp/'data')]))
  grouped={}
  for split in ('train','validation','test'):
   rows=[json.loads(line) for line in (tmp/'data'/kind/(split+'.jsonl')).read_text().splitlines()]
   grouped[split]=sorted({row['family'] for row in rows})
  assert not any(set(grouped[a])&set(grouped[b]) for a,b in (('train','validation'),('train','test'),('validation','test')))
  report['generated_data'][kind]={'manifest':v,'families':grouped}
 assert [report['generated_data']['toy-text']['manifest']['splits'][s]['records'] for s in ('train','validation','test')]==[9,1,2]
 assert [report['generated_data']['attributes-sft']['manifest']['splits'][s]['records'] for s in ('train','validation','test')]==[45,5,10]
 assert 'blue:square:low' in report['generated_data']['attributes-sft']['families']['train']
 # Execute the complete exact inline start snippet in a temporary working directory.
 snippets=re.findall(r'```python\n(.*?)```',body,re.S);start,size=snippets
 (tmp/'save_start.py').write_text(start);text=run([str(tmp/'save_start.py')],cwd=tmp)
 payload=torch.load(tmp/'checkpoints/start.pt',map_location='cpu',weights_only=True)
 report['inline_start']={'snippet_sha256':sha(start.encode()),'stdout':text,'config':payload['config'],'step':payload['step'],'optimizer':payload['optimizer'],'model_tensor_bytes':sum(t.numel()*t.element_size() for t in payload['model'].values()),'contains_rng':all(k in payload for k in ('torch_rng','python_rng'))}
 assert payload['optimizer'] is None and payload['step']==0
 # Quantization and exact inline size script use this untrained artificial fixture.
 (tmp/'checkpoints/attributes.pt').write_bytes((tmp/'checkpoints/start.pt').read_bytes())
 report['artificial_quantization']={}
 for bits in (4,8):
  text=run([str(ROOT/'scripts/quantize.py'),str(tmp/'checkpoints/attributes.pt'),'--bits',str(bits),'--output',str(tmp/f'checkpoints/attributes-int{bits}.pt')],cwd=tmp)
  report['artificial_quantization'][str(bits)]=json.loads(text)
 (tmp/'size_check.py').write_text(size);text=run([str(tmp/'size_check.py')],cwd=tmp)
 assert len(text.strip().splitlines())==3
 report['inline_size']={'snippet_sha256':sha(size.encode()),'stdout':text,'fixture':'untrained width32/layer1/head1/max128 only; values are not textbook trained-model results'}
 # Strict answer metrics and literal byte examples; no trained model loaded.
 from scripts.evaluate import answer_sample, evaluate
 tok=ByteTokenizer();report['byte_examples']={'A':tok.encode('A'),'EOS':tok.eos_id,'han':tok.encode('中'),'han_bytes':list('中'.encode()),'A_EOS_report':answer_sample(tok,[73,2],'A'),'illegal_control':answer_sample(tok,[73,3,2],'A'),'space':answer_sample(tok,tok.encode(' A')+[2],'A')}
 assert tok.encode('A')==[73] and tok.eos_id==2 and len(tok.encode('中'))==3
 assert report['byte_examples']['A_EOS_report']['completed_exact_match']
 assert not report['byte_examples']['illegal_control']['exact_match'] and not report['byte_examples']['space']['exact_match']
 model=TinyLM(ModelConfig(width=8,layers=1,heads=1,max_length=64))
 v=evaluate(model,[{'text':'AB'}],mode='text',max_new_tokens=2)
 report['artificial_text_evaluate']={'effective_tokens':v['effective_tokens'],'exact_match':v['exact_match'],'completed_exact_match':v['completed_exact_match'],'metric_denominators':v['metric_denominators'],'sample':v['samples'][0]}
 assert v['exact_match'] is None and v['effective_tokens']==3
 x,y=render_chat([{'role':'user','content':'shape?'},{'role':'assistant','content':'circle'}]);report['supervised_sft_targets']=y[y!=-100].tolist();assert report['supervised_sft_targets']==tok.encode('circle')+[2]
# Demonstrate CLI freeze masks on artificial modules, without updates.
report['freeze_masks']={}
for mode in ('none','projector','partial'):
 model=MultiModalLM(TinyLM())
 if mode!='none':
  model.requires_grad_(False);model.image_projector.requires_grad_(True);model.audio_projector.requires_grad_(True)
  if mode=='partial':model.language.blocks[0].requires_grad_(True);model.language.blocks[-1].requires_grad_(True)
 names=[name for name,p in model.named_parameters() if p.requires_grad]
 report['freeze_masks'][mode]={'trainable_names':names,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
report['simple_document_split']={k:len(v) for k,v in split_documents(toy_documents(),42).items()}
# Independently reconstruct historical input snapshots only, never run tools/model.
tools_path=ROOT/'docs/course-experiments/results/tools.json';data=json.loads(tools_path.read_bytes());checks=[]
def encode(messages):
 ids=[1]
 for m in messages:ids += [{'system':7,'user':3,'assistant':4}[m['role']]]+[b+8 for b in m['content'].encode()]+[2]
 return ids+[4]
for key in ('samples','one_step_cap_samples'):
 for ei,episode in enumerate(data['results'][key]):
  first=next(e['generation'] for e in episode['trace'] if 'generation' in e);current=copy.deepcopy(first['messages'][:2])
  for ti,event in enumerate(episode['trace']):
   if 'generation' not in event:continue
   g=event['generation'];assert encode(current)==g['input_ids'];assert len(encode(current))==g['input_tokens']
   checks.append({'pointer':f'/results/{key}/{ei}/trace/{ti}/generation','input_ids_sha256':sha(json.dumps(g['input_ids']).encode()),'message_snapshot_matches':g['messages']==current})
   if event.get('executed') and event.get('finite_result'):current += [{'role':'assistant','content':g['samples'][0]['generated']},{'role':'user','content':f"TOOL_RESULT:{event['tool_result']}"}]
report['historical_tool_inputs']={'source_sha256':sha(tools_path.read_bytes()),'generation_steps':len(checks),'metadata_mismatches':sum(not r['message_snapshot_matches'] for r in checks),'rows':checks}
assert len(checks)==56 and report['historical_tool_inputs']['metadata_mismatches']==36
report['hand_examples']={'T5_addition':2+3,'T8_mean_difference':0.6-0.8,'T11_size_decrease':60000<80000,'T11_accuracy_preserved':3/5>=4/5,'T11_MiB':2192940/(1024**2)}
report['unverified']=['Native Windows T.11 execution awaits book PR CI.','No Colab/MPS/GPU/hardware timing execution.','No training schedule or fixed final-family neural inference repeated.','No full trained textbook route recreated; measured facts are independently recomputed from preserved records in separate evidence.']
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(OUT.relative_to(ROOT)),'bytes':OUT.stat().st_size,'sha256':sha(OUT.read_bytes()),'commands_checked':len(commands),'dry_runs':len(report['dry_runs']),'status':'completed'}))
