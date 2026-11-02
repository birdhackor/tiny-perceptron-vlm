"""Small, deterministic verifications for the frozen README, without training."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import contextlib
import gzip
import hashlib
import io
import json
import os
import platform
import re
import runpy
import subprocess
import sys

ROOT=Path('/workspace/tiny-perceptron-vlm')
WORK=Path('/workspace/work/omni-technical-readme-v1')
OUT=ROOT/'docs/technical-reviews/artifacts/omni-technical-readme-v1'
COPY=WORK/'sandbox'
sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.selftrained.model import SelftrainedConfig, LimitedAssistant

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads((ROOT/p).read_text())
result={'created_at':datetime.now(timezone.utc).isoformat(),
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'},
        'scope':'Numerical and deterministic repository facts, not retraining, unseen-input capability or a new independent audit of linked lessons.'}
headings=re.compile(r'^## ([A-Z0-9]+\.\d+) .+$',re.M)
chapter_paths=sorted((ROOT/'course/chapters').glob('*.md'))
current={h for p in chapter_paths for h in headings.findall(p.read_text())}
old=set()
for p in chapter_paths:
 try:text=subprocess.check_output(['git','-C',str(ROOT),'show','80c5db17e8b077b2c0a5e4e9baa48e760ccb0532:'+p.relative_to(ROOT).as_posix()],stderr=subprocess.DEVNULL,text=True)
 except subprocess.CalledProcessError:continue
 old.update(headings.findall(text))
front={p:headings.findall((ROOT/'course'/p).read_text()) for p in ['README.md','first-steps.md','training.md','glossary.md']}
result['section_counts']={'chapters_and_ABC':len(current),'front_numbered':sum(map(len,front.values())),
    'total':len(current)+sum(map(len,front.values())),'base80c_count':len(old),'added_ids':sorted(current-old),'front_ids':front}
assert len(current)==331 and sum(map(len,front.values()))==26 and len(current-old)==18

manifest=load('docs/selftrained/v2-manifest.json')
final=load('docs/selftrained/results/v2-final-public-results.json')
selection=load('docs/selftrained/results/v2-final-training-source-selection.json')
v2={'package':manifest['package'],'file_count':len(manifest['records'])+len(manifest['assets']),
    'file_count_derivation':f"{len(manifest['records'])} record files + {len(manifest['assets'])} assets",
    'scope':final['data_scope'],'architectures':{}}
for architecture in ['moe','dense']:
 archived=final['architectures'][architecture]
 model=LimitedAssistant(SelftrainedConfig(**archived['capacity']['config']))
 n=sum(p.numel() for p in model.parameters())
 assert n==archived['capacity']['parameters']
 metrics=load(f'docs/selftrained/results/public-raw/{architecture}/test/metrics.json')
 assert metrics['count']==3734 and sum(x['count'] for x in metrics['per_task_final_reply'].values())==3734
 v2['architectures'][architecture]={'recounted_parameters':n,'raw_test_count':metrics['count'],
    'raw_task_counts':{k:v['count'] for k,v in metrics['per_task_final_reply'].items()},
    'selected_step':selection[architecture+'_joint']['selected_step'],
    'completed_steps':selection[architecture+'_joint'].get('completed_requested_steps',selection[architecture+'_joint'].get('completed_steps')),
    'original_acceptance_passed':archived['capability_acceptance_passed'],
    'metrics_sha256':sha(ROOT/f'docs/selftrained/results/public-raw/{architecture}/test/metrics.json')}
 assert not archived['capability_acceptance_passed']
assert v2['file_count']==8962
tree=json.loads((WORK/'reference/hf-v2-tree.json').read_text())
v2['public_tree_file_count']=sum(x['type']=='file' for x in tree)
v2['public_tree_groups']=sorted(x['path'] for x in tree if x['type']=='directory')
assert v2['public_tree_file_count']==16
result['v2']=v2

models=load('docs/course-experiments/public-models.json')['models']
suffixes={'.pt','.pth','.ckpt','.safetensors'}
counts={m['id']:sum(Path(f['path']).suffix in suffixes for f in m['files']) for m in models}
caps=load('docs/course-experiments/capstone-public.json')['models']
result['public_counts']={'experiments':len(models),'weight_files':sum(counts.values()),'weights_by_experiment':counts,'capstone_models':len(caps)}
assert len(models)==30 and sum(counts.values())==120 and len(caps)==11

# Recompute the quoted historical B/C scores from pinned public original scores, not review conclusions.
raw_path=WORK/'reference/bc-predictions.jsonl.gz'
raw=gzip.decompress(raw_path.read_bytes())
archived_report=load('docs/lesson-products/results/final-report-v1.json')
assert hashlib.sha256(raw).hexdigest()==archived_report['predictions']['sha256']
b=defaultdict(lambda:Counter(n=0,correct=0));c=defaultdict(lambda:Counter(n=0,correct=0))
for line in raw.splitlines():
 row=json.loads(line)
 if row['product']=='b' and row['split']=='test':
  v=b[row['task']];v['n']+=1;v['correct']+=row['answers_by_depth']['5']==row['gold']
 if row['product']=='c':
  v=c[(row['split'],row['question'])];v['n']+=1
  predicted=row.get('selected')
  if predicted is None:
   logits=row['raw_scores'];predicted=row['candidates'][max(range(len(logits)),key=logits.__getitem__)]
  v['correct']+=predicted==row['gold']
result['bc_raw_recount']={'raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_bytes':len(raw),
    'gzip_sha256':sha(raw_path),'b_five_loops':dict(b),
    'c':{split+':'+question:dict(v) for (split,question),v in c.items()},
    'scope':'Recount of original frozen predictions; no new model predictions, training or selection.'}
assert b['boxes']==Counter(n=2956,correct=2928) and b['sum']==Counter(n=42,correct=2)
assert c[('test','choice')]==Counter(n=11797,correct=11797)
assert c[('wording_shift','choice')]==Counter(n=11797,correct=11493)
freeze=load('docs/lesson-products/results/settings-freeze-v1.json')
started=load('docs/lesson-products/results/final-report-v1.started.json')
assert started['settings_freeze_sha256']==sha(ROOT/'docs/lesson-products/results/settings-freeze-v1.json')
assert freeze['recorded_at']<started['recorded_at']<archived_report['recorded_at']
result['bc_timing']={k:v for k,v in [('settings_recorded_at',freeze['recorded_at']),('test_started_at',started['recorded_at']),('report_recorded_at',archived_report['recorded_at'])]}

# Instrument a second evaluation of the actual README dry-run to verify dimensions and absence of updates.
import scripts.train as train
captured={};instance=[]
real_init=train.TinyLM.__init__
real_forward=train.TinyLM.forward
def init(self,*args,**kwargs):
 real_init(self,*args,**kwargs);instance.append((self,{k:v.detach().clone() for k,v in self.state_dict().items()}))
def forward(self,ids=None,**kwargs):
 value=real_forward(self,ids,**kwargs)
 captured.update(input_shape=list(ids.shape),input_dtype=str(ids.dtype),
    logits_shape=list(value['logits'].shape),valid_shape=list(kwargs['valid'].shape),valid_tokens=int(kwargs['valid'].sum()))
 return value
real_step=torch.optim.AdamW.step
real_save=train.save_checkpoint
def forbidden(*args,**kwargs):raise AssertionError('dry-run unexpectedly updated or saved weights')
train.TinyLM.__init__=init;train.TinyLM.forward=forward
torch.optim.AdamW.step=forbidden;train.save_checkpoint=forbidden
args=['scripts/train.py','--task','text','--data','data/generated/toy-text/train.jsonl']
before=sys.argv[:];cwd=Path.cwd();stream=io.StringIO()
try:
 os.chdir(COPY);sys.argv=args
 with contextlib.redirect_stdout(stream):train.main()
finally:
 sys.argv=before;os.chdir(cwd)
 train.TinyLM.__init__=real_init;train.TinyLM.forward=real_forward
 torch.optim.AdamW.step=real_step;train.save_checkpoint=real_save
report=json.loads(stream.getvalue().splitlines()[-1]);model,initial=instance[0]
captured.update(mode=report['mode'],history=report['history'],
   parameters_unchanged=all(torch.equal(initial[k],v) for k,v in model.state_dict().items()),
   optimizer_step_calls=0,checkpoint_save_calls=0,
   generated_train_manifest=json.loads((COPY/'data/generated/toy-text/manifest.json').read_text()))
assert captured['parameters_unchanged'] and report['mode']=='dry-run-no-weight-update'
assert report['history'][0]['loss']==5.770333290100098 and report['history'][0]['grad_norm']==0.9595053195953369
result['instrumented_readme_dry_run']=captured

(OUT/'facts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
