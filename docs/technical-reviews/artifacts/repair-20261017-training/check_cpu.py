"""Bounded CPU review, one optimizer update per branch; never a formal rerun."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.prepare_data import generate_records
from scripts.train import parser, prepare_examples
from tiny_perceptron.data import ByteTokenizer, render_chat, split_documents, toy_documents
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM
from tiny_perceptron.tokenization import generation_report
from tiny_perceptron.training import save_checkpoint, seed_everything

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
WORK = Path('/workspace/work/textbook-repair-20261017/training-technical/cpu')
WORK.mkdir(parents=True, exist_ok=True)
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
torch.set_num_threads(2)
environment = {'python':platform.python_version(),'torch':str(torch.__version__),'platform':platform.platform(),'device':'cpu','cuda_available':str(torch.cuda.is_available())}
entries=[]

def run(*args):
 command=[sys.executable,*map(str,args)]
 result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=50,env=os.environ)
 record={'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
 entries.append(record)
 if result.returncode: raise RuntimeError(record)
 return result.stdout

for script in ('train','train_simple','prepare_data','pretrain_encoders','evaluate','infer','infer_modal','quantize','evaluate_modal','prepare_ocr'):
 run(f'scripts/{script}.py','--help')
run('scripts/fetch_training_assets.py','--list')
run('scripts/fetch_course_models.py','--list')
for task in ('text','sft','vision','audio','joint'):
 run('scripts/train.py','--task',task,'--device','cpu','--output',WORK/f'dry-{task}.pt')
 assert not (WORK/f'dry-{task}.pt').exists()
for kind in ('toy-text','attributes-sft','style','safety','preference'):
 run('scripts/prepare_data.py','--kind',kind,'--seed','42','--output',WORK/'generated')
seed_everything(42)
start=TinyLM(ModelConfig(width=32,layers=1,heads=1,max_length=128))
save_checkpoint(WORK/'start.pt',start,step=0,metadata={'seed':42,'note':'untrained review fixture'})
before={k:v.clone() for k,v in start.state_dict().items()}
for task,kind in (('text','toy-text'),('sft','attributes-sft')):
 run('scripts/train.py','--task',task,'--data',WORK/f'generated/{kind}/train.jsonl','--checkpoint',WORK/'start.pt','--device','cpu','--seed','42','--train','--steps','1','--output',WORK/f'{task}.pt')
 updated=torch.load(WORK/f'{task}.pt',weights_only=True)
 assert updated['step']==1 and updated['optimizer'] is not None
 assert any(not torch.equal(before[k],v) for k,v in updated['model'].items())
 for stage,checkpoint in (('before','start.pt'),('after',f'{task}.pt')):
  run('scripts/evaluate.py',WORK/checkpoint,'--data',WORK/f'generated/{kind}/validation.jsonl','--mode',task,'--tokens','2','--limit','all','--device','cpu','--output',WORK/f'{task}-{stage}.json')
for kind in ('bigram','mlp'):
 for trained in (False,True):
  args=['scripts/train_simple.py','--model',kind,'--seed','42','--device','cpu','--steps','1','--output',WORK/f'{kind}.pt']
  if trained: args+=['--train']
  run(*args)
for modality in ('vision','audio'):
 run('scripts/pretrain_encoders.py','--modality',modality,'--device','cpu','--steps','1')
run('scripts/train.py','--task','dpo','--checkpoint',WORK/'start.pt','--data',WORK/'generated/preference/train.jsonl','--train','--steps','1','--device','cpu','--output',WORK/'dpo.pt')
for bits in (4,8):
 run('scripts/quantize.py',WORK/'sft.pt','--bits',bits,'--output',WORK/f'packed{bits}.pt')
 run('scripts/evaluate.py',WORK/f'packed{bits}.pt','--data',WORK/'generated/attributes-sft/validation.jsonl','--mode','sft','--tokens','2','--limit','all','--device','cpu','--output',WORK/f'packed{bits}-evaluation.json')
 run('scripts/infer.py',WORK/f'packed{bits}.pt','--chat','--prompt','color=blue;shape=circle;pitch=low;color?','--tokens','2','--device','cpu','--json')
modal=MultiModalLM(TinyLM(ModelConfig(width=32,layers=1,heads=1,max_length=128)))
save_checkpoint(WORK/'modal.pt',modal,metadata={'task':'joint'})
run('scripts/infer_modal.py',WORK/'modal.pt','--color','red','--shape','square','--frequency','220','--prompt','joint?','--tokens','2','--device','cpu')
summary={
 'environment':environment,
 'scope':'CPU dry-runs and one-step review fixtures only; no official experiment or existing trained product rerun; generation budget 2 deliberately tests format rather than capability.',
 'toy_document_splits':{k:len(v) for k,v in split_documents(toy_documents(),42).items()},
 'generated_splits':{kind:json.loads((WORK/f'generated/{kind}/manifest.json').read_text()) for kind in ('toy-text','attributes-sft','style','safety','preference')},
 'byte_A_EOS':generation_report(ByteTokenizer(),[73,2]),
 'answer_only_labels':render_chat(generate_records('attributes-sft')[0]['messages'])[1].tolist(),
 'tensor_bytes':{},
 'evaluations':{},
 'executions':entries,
}
for name in ('sft.pt','packed4.pt','packed8.pt'):
 path=WORK/name; state=torch.load(path,map_location='cpu',weights_only=True)['model']
 summary['tensor_bytes'][name]={'model_tensor_bytes':sum(t.numel()*t.element_size() for t in state.values()),'file_bytes':path.stat().st_size,'tensor_dtypes':sorted(set(str(t.dtype) for t in state.values()))}
for path in WORK.glob('*-*.json'):
 summary['evaluations'][path.name]=json.loads(path.read_text())
(OUT/'cpu-check.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('executions','evaluations','answer_only_labels','generated_splits')},ensure_ascii=False,indent=2))
print(f'Completed {len(entries)} bounded CPU CLI calls; persisted complete commands/results in cpu-check.json.')
