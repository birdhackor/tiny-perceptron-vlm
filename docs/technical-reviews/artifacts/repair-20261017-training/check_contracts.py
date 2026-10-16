"""Additional bounded CPU contracts; two modal steps, no long training."""
import ast
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

import torch

from scripts.course_experiments.run import experiment_spec
from tiny_perceptron.model import ModelConfig,TinyLM
from tiny_perceptron.training import load_checkpoint

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).parent
WORK=Path('/workspace/work/textbook-repair-20261017/training-technical/cpu')
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';torch.set_num_threads(2)
records=[]
def run(*args):
 cmd=[sys.executable,*map(str,args)]
 p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,env=os.environ,timeout=40)
 records.append({'command':cmd,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 if p.returncode:raise RuntimeError(records[-1])
 return p.stdout
commands=[]; experiments=[]
for lineno,line in enumerate((ROOT/'course/training.md').read_text().splitlines(),1):
 if line.startswith(('.venv/bin/python ','python scripts/')):
  words=shlex.split(line);commands.append({'line':lineno,'text':line,'script_exists':(ROOT/words[1]).is_file() if words[1]!='-m' else True})
  if '--experiment' in words:experiments.append(words[words.index('--experiment')+1])
specs={}
for name in dict.fromkeys(experiments):
 spec=experiment_spec(name); specs[name]=spec
 run('-m','scripts.course_experiments.run','--list-assets',name)
 code=ast.parse((ROOT/f"scripts/course_experiments/{spec['module']}.py").read_text())
 assert any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==spec['function'] for n in code.body)
for modality in ('vision','audio'):
 run('scripts/pretrain_encoders.py','--modality',modality,'--device','cpu','--train','--steps','1','--output',WORK/f'{modality}-encoder.pt')
modal_states={}
for task,freeze in (('vision','projector'),('audio','projector'),('joint','partial')):
 args=['scripts/train.py','--task',task,'--checkpoint',WORK/'sft.pt','--device','cpu','--freeze',freeze,'--train','--steps','2','--stop-after','1','--output',WORK/f'{task}-contract.pt']
 if task!='audio':args+=['--vision-encoder',WORK/'vision-encoder.pt']
 if task!='vision':args+=['--audio-encoder',WORK/'audio-encoder.pt']
 run(*args)
 path=WORK/f'{task}-contract.pt'; state=torch.load(path,weights_only=True);names=state['trainable_parameters']
 assert state['format_version']=='multimodal-v1' and state['step']==1 and state['optimizer'] is not None
 assert all(n.startswith(('image_projector.','audio_projector.','language.blocks.')) for n in names)
 if freeze=='projector':assert not any(n.startswith('language.') for n in names)
 run('scripts/train.py','--task',task,'--checkpoint',path,'--resume','--device','cpu','--freeze',freeze,'--train','--steps','2','--output',WORK/f'{task}-resumed.pt')
 resumed=torch.load(WORK/f'{task}-resumed.pt',weights_only=True)
 assert resumed['step']==2
 modal_states[task]={'format':state['format_version'],'step_before_resume':state['step'],'step_after_resume':resumed['step'],'trainable_parameters':names,'optimizer_present':state['optimizer'] is not None,'rng_fields':[k for k in state if k.endswith('_rng')]}
variants={}
for name,kwargs in (('baseline',{}),('rms',{'norm':'rms'}),('moe',{'experts':4,'top_k':2})):
 model=TinyLM(ModelConfig(**kwargs)); logits=model(torch.tensor([[1,73,74]]))['logits']
 logits.sum().backward()
 variants[name]={'parameters':sum(p.numel() for p in model.parameters()),'logits_shape':list(logits.shape),'first_block_norm':type(model.blocks[0].norm1).__name__,'first_block_ffn':type(model.blocks[0].ffn).__name__,'output_gradient_norm':float(model.output.weight.grad.norm())}
result={'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'},'scope':'Only two new toy modal updates per branch and three tiny forwards/backwards; experiment runner metadata listing and AST function existence, never any formal experiment execute(). Windows not executed.','commands_audited':commands,'experiment_specs':specs,'modal_checkpoint_contracts':modal_states,'architecture_fixtures':variants,'executions':records}
(OUT/'contracts-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'cli_commands_found':len(commands),'all_script_paths_exist':all(x['script_exists'] for x in commands),'fixed_experiments':list(specs),'modal_checkpoint_contracts':{k:{x:v for x,v in d.items() if x!='trainable_parameters'} for k,d in modal_states.items()},'architecture_fixtures':variants},ensure_ascii=False,indent=2))
