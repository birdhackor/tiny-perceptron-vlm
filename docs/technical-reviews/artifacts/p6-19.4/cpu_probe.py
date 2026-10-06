from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from scripts.selftrained import train, train_local_stage
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer, OCR_CHARACTERS
from safetensors import safe_open

torch.set_num_threads(2)
out = {'environment': {'python':sys.version.split()[0], 'torch':torch.__version__, 'device':'cpu', 'platform':platform.platform()}, 'purpose':'Synthetic bounded software checks; no original data, full training, GPU, heldout re-evaluation, or model upload.'}
fixture_dir=Path(tempfile.mkdtemp(prefix='p6-fact-19-4-'))
rows=[]
for split in ['train','validation']:
    for index,task in enumerate(['text','text','tool_call','vision_clothing','ocr','voice_qa','voice_topic_continuation']):
        rows.append({'id':f'{split}-{index}', 'group_id':f'{split}-{index}', 'split':split, 'task':task, 'messages':[{'role':'user','content':'-12x?'},{'role':'assistant','content':'-12x'}]})
record_path=fixture_dir/'records.jsonl'
record_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
config={'width':16,'layers':1,'heads':2,'kv_heads':1,'ffn_hidden':32,'experts':2,'top_k':1,'max_length':512}
manifest={'schema_version':1, 'initialization':'random','model_config':config,'records':[{'path':'records.jsonl','bytes':record_path.stat().st_size,'sha256':hashlib.sha256(record_path.read_bytes()).hexdigest()}],'assets':[]}
manifest_path=fixture_dir/'manifest.json';manifest_path.write_text(json.dumps(manifest))
manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
(BASE/'probe-fixture-records.jsonl').write_bytes(record_path.read_bytes())
(BASE/'probe-fixture-manifest.json').write_bytes(manifest_path.read_bytes())
commands=[]

def invoke(name,stage,steps,source=None,resume=False,weights=None):
    target=fixture_dir/name
    args=[sys.executable,str(ROOT/'scripts/selftrained/train_local_stage.py'),'--manifest',str(manifest_path),'--manifest-sha256',manifest_sha,'--data-root',str(fixture_dir),'--','--output-dir',str(target),'--stage',stage,'--architecture','moe','--steps',str(steps),'--batch-size',str(16 if weights else 2),'--context','512','--seed','20261006','--learning-rate',str(.0002 if stage=='joint' else .001),'--eval-every','1','--save-every','1','--threads','2','--device','cpu']
    if source:
        args += ['--resume' if resume else '--init-checkpoint',str(source)]
    if stage=='joint':
        args+=['--freeze-perception-backbones','--sampling-mode','task-family']
    if weights:
        args+=['--tool-loss-weight',str(weights[0]),'--numeric-run-loss-weight',str(weights[1]),'--native-voice-loss-weight',str(weights[2])]
    completed=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=45)
    (BASE/f'probe-{name}.log').write_text(completed.stdout+completed.stderr)
    commands.append({'name':name,'command':shlex.join(args),'exit_code':completed.returncode})
    assert completed.returncode==0, (name,completed.stdout[-3000:],completed.stderr)
    e=json.loads((target/'execution.json').read_text());r=json.loads((target/'receipt.json').read_text())
    assert e['status']=='completed' and e['returncode']==0 and all(r[k]==v for k,v in e.items())
    return target

full=invoke('pretrain-full','pretrain',3)
prefix=invoke('pretrain-prefix','pretrain',1)
resumed=invoke('pretrain-resumed','pretrain',3,prefix/'latest.pt',resume=True)
f=torch.load(full/'latest.pt',weights_only=False,map_location='cpu')
p=torch.load(prefix/'latest.pt',weights_only=False,map_location='cpu')
r=torch.load(resumed/'latest.pt',weights_only=False,map_location='cpu')
assert f['step']==r['step']==3 and f['tokens']==r['tokens'] and f['target_tokens']==r['target_tokens']
assert f['sampler']['draws']==r['sampler']['draws']==6
assert all(torch.equal(f['model'][k],r['model'][k]) for k in f['model'])
assert torch.equal(f['rng']['torch'],r['rng']['torch'])
assert torch.equal(f['sampler']['generator'],r['sampler']['generator'])
assert all(torch.equal(f['optimizer']['state'][k][field],r['optimizer']['state'][k][field]) for k in f['optimizer']['state'] for field in f['optimizer']['state'][k])
out['exact_resume']={'completed_steps':r['step'],'sample_draws':r['sampler']['draws'],'tokens':r['tokens'],'target_tokens':r['target_tokens'],'model_tensors_exact_equal':True,'optimizer_tensors_exact_equal':True,'rng_exact_equal':True,'sampler_exact_equal':True}
sft=invoke('sft-fresh','sft',1,prefix/'best.pt')
s=torch.load(sft/'latest.pt',weights_only=False,map_location='cpu')
assert s['step']==1 and s['sampler']['draws']==2
assert all(float(state['step'])==1 for state in s['optimizer']['state'].values())
assert s['stage_history'][-1]['checkpoint_sha256']==hashlib.sha256((prefix/'best.pt').read_bytes()).hexdigest()
out['fresh_stage']={'completed_steps':s['step'],'sample_draws':s['sampler']['draws'],'all_optimizer_steps':1,'history_parent_sha_matched':True}
joint=invoke('joint-baseline','joint',1,sft/'best.pt')
weighted=invoke('weighted44','joint',1,joint/'best.pt',weights=(4,4,1))
native=invoke('native414','joint',1,weighted/'best.pt',weights=(4,1,4))
n=torch.load(native/'latest.pt',weights_only=False,map_location='cpu')
out['native_local_integration']={'source_completed_steps':n['origin']['new_joint_initialization']['source_completed_steps'],'native_step':n['step'],'source_kind':n['origin']['new_joint_initialization']['source_checkpoint_kind'],'source_integrity':n['origin']['new_joint_initialization']['source_integrity'],'meaning':'Only validates the executable own-source and state contract on synthetic text-only fixtures; these rows do not demonstrate perception or answering quality.'}

def rejects(label,call):
    try: call()
    except (ValueError,FileNotFoundError) as e: return {'case':label,'rejected':True,'exception':str(e)}
    raise AssertionError(label+' was unexpectedly admitted')
w=torch.load(weighted/'latest.pt',weights_only=False,map_location='cpu')
out['negative_cases']=[
    rejects('resume changed 4/4/1 to 4/1/4',lambda:train.validate_resume_language_objective(w,{**w['training_options'],'tool_loss_weight':4,'numeric_run_loss_weight':1,'native_voice_loss_weight':4})),
    rejects('fresh init uses latest.pt',lambda:train_local_stage.local_source(weighted/'latest.pt',manifest_sha,False)),
    rejects('resume uses best.pt',lambda:train_local_stage.local_source(weighted/'best.pt',manifest_sha,True)),
]
execution_path=weighted/'execution.json';outer_path=weighted/'receipt.json'
original_execution=execution_path.read_bytes();original_outer=outer_path.read_bytes()
e=json.loads(original_execution);e['status']='failed';e['returncode']=1;execution_path.write_text(json.dumps(e))
o=json.loads(original_outer);o.update(e);outer_path.write_text(json.dumps(o))
out['negative_cases'].append(rejects('incomplete weighted fresh init',lambda:train_local_stage.local_source(weighted/'best.pt',manifest_sha,False)))
execution_path.write_bytes(original_execution);outer_path.write_bytes(original_outer)
with safe_open(str(native/'model.safetensors'),framework='pt',device='cpu') as safe:
    keys=list(safe.keys());assert set(keys)==set(n['model'])
    assert not any(k in keys for k in ['optimizer','rng','sampler'])
out['safe_export']={'model_tensor_count':len(keys),'contains_optimizer_rng_sampler':False}

model=LimitedAssistant(SelftrainedConfig(vocab_size=128,**config))
selection={}
for stage in train.STAGES:
    train.set_trainable(model,stage,stage=='joint')
    names=[name for name,p in model.named_parameters() if p.requires_grad]
    selection[stage]={'trainable_module_roots':sorted(set(name.split('.')[0] for name in names)), 'parameter_tensors':len(names)}
    if stage in ['vision','ocr','audio']: assert selection[stage]['trainable_module_roots']==[stage+'_encoder']
train.set_trainable(model,'joint',True)
frozen=[name for name,p in model.named_parameters() if not p.requires_grad]
assert all(any(name.startswith(prefix) for prefix in ['lm.',*train.PERCEPTION_BRIDGE_PREFIXES]) for name,p in model.named_parameters() if p.requires_grad)
assert all(not p.requires_grad for name,p in model.named_parameters() if '.head.' in name)
out['trainable_modules']=selection
out['perception_heads']={'vision_classes':model.vision_encoder.head.out_features,'ocr_known_characters':len(OCR_CHARACTERS),'ocr_classes_including_blank':model.ocr_encoder.head.out_features,'audio_intent_classes':model.audio_encoder.head.out_features}
tok=CharacterTokenizer.build(['-12x'])
labels=torch.tensor([[tok.character_ids[c] for c in '-12x']+[-100]]*4)
test_rows=[{'task':'tool_call','split':'train'},{'task':'voice_qa','split':'train'},{'task':'voice_qa','split':'train','augmentation':{}},{'task':'text','split':'train'}]
weighted_weights=train.language_loss_weights(labels,test_rows,tok,4,4,1)
native_weights=train.language_loss_weights(labels,test_rows,tok,4,1,4)
assert weighted_weights.tolist()==[[16,16,16,16,0],[4,4,4,4,0],[4,4,4,4,0],[4,4,4,4,0]]
assert native_weights.tolist()==[[4,4,4,4,0],[4,4,4,4,0],[1,1,1,1,0],[1,1,1,1,0]]
out['loss_weights']={'labels_text':'-12x followed by ignored position','weighted44':weighted_weights.tolist(),'native414':native_weights.tolist(),'mask_notes':'minus, both digits, immediate nondigit boundary; tool and numeric factors multiply; augmentation key excludes native multiplier.'}
# Verify validation calls objective without the new language-weight arguments.
main_ast=ast.parse((ROOT/'scripts/selftrained/train.py').read_text())
val_fn=next(x for x in main_ast.body if isinstance(x,ast.FunctionDef) and x.name=='validation_loss')
objective_calls=[x for x in ast.walk(val_fn) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='objective']
assert len(objective_calls)==1 and len(objective_calls[0].args)==6 and not objective_calls[0].keywords
out['validation_unweighted_call']=True
out['commands']=commands
(BASE/'cpu-probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['commands','native_local_integration']},ensure_ascii=False,indent=2))
print('Seven bounded real local wrapper attempts passed; uninterrupted 3 updates equals exact resume to 3 updates. Artifacts include actual command and child logs.')
