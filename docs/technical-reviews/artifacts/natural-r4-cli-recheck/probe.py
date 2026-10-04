from pathlib import Path
from unittest import mock
import ast,contextlib,difflib,hashlib,importlib,inspect,io,json,platform,sys
root=Path.cwd();sys.path.insert(0,str(root))
import torch
from scripts import natural_assistant as cli
from tiny_perceptron import natural_assistant as core
folder=root/'docs/technical-reviews/artifacts/natural-r4-cli-recheck'
old=(folder/'previous-reviewed-natural-cli.py').read_bytes();new=(root/'scripts/natural_assistant.py').read_bytes()
assert new==(folder/'current-natural-cli.py').read_bytes()
prior_scope={'__name__':'r4_previous_cli','__file__':str(root/'scripts/natural_assistant.py')}
exec(compile(old,str(folder/'previous-reviewed-natural-cli.py'),'exec'),prior_scope)
base=['chat','--output',str(root/'outputs/natural-r4-cli-recheck'),'--device','cpu','--dtype','float32','--user','typed-route-probe']
previous=prior_scope['parser']().parse_args(base);current=cli.parser().parse_args(base)
assert previous.max_new_tokens==current.max_new_tokens==384
assert previous.learning_rate==1e-4 and current.learning_rate==3e-5
fields=sorted(set(vars(previous))|set(vars(current)))
changed={k:{'previous':str(getattr(previous,k)),'current':str(getattr(current,k))} for k in fields if getattr(previous,k)!=getattr(current,k)}
assert set(changed)=={'learning_rate'}
override=cli.parser().parse_args(base+['--learning-rate','0.0001','--max-new-tokens','96'])
assert override.learning_rate==1e-4 and override.max_new_tokens==96
calls=[];models=[];core_model=object();processor=object();asr_model=object();asr_processor=object()
def load_core(options,adapter=None):models.append(core_model);calls.append({'event':'load_core','model':options.model,'revision':options.model_revision,'device':options.device,'adapter':str(adapter),'learning_rate_option':options.learning_rate,'generation_limit_option':options.max_new_tokens});return core_model,processor
def load_asr(options):calls.append({'event':'load_asr','model':options.asr_model,'revision':options.asr_revision});return asr_model,asr_processor
def transcribe(model,proc,path):assert model is asr_model and proc is asr_processor;calls.append({'event':'transcribe','path':str(path)});return {'transcript':'asr-route-probe'}
def generate(model,proc,row,data_root,options):assert model is core_model and proc is processor;calls.append({'event':'generate','user':row['user'],'same_core_object':True,'limit':options.max_new_tokens});return {'prediction':'stub-text-output'}
def run_train(options):calls.append({'event':'run_train_dispatch','learning_rate':options.learning_rate,'model':options.model,'lora_rank':options.lora_rank});return {'scope':'configuration-dispatch-only'}
patches={'load_core':load_core,'load_asr':load_asr,'transcribe':transcribe,'generate':generate,'asset_path':lambda audio,root:root/audio,'messages_for':lambda row,root:[{'role':'user','content':[{'type':'text','text':row['user']}]}],'write_json':lambda path,obj:None,'run_train':run_train}
with mock.patch.multiple(core,**patches):
 for args in [base, ['chat','--output',str(root/'outputs/natural-r4-cli-recheck'),'--device','cpu','--dtype','float32','--audio','stub-audio.wav'],['train','--output',str(root/'outputs/natural-r4-cli-recheck'),'--device','cpu','--dtype','float32','--manifest','docs/natural-assistant/manifest.json']]:
  with mock.patch.object(sys,'argv',['scripts/natural_assistant.py']+args),contextlib.redirect_stdout(io.StringIO()):cli.main()
assert len(models)==2 and all(m is core_model for m in models)
users=[c['user'] for c in calls if c['event']=='generate'];assert users==['typed-route-probe','asr-route-probe']
assert next(c for c in calls if c['event']=='run_train_dispatch')['learning_rate']==3e-5
train=ast.parse(inspect.getsource(core.run_train));generation=ast.parse(inspect.getsource(core.generate))
optimizer=[n for n in ast.walk(train) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='AdamW']
assert len(optimizer)==1
lr=next(k.value for k in optimizer[0].keywords if k.arg=='lr');assert isinstance(lr,ast.Attribute) and lr.attr=='learning_rate' and isinstance(lr.value,ast.Name) and lr.value.id=='options'
assert not any(isinstance(n,ast.Attribute) and n.attr=='learning_rate' for n in ast.walk(generation))
record={'command':'.venv/bin/python docs/technical-reviews/artifacts/natural-r4-cli-recheck/probe.py','environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'CPU argument parsing and controlled dispatch; no pretrained inference'},'previous_cli_sha256':hashlib.sha256(old).hexdigest(),'current_cli_sha256':hashlib.sha256(new).hexdigest(),'changed_parser_defaults':changed,'unchanged_max_new_tokens':384,'explicit_overrides':{'learning_rate':override.learning_rate,'max_new_tokens':override.max_new_tokens},'calls':calls,'training_lr_source':'run_train AdamW(..., lr=options.learning_rate), lines477–481; resume contract compares learning_rate at lines503–513','generation_does_not_read_learning_rate':True,'scope':'Controlled call-routing probes with stub loaders/generation/transcription and static optimizer argument inspection. No training, no 2B loading/inference, no ASR decoding, no GPU, no test/quality outputs inspected. Historical CPU base observations remain historical evidence only.','result':'passed'}
(folder/'execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['changed_parser_defaults','unchanged_max_new_tokens','generation_does_not_read_learning_rate','result']},ensure_ascii=False))
