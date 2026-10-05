"""Execute original AST-selected loader functions against bounded CPU doubles, not real pretrained weights."""
from pathlib import Path
import ast,sys,json,types,platform
from types import SimpleNamespace
import torch
from torch import nn
b=Path(__file__).resolve().parents[1]
p=b/'inputs/tiny_perceptron/natural_assistant.py';tree=ast.parse(p.read_text());names={'load_core','parameter_counts','load_asr'}
selected=ast.Module(body=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LORA_TARGETS' for t in n.targets))],type_ignores=[])
ns={'torch':torch};exec(compile(selected,str(p),'exec'),ns)
import re
assert re.fullmatch(ns['LORA_TARGETS'],'model.language_model.layers.0.self_attn.q_proj')
events=[]
class TinyModel(nn.Module):
 def __init__(self):
  super().__init__();self.weight=nn.Parameter(torch.ones(2,2));self.config=SimpleNamespace(use_cache=True)
 def to(self,device):events.append(['to',device]);return super().to(device)
 def requires_grad_(self,value=True):events.append(['requires_grad',value]);return super().requires_grad_(value)
 def gradient_checkpointing_enable(self,**kw):events.append(['checkpointing',kw])
class Processor:
 @classmethod
 def from_pretrained(cls,name,**kw):events.append(['processor',name,kw]);return object()
class Factory:
 @classmethod
 def from_pretrained(cls,name,**kw):events.append(['core',name,{k:str(v) for k,v in kw.items()}]);return TinyModel()
class WhisperFactory(Factory):
 @classmethod
 def from_pretrained(cls,name,**kw):events.append(['asr',name,{k:str(v) for k,v in kw.items()}]);return TinyModel()
class PeftModel:
 @classmethod
 def from_pretrained(cls,model,adapter,is_trainable=False):
  assert model.weight.requires_grad is False
  events.append(['adapter',adapter,is_trainable]);model.register_parameter('lora',nn.Parameter(torch.zeros(2),requires_grad=is_trainable));return model
class LoraConfig:
 def __init__(self,**kw):self.values=kw
class TaskType:CAUSAL_LM='CAUSAL_LM'
def get_peft_model(model,config):
 assert model.weight.requires_grad is False
 events.append(['new_lora',config.values]);model.register_parameter('lora',nn.Parameter(torch.zeros(2)));return model
sys.modules['transformers']=SimpleNamespace(AutoProcessor=Processor,Qwen3VLForConditionalGeneration=Factory,WhisperForConditionalGeneration=WhisperFactory,WhisperProcessor=Processor)
sys.modules['peft']=SimpleNamespace(PeftModel=PeftModel,LoraConfig=LoraConfig,TaskType=TaskType,get_peft_model=get_peft_model)
manifest=json.loads((b/'inputs/docs/natural-assistant/v4/public-release.json').read_bytes());q=manifest['base_model'];a=manifest['asr_model'];o=SimpleNamespace(dtype='float32',device='cpu',model=q['repo'],model_revision=q['revision'],cache_dir='not-read',local_files_only=True,min_pixels=65536,max_pixels=524288,lora_rank=8,asr_model=a['repo'],asr_revision=a['revision'])
for label,adapter,train in [('base',None,False),('saved_adapter','local-adapter',False),('saved_adapter_train','local-adapter',True),('new_adapter_train',None,True)]:
 events.clear();model,_=ns['load_core'](o,adapter=adapter,train=train);counts=ns['parameter_counts'](model)
 assert events[1][0]=='core' and events[2]==['to','cpu'] and events[3]==['requires_grad',False]
 assert counts['total_parameters']==(4 if label=='base' else 6)
 assert counts['trainable_parameters']==(2 if train else 0)
 assert model.training==train
 if train:assert model.config.use_cache is False and events[-1][0]=='checkpointing'
 print(json.dumps({'case':label,'events':events,'toy_parameter_counts':counts},ensure_ascii=False))
events.clear();o.dtype='bfloat16'
try:ns['load_core'](o)
except ValueError as e:assert not events;print('cpu_non_float32_rejected_before_load',str(e))
else:raise AssertionError('CPU non-float32 route unexpectedly accepted')
events.clear();model,_=ns['load_asr'](o);assert events[-1][0]=='asr' and events[-1][2]['dtype']=='torch.float32';assert not model.training
print(json.dumps({'case':'asr_separate_loader','events':events},ensure_ascii=False))
print(json.dumps({'python':sys.version,'torch':torch.__version__,'device':'cpu','transformers':'not installed; fake module only for original loader contract','peft':'not installed; fake module only for original loader contract','platform':platform.platform(),'actual_model_inference':'not executed'},ensure_ascii=False))
