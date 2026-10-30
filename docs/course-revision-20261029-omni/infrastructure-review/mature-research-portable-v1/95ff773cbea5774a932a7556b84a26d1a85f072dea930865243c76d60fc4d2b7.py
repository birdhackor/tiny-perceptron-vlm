"""Real tiny random own parameters, count/conversion only; no forward or pretrained model."""
import copy,json,sys,types
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,'/workspace/work/omni-models-v1')
from scripts.omni_revision.research_model_files import component_parameter_dtypes
from scripts.omni_revision.research_measurements import parameter_precision
components=types.SimpleNamespace(**{name:torch.nn.Linear(2,2).to(torch.bfloat16) for name in ('thinker','talker','token2wav')});components.thinker.register_parameter('mixed_precision_fixture',torch.nn.Parameter(torch.zeros(3,dtype=torch.float32)));before=component_parameter_dtypes(components);components.token2wav.float();after=component_parameter_dtypes(components)
result={'parameter_numel_by_dtype_after_load':before,'parameter_numel_by_dtype_after_speech':after,'speech_outputs':[{'generated':True}]};parameter_precision(result)
assert before['thinker']=={'torch.bfloat16':6,'torch.float32':3} and before['token2wav']=={'torch.bfloat16':6} and after['token2wav']=={'torch.float32':6}
cases=[]
for mode in ('wrong_after_dtype','changed_parameter_numel','quantized_dtype','missing_component'):
 value=copy.deepcopy(result)
 if mode=='wrong_after_dtype':value['parameter_numel_by_dtype_after_speech']['token2wav']={'torch.bfloat16':6}
 if mode=='changed_parameter_numel':value['parameter_numel_by_dtype_after_speech']['thinker']['torch.float32']+=1
 if mode=='quantized_dtype':value['parameter_numel_by_dtype_after_load']['thinker']={'torch.int8':9}
 if mode=='missing_component':value['parameter_numel_by_dtype_after_load'].pop('talker')
 try:parameter_precision(value);accepted=True;error=None
 except ValueError as e:accepted=False;error=str(e)
 assert not accepted;cases.append({'case':mode,'accepted':accepted,'reason':error})
no_generation={'parameter_numel_by_dtype_after_load':before,'parameter_numel_by_dtype_after_speech':before,'speech_outputs':[{'generated':False}]};parameter_precision(no_generation)
record={'scope':'Actual tiny torch.nn parameters and dtype conversion, mixed dtype counts and validator; no forward, neural prediction, pretrained model or provider call. This is not actual Qwen dtype observation.','actual_before':before,'actual_after':after,'negative_controls':cases,'no_completed_speech_keeps_observed_BF16_allowed':True,'real_pretrained_or_neural_forward_or_final_or_paid_calls':0}
p=BASE/'current-v3-precision-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
