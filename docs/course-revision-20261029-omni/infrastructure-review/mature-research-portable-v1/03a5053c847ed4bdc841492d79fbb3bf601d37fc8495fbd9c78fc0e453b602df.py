"""Run the actual Qwen research loop against explicit non-neural model/IO fixtures."""
import json,os,sys,types
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'original-source';sys.path.insert(0,str(SOURCE))
# soundfile is a worker dependency; all media reads here are explicit generated fixture arrays.
sf=types.ModuleType('soundfile');sf.read=lambda *a,**k:(np.zeros(64000,dtype=np.float32),16000)
with patch.dict(sys.modules,{'soundfile':sf}):
 from scripts.omni_revision import qwen_research as q
root=BASE/'probes/original-qwen-failures';root.mkdir(parents=True,exist_ok=False);(root/'cases').mkdir();(root/'cases/cases.json').write_bytes(Path('/workspace/work/omni-mature-payload-api-preflight-v1/qwen25/extracted/cases/cases.json').read_bytes())
signal_fixture=types.ModuleType('scipy.signal');signal_fixture.resample_poly=lambda values,*args:values
sys.modules['scipy.signal']=signal_fixture
loads=[]
class NoNeuralModel:
 has_talker=True
 dtype=torch.bfloat16
 def eval(self):return self
 def to(self,*args):return self
class Loader:
 @classmethod
 def from_pretrained(cls,*args,**kwargs):loads.append({'class':cls.__name__,'model':args[0],'kwargs':{k:str(v) for k,v in kwargs.items()}});return NoNeuralModel()
tfm=types.ModuleType('transformers')
for n in ('Qwen2_5OmniForConditionalGeneration','Qwen2_5OmniProcessor','WhisperForConditionalGeneration','WhisperProcessor'):setattr(tfm,n,type(n,(Loader,),{}))
calls=[]
def infer(*args,**kwargs):
 calls.append({'speech':kwargs.get('speech',False),'frames':kwargs.get('frames') is not None,'audio':kwargs.get('audio') is not None,'seed':kwargs['seed']});raise RuntimeError('explicit review-only generation-failure fixture, no neural model exists')
class Frame:
 size=(112,112)
 def convert(self,*a):return self
old=Path.cwd()
try:
 os.chdir(root)
 with patch.dict(sys.modules,{'transformers':tfm}),patch.object(q,'infer',side_effect=infer),patch.object(q.Image,'open',return_value=Frame()),patch.object(q.torch.cuda,'empty_cache'),patch.object(q.torch.cuda,'max_memory_allocated',return_value=0):q.main()
finally:os.chdir(old)
result=json.loads((root/'result/research-result.json').read_bytes());raw=(root/'result/qwen-raw-predictions.jsonl').read_text().splitlines();speech=(root/'result/qwen-speech-raw.jsonl').read_text().splitlines();proxy=(root/'result/qwen-asr-raw.jsonl').read_text().splitlines()
assert len(calls)==104 and len(raw)==96 and len(speech)==len(proxy)==8
assert result['audio_text_counts']=={'families':24,'both_correct':0} and result['AV_counts']=={'families':24,'both_correct':0}
assert result['speech_counts']['ASR_proxy_correct_out_of_all8']==0 and result['speech_count']==8
# Actual infer() options are separately checked using only an in-memory fake generate implementation.
class Batch(dict):
 def to(self,*args):return self
class Tokenizer:
 def decode(self,*args,**kwargs):return 'review-fixture-text'
processor=types.SimpleNamespace(tokenizer=Tokenizer());options=[]
class GenerateFixture:
 dtype=torch.bfloat16
 def generate(self,**kwargs):
  options.append({k:v for k,v in kwargs.items() if not torch.is_tensor(v)})
  tokens=torch.tensor([[1,2,3,4]])
  return (tokens,torch.zeros(2400)) if kwargs['return_audio'] else tokens
with patch.object(q,'make_inputs',return_value=Batch(input_ids=torch.tensor([[1,2]]))),patch.object(q.torch.cuda,'synchronize'):
 q.infer(GenerateFixture(),processor,text='review text',speech=False)
 q.infer(GenerateFixture(),processor,text='review word',speech=True)
 q.infer(GenerateFixture(),processor,text='review AV',audio=np.zeros(64000),frames=[Frame()]*8,speech=False)
record={'scope':'Actual main() loops and infer() options, but every model constructor/generate and media read is an explicit no-neural fixture. This is not model research output or final prediction evidence.','fixed_outer_Qwen_attempts':len(calls),'audio_text_attempts':sum(not x['speech'] and not x['frames'] for x in calls),'AV_attempts':sum(x['frames'] for x in calls),'speech_attempts':sum(x['speech'] for x in calls),'raw_rows':len(raw),'speech_rows':len(speech),'ASR_rows':len(proxy),'all_failures_full_n':{'audio_text':result['audio_text_counts'],'AV':result['AV_counts'],'speech':result['speech_counts']},'measurement_completed_flag':result['completed'],'capability_success_claimed':False,'generate_option_controls':options,'real_pretrained_or_neural_or_final_or_paid_calls':0}
p=BASE/'original-qwen-failures-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
