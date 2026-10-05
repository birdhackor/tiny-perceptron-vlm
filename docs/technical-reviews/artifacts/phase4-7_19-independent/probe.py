"""Bounded CPU contract checks; no pretrained model, dataset, training, or weights saved."""
import os,sys,ast,json,math,hashlib,platform,importlib.metadata,typing
from pathlib import Path
from abc import ABC
from types import SimpleNamespace
A=Path(__file__).resolve().parent; ROOT=A.parents[3]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from tiny_perceptron.model import TinyLM,ModelConfig,generate
from tiny_perceptron.data import ByteTokenizer,render_chat
from tiny_perceptron.tokenization import generation_report
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1); torch.set_default_device('cpu');torch.manual_seed(19)
env={'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__,'torch_git':torch.version.git_version,'cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'device':'cpu','threads':str(torch.get_num_threads()),'cwd':str(Path.cwd())}
for name in ('transformers','vllm','tokenizers'):
 try:env[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:env[name]='not installed; original selected source fragments executed only' if name!='tokenizers' else 'not installed'
(A/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2)+'\n')
results={}
# Each displayed slash-separated toy item is exactly one stipulated token, not one real tokenizer token.
u,a,e=3,4,2;fa,fb,fc,fd,fe,ff,fg=range(8,15)
rows=[[u,fa,e],[a,fb,e],[u,fc,fd,fe,e],[a]]
prompt=torch.tensor([sum(rows,[])],dtype=torch.long)
assert [len(r) for r in rows]==[3,3,5,1] and prompt.shape==(1,12)
assert 16-prompt.shape[1]==4
results['toy_budget']={'row_lengths':[len(r) for r in rows],'P':12,'C':16,'remaining':4,'G4_valid':12+4<=16,'G5_valid':12+5<=16,'unit':'stipulated toy tokens; sequence axis 1 of [batch, sequence]','tolerance':'integer equality'}
class Scripted(nn.Module):
 def __init__(self,candidates,capacity=16):
  super().__init__();self.config=SimpleNamespace(max_length=capacity);self.candidates=candidates;self.calls=[]
 def forward(self,ids,cache=None):
  offset=0 if cache is None else cache[0][0].shape[2]
  self.calls.append({'input_shape':list(ids.shape),'cache_offset':offset})
  assert offset+ids.shape[1]<=self.config.max_length
  scores=torch.full((*ids.shape,16),-100.)
  scores[:,-1,self.candidates[len(self.calls)-1]]=100.
  kv=torch.zeros((ids.shape[0],1,offset+ids.shape[1],1))
  return {'logits':scores,'cache':[(kv,kv)]}
for name,candidates,budget,expected in [('full',[ff,fg,e,fa],4,[ff,fg,e]),('budget2',[ff,fg,e],2,[ff,fg]),('early_eos',[e,ff],4,[e]),('partial_eos',[ff,e,fg],4,[ff,e]),('exact3',[ff,fg,e],3,[ff,fg,e]),('capacity_clipping',[ff,fg,ff,fg,ff],5,[ff,fg,ff,fg])]:
 for cached in (False,True):
  model=Scripted(candidates);model.train();out=generate(model,prompt,max_new_tokens=budget,eos_id=e,use_cache=cached)
  actual=out[0,12:].tolist();assert actual==expected and model.training
  assert out.shape[1]<=16 and out[0,:12].tolist()==prompt[0].tolist()
  assert 0 not in actual # unused capacity was not turned into PAD
  content=[v for v in actual if v!=e];complete=content==[ff,fg]
  stop='eos' if actual and actual[-1]==e else 'capacity' if out.shape[1]==16 else 'new_token_budget'
  results[name+('_cache' if cached else '_full')]={'generated':actual,'new_tokens':len(actual),'total_tokens':out.shape[1],'stop':stop,'answer_complete':complete,'calls':model.calls}
# Real tokenizer: model-specific serialization is not the toy 12. No tokenization model is downloaded.
tok=ByteTokenizer();actual_prompt=[tok.bos_id]
for role,text in [('user','店名是小小書店。'),('assistant','好。'),('user','現在只答店名。')]:
 actual_prompt += [getattr(tok,role+'_id'),*tok.encode(text),tok.eos_id]
actual_prompt += [tok.assistant_id]
assert len(actual_prompt)!=12
x,y=render_chat([{'role':'user','content':'Q'},{'role':'assistant','content':'A'}]);first=(y!=-100).nonzero()[0].item()
assert x[first].item()==tok.assistant_id and y[first].item()==tok.encode('A')[0] and y[-1].item()==e
reports={}
for name,ids in [('full',tok.encode('小小書店')+[e]),('no_eos',tok.encode('小小書店')),('partial',tok.encode('小小')+[e]),('early',[e])]:
 r=generation_report(tok,ids);assert r['generated_ids']==ids;reports[name]=r
assert reports['full']['answer']==reports['no_eos']['answer']=='小小書店'
assert reports['full']['eos'] and not reports['no_eos']['eos']
assert reports['partial']['answer']=='小小' and reports['early']['answer']==''
results['repo_serialization']={'byte_prompt_length':len(actual_prompt),'byte_prompt_ids':actual_prompt,'first_answer_position':first,'X_first':x[first].item(),'Y_first':y[first].item(),'last_target':y[-1].item(),'reports':reports,'scope':'ByteTokenizer is one real repository contract, not a count for a different model.'}
# Actual TinyLM tests: sequence-length axis is KV axis 2; cached path retains all earlier K/V.
real=TinyLM(ModelConfig(vocab_size=16,width=8,layers=1,heads=2,max_length=16));real.eval()
with torch.no_grad():
 full=real(prompt);prefill=real(prompt[:,:8]);suffix=real(prompt[:,8:],cache=prefill['cache'])
 delta=(full['logits'][:,8:]-suffix['logits']).abs().max().item()
 assert torch.allclose(full['logits'][:,8:],suffix['logits'],atol=1e-6,rtol=1e-5)
 off=generate(real,prompt,max_new_tokens=5,eos_id=99,use_cache=False)
 on=generate(real,prompt,max_new_tokens=5,eos_id=99,use_cache=True)
 assert torch.equal(off,on) and off.shape==(1,16)
 exact=real(off);assert exact['cache'][0][0].shape[2]==16
 for label,fn in [('uncached17',lambda:real(torch.ones((1,17),dtype=torch.long))),('cached17',lambda:real(torch.ones((1,1),dtype=torch.long),cache=exact['cache']))]:
  try:fn()
  except ValueError as exc:results[label]={'exception':str(exc)}
  else:raise AssertionError(label+' should reject overlong forward')
 no_room=Scripted([ff]);at_capacity=generate(no_room,off,max_new_tokens=1)
 assert torch.equal(at_capacity,off) and not no_room.calls
results['repo_cache']={'KV_prefill_shape':list(prefill['cache'][0][0].shape),'KV_suffix_shape':list(suffix['cache'][0][0].shape),'KV_exact_capacity_shape':list(exact['cache'][0][0].shape),'logit_max_abs_difference':delta,'tolerance':'atol=1e-6 rtol=1e-5; generated ids exact','cached_and_uncached_ids_equal':True,'output_total_length':16,'P_equals_C_forward_calls':len(no_room.calls),'scope':'Single row, no padding or packing, manual attention, one seeded tiny random CPU model; no training or quality evidence.'}
# Execute exact original official class/function bodies with minimal dependency globals.
fragments=json.loads((A/'sources/executed-original-fragments.json').read_text())
notes=[]
class Logger:
 def warning(self,msg):notes.append(msg)
 def warning_once(self,msg):notes.append(msg)
g={'torch':torch,'math':math,'ABC':ABC,'Optional':typing.Optional,'Union':typing.Union,'logger':Logger(),'add_start_docstrings':lambda *args:lambda obj:obj,'STOPPING_CRITERIA_INPUTS_DOCSTRING':'','LOGITS_PROCESSOR_INPUTS_DOCSTRING':'','LogitsProcessor':object,'is_torch_greater_or_equal_than_2_4':True}
for f in fragments:
 raw=f['raw_code'];assert hashlib.sha256(raw.encode()).hexdigest()==f['sha256']
 # Exact indented method bytes are placed into an explicit review-only class; no function body is changed.
 if f['name']=='_prepare_generated_length':raw='class ReviewOnlyGenerationMixin:\n'+raw
 exec(compile(raw,f['source_file']+':original-lines-'+str(f['original_start_line']),'exec'),g)
EOS=g['EosTokenCriteria'](e);MAX=g['MaxLengthCriteria'](16)
assert not EOS(prompt,None).item() # historical EOS is not last token
assert not EOS(torch.cat((prompt,torch.tensor([[ff]])),1),None).item()
assert EOS(torch.cat((prompt,torch.tensor([[e]])),1),None).item()
assert not MAX(torch.ones((1,15),dtype=torch.long),None).item() and MAX(torch.ones((1,16),dtype=torch.long),None).item()
lengths=[]
for budget in (4,5):
 cfg=SimpleNamespace(max_length=16,max_new_tokens=budget,min_length=0,min_new_tokens=None)
 cfg=g['ReviewOnlyGenerationMixin']()._prepare_generated_length(cfg,False,True,'input_ids',12,prompt)
 assert cfg.max_length==12+budget;lengths.append({'G':budget,'computed_max_length':cfg.max_length})
forced=g['ForcedEOSTokenLogitsProcessor'](16,e)
scores=torch.zeros((1,16));scores[0,ff]=10
ordinary=scores.argmax(-1).item();forced15=forced(torch.ones((1,15),dtype=torch.long),scores).argmax(-1).item()
assert ordinary==ff and forced15==e and forced(torch.ones((1,14),dtype=torch.long),scores).argmax(-1).item()==ff
results['official_fragments']={'EOS_only_last_token':True,'MaxLength_15_stop':False,'MaxLength_16_stop':True,'prepared_lengths':lengths,'length_warnings':notes,'ordinary_argmax':ordinary,'forced_final_argmax':forced15,'scope':'Exact selected Transformers v4.57.1 original bodies, documentation decorators replaced with identity, minimal globals; neither installed Transformers generate nor vLLM engine executed.'}
(A/'probe-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks':'all assertions passed','toy_counts':results['toy_budget'],'real_byte_prompt_length':len(actual_prompt),'cache':results['repo_cache'],'official':results['official_fragments']},ensure_ascii=False,indent=2))
